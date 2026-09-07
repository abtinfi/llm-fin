"""
Renal-function formulas, and the inverse solve used to build counterfactuals.

These are implemented rather than taken on trust because the whole benchmark
rests on them twice over: once to assign the ground-truth label, and once to
compute the creatinine value that moves a real patient across the threshold.
`validate()` checks every implementation against MedCalc-Bench's own
`Ground Truth Answer` / `Lower Limit` / `Upper Limit` columns before any data is
generated. If a formula does not reproduce their numbers, the build aborts --
a silently wrong formula would produce a benchmark whose labels are wrong in a
way no downstream check could catch.

Units: MedCalc-Bench states creatinine in either mg/dL or µmol/L. Conversion is
mg/dL = µmol/L / 88.4.
"""

import math

UMOL_PER_MGDL = 88.4


def to_mgdl(value, unit):
    u = (unit or "").lower().replace("μ", "µ")
    if "µmol" in u or "umol" in u:
        return value / UMOL_PER_MGDL
    if "mg/dl" in u or "mg/dl" == u:
        return value
    # MedCalc uses only these two; anything else must not be guessed at.
    raise ValueError(f"unhandled creatinine unit {unit!r}")


def ideal_body_weight(height_cm, sex):
    """Devine IBW in kg."""
    inches = height_cm / 2.54
    base = 50.0 if str(sex).lower().startswith("m") else 45.5
    return base + 2.3 * (inches - 60.0)


def cg_dosing_weight(weight_kg, height_cm, sex):
    """
    The weight Cockcroft-Gault is actually evaluated at.

    Plain actual body weight reproduced only 71 of 151 MedCalc-Bench CG rows.
    MedCalc (and MDCalc, and routine practice) selects the weight by BMI, which
    its own `Ground Truth Explanation` column spells out:

        BMI < 18.5   underweight  -> actual body weight
        18.5-24.9    normal       -> min(ideal, actual)
        >= 25        over/obese   -> adjusted = IBW + 0.4 * (ABW - IBW)

    Implemented from that text rather than from memory, and validated against
    the Lower/Upper limit columns before any item is built.
    """
    ibw = ideal_body_weight(height_cm, sex)
    bmi = weight_kg / ((height_cm / 100.0) ** 2)
    if bmi < 18.5:
        return weight_kg
    if bmi < 25.0:
        return min(ibw, weight_kg)
    return ibw + 0.4 * (weight_kg - ibw)


def cockcroft_gault(scr_mgdl, age, weight_kg, sex, height_cm=None):
    """
    CrCl in mL/min.

    `height_cm` is optional only so the signature stays backward compatible;
    when it is given the BMI-selected dosing weight is used, which is what
    MedCalc-Bench's ground truth encodes.
    """
    coef = 1.0 if str(sex).lower().startswith("m") else 0.85
    w = (weight_kg if height_cm is None
         else cg_dosing_weight(weight_kg, height_cm, sex))
    return ((140 - age) * w * coef) / (scr_mgdl * 72)


def mdrd(scr_mgdl, age, sex, race=None):
    """GFR in mL/min/1.73m^2."""
    g = 0.742 if str(sex).lower().startswith("f") else 1.0
    r = 1.212 if race and "black" in str(race).lower() else 1.0
    return 175 * (scr_mgdl ** -1.154) * (age ** -0.203) * r * g


def ckd_epi(scr_mgdl, age, sex):
    """GFR in mL/min/1.73m^2. CKD-EPI 2021, race-free."""
    female = str(sex).lower().startswith("f")
    if female:
        A, B = 0.7, (-0.241 if scr_mgdl <= 0.7 else -1.2)
        g = 1.012
    else:
        A, B = 0.9, (-0.302 if scr_mgdl <= 0.9 else -1.2)
        g = 1.0
    return 142 * ((scr_mgdl / A) ** B) * (0.9938 ** age) * g


CALCULATORS = {
    "Creatinine Clearance (Cockcroft-Gault Equation)": "cockcroft_gault",
    "MDRD GFR Equation": "mdrd",
    "CKD-EPI Equations for Glomerular Filtration Rate": "ckd_epi",
}

# Unit of the OUTPUT, which decides which drug rule is clinically appropriate.
OUTPUT_UNIT = {
    "cockcroft_gault": "mL/min",
    "mdrd": "mL/min/1.73m2",
    "ckd_epi": "mL/min/1.73m2",
}


def compute(kind, scr_mgdl, ents):
    age = float(ents["age"][0])
    sex = ents.get("sex", "Male")
    if kind == "cockcroft_gault":
        h = ents.get("height")
        return cockcroft_gault(scr_mgdl, age, float(ents["weight"][0]), sex,
                               float(h[0]) if h else None)
    if kind == "mdrd":
        return mdrd(scr_mgdl, age, sex, ents.get("Race"))
    if kind == "ckd_epi":
        return ckd_epi(scr_mgdl, age, sex)
    raise ValueError(kind)


def solve_creatinine(kind, target, ents, lo=0.05, hi=60.0, tol=1e-9):
    """
    Creatinine (mg/dL) at which `kind` evaluates to `target`.

    Every one of these formulas is strictly decreasing in creatinine, so a
    bisection is exact and needs no derivative. Monotonicity is asserted rather
    than assumed -- CKD-EPI is piecewise and a sign error in the branch
    selection would otherwise pass unnoticed.
    """
    f_lo, f_hi = compute(kind, lo, ents), compute(kind, hi, ents)
    assert f_lo > f_hi, f"{kind} is not decreasing in creatinine"
    if not (f_hi <= target <= f_lo):
        return None                      # target unreachable for this patient
    for _ in range(200):
        mid = (lo + hi) / 2
        v = compute(kind, mid, ents)
        if abs(v - target) < tol:
            return mid
        if v > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# --------------------------------------------------------------------------
# Shared with build_medcalc.py and medcalc_gate.py. These live here rather than
# in either of those because both need them and they must not import each other.
# --------------------------------------------------------------------------

def qtc_bazett(qt_ms, hr):
    """Bazett-corrected QT in ms. QTc = QT / sqrt(RR), RR = 60/HR."""
    return qt_ms / math.sqrt(60.0 / hr)


def qtc_fridericia(qt_ms, hr):
    """QTc = QT / RR^(1/3)."""
    return qt_ms / ((60.0 / hr) ** (1.0 / 3.0))


def qtc_framingham(qt_ms, hr):
    """QTc = QT + 154 * (1 - RR)."""
    return qt_ms + 154.0 * (1.0 - 60.0 / hr)


def qtc_hodges(qt_ms, hr):
    """QTc = QT + 1.75 * (HR - 60)."""
    return qt_ms + 1.75 * (hr - 60.0)


def qtc_rautaharju(qt_ms, hr):
    """QTc = QT * (120 + HR) / 180."""
    return qt_ms * (120.0 + hr) / 180.0


# MedCalc-Bench ships five QT corrections over the same underlying quantity
# (QT interval and heart rate). Only Bazett was used until 2026-09-06, which
# left 400 of the 500 available notes on the floor -- and the QT family is the
# one carrying the Aim 3 result, so it is exactly the family that most needed
# the n.
#
# The label is assigned with BAZETT for every note, whichever correction the
# source row used, for the same reason build_renal labels everything with
# CKD-EPI: a benchmark whose ground truth depends on which row of the source
# file an item came from measures bookkeeping, not a clinical rule. The FDA
# ondansetron labelling and the 500 ms threshold are stated in Bazett terms.
# The row's own formula is still checked against MedCalc's ground truth, and
# that check is what certifies the note's QT and heart rate were extracted
# correctly.
QT_CALCULATORS = {
    "QTc Bazett Calculator":      "qtc_bazett",
    "QTc Fridericia Calculator":  "qtc_fridericia",
    "QTc Framingham Calculator":  "qtc_framingham",
    "QTc Hodges Calculator":      "qtc_hodges",
    "QTc Rautaharju Calculator":  "qtc_rautaharju",
}

_QT_FN = {
    "qtc_bazett": qtc_bazett,
    "qtc_fridericia": qtc_fridericia,
    "qtc_framingham": qtc_framingham,
    "qtc_hodges": qtc_hodges,
    "qtc_rautaharju": qtc_rautaharju,
}


def qtc(kind, qt_ms, hr):
    return _QT_FN[kind](qt_ms, hr)


# Physiological plausibility windows. Deliberately wide -- these exist to
# reject impossible source data, not to reject sick patients. Dialysis-level
# creatinines of 15 mg/dL and eGFRs of 3 are real and must survive.
PLAUSIBLE = {
    "creatinine_mgdl": (0.15, 25.0),
    "eGFR": (2.0, 200.0),
    "qt_ms": (200.0, 700.0),
    "heart_rate": (25.0, 220.0),
    "QTc": (250.0, 750.0),
    # added 2026-09-02 for build_mimic.py's potassium/INR rule families --
    # these are lab-panic-value bounds (transcription-error guards), not
    # clinical safety thresholds, same role as the ranges above.
    "potassium_mEqL": (1.5, 9.0),
    "inr": (0.5, 12.0),
}


def implausible(**kw):
    """Name of the first out-of-range quantity, or None if all are possible."""
    for k, v in kw.items():
        lo, hi = PLAUSIBLE[k]
        if not (lo <= v <= hi):
            return k
    return None
