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


def cockcroft_gault(scr_mgdl, age, weight_kg, sex):
    """CrCl in mL/min. Uses actual body weight, matching MedCalc's entities."""
    coef = 1.0 if str(sex).lower().startswith("m") else 0.85
    return ((140 - age) * weight_kg * coef) / (scr_mgdl * 72)


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
        return cockcroft_gault(scr_mgdl, age, float(ents["weight"][0]), sex)
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
