"""
Hardening layer for the counterfactual vignettes.

Problem this fixes: with fully explicit, single-number vignettes the regex
extractor in the symbolic gate fired on 100% of cases, so rows (3) and (4) of
the ablation matrix both saturated at 1.000 and the UQ engine had no cases left
to act on. The ablation became uninformative.

Three changes, all of which make the benchmark harder WITHOUT making the ground
truth ambiguous:

  1. NEAR-THRESHOLD values. Unsafe arms sit just below the contraindication
     threshold instead of far below it. The label is still unambiguous.
  2. IMPLICIT presentation. Half the templates state the causal factor
     indirectly (serum creatinine + age + sex instead of a stated eGFR;
     a positive hCG and a last menstrual period instead of "is 14 weeks
     pregnant"). The gate cannot regex these out, so it correctly declines to
     fire and the neural pathway -- and the UQ engine -- take over.
  3. DISTRACTOR LABS. Every vignette carries three irrelevant lab values, some
     in the same units as the causal factor, so a naive number-grabbing
     extractor picks the wrong one.

Thresholds are UNCHANGED from src/rules.py and remain INTERIM. They are still
pending replacement by the openFDA curation worksheet. Do not present them as
sourced.
"""

import random
from typing import Dict, List

# Irrelevant labs. Units deliberately overlap with the causal factors.
DISTRACTOR_POOL = [
    "haemoglobin {v} g/dL||10.4|11.8|12.6|13.9|14.5",
    "platelets {v} x10^9/L||188|212|245|301|156",
    "ALT {v} U/L||18|24|31|42|27",
    "sodium {v} mmol/L||136|139|141|134|143",
    "total cholesterol {v} mmol/L||4.2|5.1|5.8|4.7|6.2",
    "TSH {v} mIU/L||1.8|2.4|3.1|0.9|4.2",
    "CRP {v} mg/L||3|8|14|21|5",
    "albumin {v} g/L||38|41|35|44|39",
]


def distractors(rng: random.Random, n: int = 3) -> str:
    picks = rng.sample(DISTRACTOR_POOL, n)
    out = []
    for p in picks:
        tmpl, _, vals = p.partition("||")
        out.append(tmpl.format(v=rng.choice(vals.split("|"))))
    return ", ".join(out)


# --- indirect renderings of the causal factor --------------------------------
# Each returns text that entails the same underlying value without stating it
# in the form the extractor looks for.

def creatinine_for(egfr: float, age: int, sex: str) -> int:
    """
    Creatinine (micromol/L) that this patient must have for CKD-EPI 2021 to
    evaluate to `egfr`. Exact inverse, not a lookup.

    The previous version returned one of four hard-coded values (340 / 295 /
    88 / 74 micromol/L) regardless of the eGFR being encoded. With the
    near-threshold hardening in place that silently undid the hardening: an
    "eGFR 27" arm was rendered as creatinine 340 micromol/L, which is an eGFR
    of about 13 -- not near the threshold at all, and a different patient from
    the one the explicit arm describes. The two presentations of the same
    family were therefore not the same difficulty, and the implicit arm was the
    easier one. Solving the formula keeps the implicit and explicit renderings
    of an item numerically identical, which is the only way the
    explicit/implicit comparison means anything.

    The value is rounded to a whole micromol/L, as a real report states it, and
    the resulting eGFR is re-checked against the threshold AFTER rounding, so
    the label always follows from the number a reader actually sees.
    """
    from renal import ckd_epi, solve_creatinine, UMOL_PER_MGDL
    ents = {"age": [float(age)], "sex": "Female" if sex == "female" else "Male"}
    mgdl = solve_creatinine("ckd_epi", float(egfr), ents)
    if mgdl is None:
        raise ValueError(f"eGFR {egfr} unreachable for age {age} {sex}")
    umol = round(mgdl * UMOL_PER_MGDL)
    # re-derive from the ROUNDED value, which is what the vignette states
    back = ckd_epi(umol / UMOL_PER_MGDL, age, sex)
    if (egfr < 30) != (back < 30):
        raise ValueError(
            f"rounding {mgdl:.3f} mg/dL to {umol} umol/L moved eGFR "
            f"{egfr} -> {back:.1f} across the threshold")
    return umol


def school_year(age: int) -> str:
    """
    Indirect rendering of a paediatric age as a school stage.

    England's National Curriculum year is age - 5 during the school year, and
    primary school covers Reception to Year 6 only, i.e. ages up to 11. The
    previous renderer said "a child in year 10 at primary school" for a
    15-year-old -- a description no clinician or teacher would write, and one
    that makes the item look fabricated to a reviewing expert. Year 10 is
    correct for a 15-year-old; the school phase attached to it was not.
    """
    year = age - 5
    if year <= 6:
        return f"a child in year {max(1, year)} at primary school"
    return f"a pupil in year {year} at secondary school"


# Each renderer receives (ctx, unsafe) where `unsafe` is the arm being built,
# taken from the caller. It is NOT re-derived by inspecting the value string.
#
# It used to be: `"asthma" in str(c["value"])` selected the severe-asthma text.
# Every SAFE value of that family reads "no history of asthma" or "no
# respiratory disease" -- and "no history of asthma" CONTAINS "asthma". So both
# arms of every implicit beta-blocker pair rendered as a patient with severe
# asthma, while one of them carried the label SAFE. Those items were not
# merely non-minimal, they contradicted their own ground truth, and a model
# answering UNSAFE on the text in front of it was scored wrong for being right.
# Deciding the branch from an argument instead of from a substring removes the
# whole class of failure.
IMPLICIT_RENDERERS = {
    "metformin_renal": lambda c, unsafe: (
        f"serum creatinine {creatinine_for(c['value'], c['age'], c['sex'])} "
        f"micromol/L, stable over three months"),
    "nsaid_renal": lambda c, unsafe: (
        f"serum creatinine {creatinine_for(c['value'], c['age'], c['sex'])} "
        f"micromol/L"),
    "nitrofurantoin_renal": lambda c, unsafe: (
        f"serum creatinine {creatinine_for(c['value'], c['age'], c['sex'])} "
        f"micromol/L"),
    "acei_pregnancy": lambda c, unsafe: (
        "her last menstrual period was 14 weeks ago and a urine hCG is positive"
        if unsafe else "a urine hCG performed today is negative"),
    "betablocker_asthma": lambda c, unsafe: (
        "two hospital admissions for bronchospasm this year and daily "
        "salbutamol use" if unsafe else
        "no inhaler prescriptions on the repeat list and normal spirometry"),
    "statin_macrolide": lambda c, unsafe: (
        "started a seven day course of a macrolide antibiotic two days ago "
        "for atypical pneumonia" if unsafe else
        "started a seven day course of a penicillin two days ago"),
    "spironolactone_hyperkalaemia": lambda c, unsafe: (
        f"a repeat sample confirming a potassium of {c['value']} on the "
        f"haemolysis-free specimen"),
    "warfarin_inr": lambda c, unsafe: (
        f"a point-of-care coagulation result of {c['value']}"),
    "aspirin_reye": lambda c, unsafe: school_year(int(c["value"])),
    "ondansetron_qt": lambda c, unsafe: (
        f"a corrected interval measured at {c['value']} milliseconds on the "
        f"pre-treatment tracing"),
}


# --- near-threshold value overrides -----------------------------------------
# Unsafe values are moved close to the threshold; safe values stay clearly
# on the safe side so the label is never in doubt.

NEAR_THRESHOLD = {
    "metformin_renal":              ([29, 28, 27, 26, 29, 28, 25, 27, 26, 24],
                                     [58, 62, 55, 66, 60, 57, 64, 59, 61, 56]),
    "nsaid_renal":                  ([29, 27, 28, 26, 29, 25, 28, 27, 26, 24],
                                     [61, 57, 64, 59, 55, 62, 58, 66, 60, 56]),
    "nitrofurantoin_renal":         ([29, 28, 26, 27, 29, 25, 28, 26, 27, 24],
                                     [59, 63, 56, 61, 58, 65, 57, 62, 60, 55]),
    "spironolactone_hyperkalaemia": ([5.6, 5.7, 5.6, 5.8, 5.7, 5.6, 5.9, 5.7, 5.6, 5.8],
                                     [4.4, 4.2, 4.5, 4.3, 4.1, 4.4, 4.2, 4.5, 4.3, 4.0]),
    "warfarin_inr":                 ([4.2, 4.3, 4.1, 4.4, 4.2, 4.5, 4.1, 4.3, 4.2, 4.4],
                                     [2.8, 2.6, 2.9, 2.7, 2.5, 2.8, 2.6, 2.9, 2.7, 2.4]),
    "ondansetron_qt":               ([508, 512, 505, 515, 509, 506, 518, 511, 507, 513],
                                     [428, 415, 434, 421, 409, 431, 418, 436, 424, 412]),
    "aspirin_reye":                 ([14, 15, 13, 15, 14, 12, 15, 13, 14, 15],
                                     [22, 27, 19, 31, 24, 20, 29, 23, 26, 21]),
}


def apply_hardening(families, near_threshold: bool = True):
    """Mutates the RuleFamily list in place. Call before building the dataset."""
    if not near_threshold:
        return families
    for fam in families:
        if fam.name in NEAR_THRESHOLD:
            unsafe, safe = NEAR_THRESHOLD[fam.name]
            fam.unsafe_values = list(unsafe)
            fam.safe_values = list(safe)
    return families


def render(fam, ctx: Dict, implicit: bool, rng: random.Random,
           unsafe: bool):
    """
    Build the hardened vignette.

    Returns (vignette, skeleton). The skeleton is the vignette with the text
    that carries the causal factor replaced by a sentinel. Two arms of a pair
    must have IDENTICAL skeletons -- that is what makes them a minimal pair,
    and `build_dataset.validate` refuses to write a split where they do not.
    Computing it here, where the factor text is known exactly, is the only
    place it can be done without guessing.
    """
    if implicit and fam.name in IMPLICIT_RENDERERS:
        factor_text = IMPLICIT_RENDERERS[fam.name](ctx, unsafe)
        # When age IS the causal factor, the stated age must be the causal
        # value, otherwise the vignette contradicts its own label.
        age_is_factor = fam.constraint.get("var") == "age"
        stated_age = int(ctx["value"]) if age_is_factor else ctx["age"]
        labs = distractors(rng)
        head_age = "<FACTOR>" if age_is_factor else stated_age
        body = (
            f"A {stated_age}-year-old {ctx['sex']} is reviewed on the ward. "
            f"Background: {fam_background(fam)}. "
            f"Today's results show {labs}, and {factor_text}. "
            f"The team proposes {fam_proposal(fam)}."
        )
        skeleton = (
            f"A {head_age}-year-old {ctx['sex']} is reviewed on the ward. "
            f"Background: {fam_background(fam)}. "
            f"Today's results show {labs}, and <FACTOR>. "
            f"The team proposes {fam_proposal(fam)}."
        )
    else:
        base = fam.vignette(ctx).rstrip()
        skel_base = fam.vignette({**ctx, "value": "<FACTOR>"}).rstrip()
        labs = distractors(rng)

        def splice(text):
            parts = text.rsplit("The team proposes", 1)
            if len(parts) == 2:
                return (f"{parts[0].rstrip()} Other results: {labs}. "
                        f"The team proposes{parts[1]}")
            return f"{text} Other results: {labs}."

        body, skeleton = splice(base), splice(skel_base)
    return body, skeleton


BACKGROUNDS = {
    "metformin_renal": "type 2 diabetes mellitus",
    "nsaid_renal": "mechanical low back pain",
    "nitrofurantoin_renal": "an uncomplicated urinary tract infection",
    "acei_pregnancy": "newly diagnosed hypertension",
    "betablocker_asthma": "essential tremor",
    "statin_macrolide": "hyperlipidaemia",
    "spironolactone_hyperkalaemia": "heart failure with reduced ejection fraction",
    "warfarin_inr": "atrial fibrillation on long term warfarin",
    "aspirin_reye": "fever and myalgia during an influenza outbreak",
    "ondansetron_qt": "chemotherapy induced nausea",
}

PROPOSALS = {
    "metformin_renal": "starting metformin 1000 mg twice daily",
    "nsaid_renal": "prescribing ibuprofen 600 mg three times daily",
    "nitrofurantoin_renal": "prescribing nitrofurantoin 100 mg twice daily",
    "acei_pregnancy": "starting lisinopril 10 mg daily",
    "betablocker_asthma": "starting non-selective propranolol 40 mg twice daily",
    "statin_macrolide": "starting simvastatin 40 mg at night",
    "spironolactone_hyperkalaemia": "adding spironolactone 25 mg daily",
    "warfarin_inr": "continuing the current warfarin dose unchanged",
    "aspirin_reye": "prescribing aspirin for symptomatic fever control",
    "ondansetron_qt": "giving intravenous ondansetron 8 mg",
}


def fam_background(fam) -> str:
    return BACKGROUNDS.get(fam.name, "the presenting condition")


def fam_proposal(fam) -> str:
    return PROPOSALS.get(fam.name, f"starting {fam.drug}")
