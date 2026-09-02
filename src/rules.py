"""
Deterministic clinical contraindication rules.

Each rule family defines ONE causal factor. A counterfactual pair is produced by
flipping only that factor across a threshold, so the correct label is guaranteed
to flip too. Ground truth is generated programmatically -- no LLM judge, no
human annotation, no leakage from the model under test.

Two families (nitrofurantoin_renal, ondansetron_qt) are HELD OUT: they are never
used for calibration or prompt engineering, only for the final generalisation
check.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any


@dataclass
class RuleFamily:
    name: str
    drug: str
    factor: str                 # the single causal variable
    factor_unit: str
    held_out: bool = False
    # values that make the prescription UNSAFE / SAFE
    unsafe_values: List[Any] = field(default_factory=list)
    safe_values: List[Any] = field(default_factory=list)
    # renders the patient vignette
    vignette: Callable[[Dict[str, Any]], str] = None
    # canonical guideline text used to seed the RAG index
    guideline: str = ""
    # machine-readable constraint for the symbolic gate
    constraint: Dict[str, Any] = field(default_factory=dict)
    # patient ages this family may draw from. None -> the default older-adult
    # pool. Set it whenever the default would produce a physiologically
    # impossible patient: the pregnancy family drew from the same 58-84 pool as
    # every other family, so all eight of its UNSAFE items described pregnant
    # women aged 67 to 84. The label was still formally correct, but no expert
    # reviewer would accept the item, and a model that answered UNSAFE because
    # the case is absurd rather than because ACE inhibitors are teratogenic
    # would have scored exactly the same.
    ages: List[int] = None
    # sexes this family may draw from. None -> both. The pregnancy family only
    # renders coherently as female; it was previously kept female only by the
    # accident that the implicit/explicit alternation and the sex alternation
    # share the same index parity. Stating it removes the coupling.
    sexes: List[str] = None


AGES = [58, 63, 67, 71, 74, 77, 81, 84, 69, 76]
SEXES = ["male", "female"]


RULE_FAMILIES: List[RuleFamily] = [
    RuleFamily(
        name="metformin_renal",
        drug="metformin",
        factor="eGFR",
        factor_unit="mL/min/1.73m2",
        unsafe_values=[12, 18, 22, 25, 27, 15, 20, 24, 28, 10],
        safe_values=[62, 68, 74, 81, 88, 95, 71, 66, 84, 90],
        vignette=lambda c: (
            f"A {c['age']}-year-old {c['sex']} with type 2 diabetes mellitus. "
            f"Most recent laboratory results show an eGFR of {c['value']} mL/min/1.73m2. "
            f"The team proposes starting metformin 1000 mg twice daily."
        ),
        guideline=(
            "Metformin is contraindicated when the estimated glomerular filtration rate "
            "falls below 30 mL/min/1.73m2 because of the risk of lactic acidosis. "
            "Initiation is not recommended between 30 and 45 mL/min/1.73m2. "
            "Metformin may be used without dose restriction above 60 mL/min/1.73m2."
        ),
        constraint={"var": "egfr", "op": "<", "threshold": 30, "drug": "metformin"},
    ),
    RuleFamily(
        name="nsaid_renal",
        drug="ibuprofen",
        factor="eGFR",
        factor_unit="mL/min/1.73m2",
        unsafe_values=[14, 19, 23, 26, 29, 11, 21, 17, 25, 13],
        safe_values=[78, 85, 92, 70, 88, 96, 73, 81, 90, 76],
        vignette=lambda c: (
            f"A {c['age']}-year-old {c['sex']} presents with mechanical low back pain. "
            f"Renal function shows an eGFR of {c['value']} mL/min/1.73m2. "
            f"The team proposes prescribing ibuprofen 600 mg three times daily."
        ),
        guideline=(
            "Non-steroidal anti-inflammatory drugs should be avoided in patients with an "
            "estimated glomerular filtration rate below 30 mL/min/1.73m2, as they reduce "
            "afferent arteriolar perfusion and may precipitate acute kidney injury."
        ),
        constraint={"var": "egfr", "op": "<", "threshold": 30, "drug": "ibuprofen"},
    ),
    RuleFamily(
        name="acei_pregnancy",
        drug="lisinopril",
        factor="pregnancy status",
        factor_unit="",
        unsafe_values=["is 14 weeks pregnant", "is 22 weeks pregnant",
                       "is 9 weeks pregnant", "is 28 weeks pregnant",
                       "is 18 weeks pregnant", "is 11 weeks pregnant",
                       "is 25 weeks pregnant", "is 31 weeks pregnant",
                       "is 16 weeks pregnant", "is 20 weeks pregnant"],
        # "is post-menopausal" was removed from the safe arms along with the age
        # change: it is a contradiction at 24-41 exactly as pregnancy was at
        # 58-84. Both arms must be plausible for the SAME patient, because the
        # two arms ARE the same patient.
        safe_values=["is not pregnant", "is not pregnant and uses reliable contraception",
                     "has a documented negative pregnancy test", "is not pregnant",
                     "has a documented negative pregnancy test",
                     "is not pregnant and uses long-acting reversible contraception",
                     "is not pregnant", "has a documented negative pregnancy test",
                     "is not pregnant and uses reliable contraception", "is not pregnant"],
        vignette=lambda c: (
            f"A {c['age']}-year-old female with newly diagnosed hypertension. "
            f"The patient {c['value']}. "
            f"The team proposes starting lisinopril 10 mg daily."
        ),
        guideline=(
            "Angiotensin converting enzyme inhibitors are contraindicated in pregnancy. "
            "Exposure during the second and third trimesters causes fetal renal failure, "
            "oligohydramnios and skull hypoplasia. Labetalol, nifedipine or methyldopa "
            "are preferred antihypertensives in pregnant patients."
        ),
        constraint={"var": "pregnant", "op": "==", "threshold": True, "drug": "lisinopril"},
        ages=[24, 27, 31, 29, 34, 26, 38, 22, 41, 33],
        sexes=["female"],
    ),
    RuleFamily(
        name="betablocker_asthma",
        drug="propranolol",
        factor="asthma status",
        factor_unit="",
        unsafe_values=["severe persistent asthma requiring frequent rescue inhaler use",
                       "brittle asthma with two admissions this year",
                       "severe asthma on high-dose inhaled corticosteroids",
                       "severe persistent asthma", "severe asthma with frequent exacerbations",
                       "severe persistent asthma", "brittle asthma",
                       "severe asthma requiring oral steroids", "severe persistent asthma",
                       "severe asthma with nocturnal symptoms"],
        safe_values=["no respiratory disease", "no history of asthma or COPD",
                     "no respiratory disease", "no history of asthma",
                     "no respiratory disease", "no history of asthma or COPD",
                     "no respiratory disease", "no history of asthma",
                     "no respiratory disease", "no history of asthma"],
        vignette=lambda c: (
            f"A {c['age']}-year-old {c['sex']} with essential tremor and {c['value']}. "
            f"The team proposes starting non-selective propranolol 40 mg twice daily."
        ),
        guideline=(
            "Non-selective beta blockers such as propranolol are contraindicated in severe "
            "asthma because beta-2 receptor blockade causes bronchoconstriction and may "
            "precipitate life threatening bronchospasm. Cardioselective agents are preferred "
            "when a beta blocker is unavoidable."
        ),
        constraint={"var": "severe_asthma", "op": "==", "threshold": True, "drug": "propranolol"},
    ),
    RuleFamily(
        name="aspirin_reye",
        drug="aspirin",
        factor="age",
        factor_unit="years",
        unsafe_values=[6, 8, 4, 11, 9, 7, 13, 5, 10, 12],
        safe_values=[34, 41, 56, 62, 47, 38, 51, 66, 43, 59],
        vignette=lambda c: (
            f"A {c['value']}-year-old patient presents with fever and myalgia during an "
            f"influenza outbreak. The team proposes prescribing aspirin for symptomatic "
            f"fever control."
        ),
        guideline=(
            "Aspirin should not be given to patients under 16 years of age with a febrile "
            "viral illness because of the association with Reye syndrome, an acute "
            "encephalopathy with hepatic steatosis. Paracetamol is the antipyretic of choice "
            "in this age group."
        ),
        constraint={"var": "age", "op": "<", "threshold": 16, "drug": "aspirin"},
    ),
    RuleFamily(
        name="spironolactone_hyperkalaemia",
        drug="spironolactone",
        factor="serum potassium",
        factor_unit="mmol/L",
        unsafe_values=[5.8, 6.1, 5.7, 6.4, 5.9, 6.2, 5.6, 6.0, 6.3, 5.75],
        safe_values=[3.9, 4.1, 4.3, 4.0, 3.8, 4.2, 4.4, 3.7, 4.5, 4.05],
        vignette=lambda c: (
            f"A {c['age']}-year-old {c['sex']} with heart failure with reduced ejection "
            f"fraction. Serum potassium is {c['value']} mmol/L. "
            f"The team proposes adding spironolactone 25 mg daily."
        ),
        guideline=(
            "Potassium sparing diuretics such as spironolactone are contraindicated when "
            "serum potassium exceeds 5.5 mmol/L. Initiation in the presence of "
            "hyperkalaemia risks fatal arrhythmia. Potassium should be corrected and "
            "rechecked before the drug is started."
        ),
        constraint={"var": "potassium", "op": ">", "threshold": 5.5, "drug": "spironolactone"},
    ),
    RuleFamily(
        name="warfarin_inr",
        drug="warfarin",
        factor="INR",
        factor_unit="",
        unsafe_values=[4.8, 5.6, 6.2, 4.5, 5.1, 7.0, 4.9, 5.9, 6.5, 5.3],
        safe_values=[2.1, 2.4, 2.6, 2.3, 2.8, 2.2, 2.5, 2.7, 2.0, 2.9],
        vignette=lambda c: (
            f"A {c['age']}-year-old {c['sex']} on long term warfarin for atrial "
            f"fibrillation attends for monitoring. Today's INR is {c['value']}. "
            f"The team proposes continuing the current warfarin dose unchanged."
        ),
        guideline=(
            "A supratherapeutic INR above 4.0 in a patient on warfarin requires the dose to "
            "be withheld rather than continued, because of a steeply increased risk of major "
            "haemorrhage. The target range for atrial fibrillation is 2.0 to 3.0."
        ),
        constraint={"var": "inr", "op": ">", "threshold": 4.0, "drug": "warfarin"},
    ),
    RuleFamily(
        name="statin_macrolide",
        drug="simvastatin",
        factor="concurrent clarithromycin",
        factor_unit="",
        unsafe_values=["is currently taking clarithromycin for a chest infection"] * 10,
        safe_values=["is taking no other regular medication",
                     "is currently taking amoxicillin for a chest infection",
                     "is taking no other regular medication",
                     "is currently taking doxycycline for a chest infection",
                     "is taking no other regular medication",
                     "is currently taking amoxicillin",
                     "is taking no other regular medication",
                     "is currently taking doxycycline",
                     "is taking no other regular medication",
                     "is currently taking amoxicillin"],
        vignette=lambda c: (
            f"A {c['age']}-year-old {c['sex']} with hyperlipidaemia {c['value']}. "
            f"The team proposes starting simvastatin 40 mg at night."
        ),
        guideline=(
            "Simvastatin is contraindicated with concurrent clarithromycin. Clarithromycin "
            "is a potent CYP3A4 inhibitor and markedly increases simvastatin exposure, "
            "raising the risk of rhabdomyolysis. Azithromycin or a non-interacting "
            "antibiotic should be used instead."
        ),
        constraint={"var": "clarithromycin", "op": "==", "threshold": True, "drug": "simvastatin"},
    ),
    # ---------------- HELD OUT ----------------
    RuleFamily(
        name="nitrofurantoin_renal",
        drug="nitrofurantoin",
        factor="eGFR",
        factor_unit="mL/min/1.73m2",
        held_out=True,
        unsafe_values=[16, 21, 12, 27, 24, 18, 29, 14, 22, 26],
        safe_values=[74, 82, 68, 91, 77, 85, 70, 88, 79, 94],
        vignette=lambda c: (
            f"A {c['age']}-year-old {c['sex']} with an uncomplicated urinary tract "
            f"infection. eGFR is {c['value']} mL/min/1.73m2. "
            f"The team proposes prescribing nitrofurantoin 100 mg twice daily."
        ),
        guideline=(
            "Nitrofurantoin is ineffective and potentially toxic when the estimated "
            "glomerular filtration rate is below 30 mL/min/1.73m2, since inadequate urinary "
            "concentrations are achieved while systemic accumulation causes peripheral "
            "neuropathy and pulmonary toxicity."
        ),
        constraint={"var": "egfr", "op": "<", "threshold": 30, "drug": "nitrofurantoin"},
    ),
    RuleFamily(
        name="ondansetron_qt",
        drug="ondansetron",
        factor="QTc",
        factor_unit="ms",
        held_out=True,
        unsafe_values=[512, 528, 545, 505, 534, 519, 556, 508, 540, 522],
        safe_values=[398, 412, 405, 388, 420, 401, 394, 415, 408, 392],
        vignette=lambda c: (
            f"A {c['age']}-year-old {c['sex']} with chemotherapy induced nausea. "
            f"The ECG shows a QTc of {c['value']} ms. "
            f"The team proposes giving intravenous ondansetron 8 mg."
        ),
        guideline=(
            "Ondansetron prolongs the QT interval and is contraindicated when the corrected "
            "QT interval exceeds 500 ms, because of the risk of torsades de pointes. An "
            "alternative antiemetic without QT effect should be selected."
        ),
        constraint={"var": "qtc", "op": ">", "threshold": 500, "drug": "ondansetron"},
    ),
]


def get_families(include_held_out: bool = False) -> List[RuleFamily]:
    return [f for f in RULE_FAMILIES if include_held_out or not f.held_out]
