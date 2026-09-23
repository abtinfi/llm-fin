"""
Build a counterfactual drug-safety benchmark from MIMIC-IV Clinical Database
Demo v2.2 -- the "MIMIC-IV Real-Value Cohort".

WHY THIS ARM EXISTS
--------------------
The repository already has two benchmarks and both fabricate numbers:

  data/synthetic_control  every number invented, vignette templated.
  data/medcalc            note is REAL published prose, but one arm of each
                          pair has its driving number EDITED to cross the
                          threshold. Half of every pair is synthetic.

This arm removes the fabricated number entirely. A counterfactual pair is
built from TWO REAL MEASUREMENTS of the SAME patient that happen to fall on
opposite sides of a threshold. Nothing is invented: not the lab value, not the
age, not the sex.

FOUR RULE FAMILIES (expanded 2026-09-02, was one)
---------------------------------------------------
Each maps to a family in `src/rules.py` so its threshold provenance is the
SAME audited number, not a second, uncoordinated one
(`results/threshold_provenance.md`):

  metformin_egfr30    eGFR < 30    contraindicated        attested_exact
  metformin_egfr45    eGFR < 45    not-recommended-to-initiate (weaker claim,
                                    same FDA label, dosage_and_administration
                                    section: "not recommended in patients with
                                    an eGFR between 30 and less than 45")
  spironolactone_k5_5 K > 5.5      hyperkalaemia           construct_mismatch
                                    (rules.py's own status: the label's 5.0
                                    mEq/L is an initiation criterion, not this
                                    ceiling -- built anyway, flagged, not hidden)
  warfarin_inr4       INR > 4.0    bleeding risk           attested_exact
                                    HELD OUT -- mechanistically distinct
                                    (anticoagulation, not renal clearance),
                                    same role nitrofurantoin_renal /
                                    ondansetron_qt play in the other arms

GLOBAL PATIENT DISJOINTNESS
----------------------------
eGFR30 and eGFR45 are the SAME lab (creatinine) at two thresholds: 20 of 23
eGFR30 patients also qualify for eGFR45, and eGFR45 overlaps potassium in 18
patients. A patient is therefore assigned to exactly ONE split -- never one
family per split -- and contributes whatever pairs they qualify for to that
split. The 8 patients whose INR straddles 4.0 are removed from every other
family's pool entirely before the train/test/calib partition, so no patient
who appears in `heldout` (via warfarin) can also appear in `train` (via
metformin or spironolactone) even though their creatinine or potassium might
independently qualify.

WHAT THIS BUYS, AND WHAT IT DOES NOT
-------------------------------------
It buys three of the FOUR attested-or-flagged thresholds in this project
built on real patient measurements. It does NOT buy a strict minimal pair --
the two arms are different TIMEPOINTS in the same patient, so the underlying
clinical state genuinely differed. What is controlled is the rendered note:
minimal by construction (age, sex, one lab value, the drug), so the two
prompts differ in exactly one number and both numbers are real. medcalc makes
the opposite trade -- timepoint fixed, number faked. Report both.

INTEGRITY GUARDS (each aborts or drops rather than guesses)
-------------------------------------------------------------
1. eGFR is recomputed from the stored creatinine by src/renal.py; potassium
   and INR are used as stored (no derived formula to re-check).
2. Both arms of a pair share the same age and sex.
3. The two arms must land on opposite sides of the threshold AFTER rounding to
   printed precision, not before.
4. Splits are GLOBALLY patient-disjoint -- one patient, one split, across
   every family.
5. Physiologically implausible values are rejected via renal.implausible
   (potassium_mEqL, inr ranges added 2026-09-02 alongside the existing ones).

USAGE
  python src/build_mimic.py --src data/mimic_demo --out data/mimic
"""

import argparse
import csv
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from renal import ckd_epi, implausible          # noqa: E402

# Provenance labels, keyed by --source. Stamped into every item and into
# build_meta.json. These used to be hard-coded to the Demo, so the first build
# from full MIMIC-IV v3.1 labelled all 369,354 of its items "Demo v2.2 (ODbL)"
# and its build_meta.json "credentialed: false" -- i.e. credentialed data
# carrying a label that says it may be redistributed. main() now refuses a
# build whose patient count contradicts the label (see DEMO_PATIENTS).
DATA_SOURCES = {
    "demo": dict(item="MIMIC-IV Clinical Database Demo v2.2 (ODbL)",
                 source="MIMIC-IV Clinical Database Demo v2.2",
                 licence="ODbL", credentialed=False,
                 url="https://physionet.org/content/mimic-iv-demo/2.2/"),
    "v3.1": dict(item="MIMIC-IV v3.1 (PhysioNet credentialed DUA)",
                 source="MIMIC-IV v3.1",
                 licence="PhysioNet Credentialed Health Data License 1.5.0",
                 credentialed=True,
                 url="https://physionet.org/content/mimiciv/3.1/"),
}
DEMO_PATIENTS = 100      # the Demo is a fixed 100-patient subset

AGE_RANGE = (18, 91)   # MIMIC caps de-identified ages at 91; CKD-EPI needs 18+


def egfr_from_scr(scr, age, sex):
    return ckd_epi(scr, age, sex)


def identity(v, age, sex):
    return v


# One entry per rule family. `convert` turns the stored lab value into the
# quantity the threshold is stated in (identity for potassium/INR, CKD-EPI for
# creatinine). `render_unit`/`render_label` drive the rendered sentence.
FAMILIES = {
    "metformin_egfr30": dict(
        itemid="50912", lo=0.1, hi=25.0, plausible_key="creatinine_mgdl",
        convert=egfr_from_scr, threshold=30.0, op="<", drug="metformin",
        lab_name="creatinine", render_unit="mg/dL", render_precision=2,
        threshold_source="openfda", threshold_status="attested_exact",
        threshold_evidence="FDA label, CONTRAINDICATIONS: 'Severe renal "
                           "impairment: (eGFR below 30 mL/min/1.73 m2)'",
        held_out=False, calculator="ckd_epi"),
    "metformin_egfr45": dict(
        itemid="50912", lo=0.1, hi=25.0, plausible_key="creatinine_mgdl",
        convert=egfr_from_scr, threshold=45.0, op="<", drug="metformin",
        lab_name="creatinine", render_unit="mg/dL", render_precision=2,
        threshold_source="openfda", threshold_status="attested_exact",
        threshold_evidence="FDA label, DOSAGE AND ADMINISTRATION: "
                           "'not recommended in patients with an eGFR "
                           "between 30 and less than 45 mL/min/1.73 m2' -- "
                           "a NOT-RECOMMENDED-TO-INITIATE zone, weaker than "
                           "the eGFR<30 contraindication; do not conflate "
                           "the two families in prose",
        held_out=False, calculator="ckd_epi"),
    "spironolactone_k5_5": dict(
        itemid="50971", lo=0.5, hi=12.0, plausible_key="potassium_mEqL",
        convert=identity, threshold=5.5, op=">", drug="spironolactone",
        lab_name="potassium", render_unit="mEq/L", render_precision=1,
        threshold_source="UNSOURCED", threshold_status="construct_mismatch",
        threshold_evidence="rules.py's own audited status "
                           "(results/threshold_provenance.md): the FDA "
                           "label's potassium <=5.0 mEq/L is a heart-failure "
                           "INITIATION criterion, not this contraindication "
                           "ceiling. Built anyway and flagged, not hidden.",
        held_out=False, calculator="direct_lab_value"),
    "warfarin_inr4": dict(
        itemid="51237", lo=0.1, hi=15.0, plausible_key="inr",
        convert=identity, threshold=4.0, op=">", drug="warfarin",
        lab_name="INR", render_unit="", render_precision=1,
        threshold_source="openfda", threshold_status="attested_exact",
        threshold_evidence="FDA label, WARNINGS AND CAUTIONS: 'high "
                           "intensity of anticoagulation (INR > 4)'",
        held_out=True, calculator="direct_lab_value"),
}


def read_gz(path):
    with gzip.open(path, "rt") as fh:
        yield from csv.DictReader(fh)


def load_patients(src):
    pats = {}
    for r in read_gz(Path(src) / "patients.csv.gz"):
        pats[r["subject_id"]] = (int(r["anchor_age"]),
                                 "female" if r["gender"] == "F" else "male")
    return pats


# One parsed copy of labevents per (src, itemid), shared by every caller.
#
# WHY THIS EXISTS. build_family_items() is called seven times per run -- once
# for the held-out family, then TWICE for each of the three trainable families
# (once unrestricted to discover who straddles the threshold, once restricted
# to the post-exclusion pool). Each call used to re-read labevents.csv.gz end
# to end. On the Demo's 107k rows that was invisible; on MIMIC-IV v3.1's ~158M
# rows it is seven full gzip+CSV passes for one dataset. prime_lab_cache()
# makes a SINGLE pass that collects every itemid the families need, and
# load_lab() serves the rest from memory.
_LAB_CACHE = {}


def prime_lab_cache(src, specs):
    """One pass over labevents.csv.gz for every itemid in `specs`."""
    wanted = {}                       # itemid -> (lo, hi)
    for spec in specs.values():
        lo, hi = wanted.get(spec["itemid"], (spec["lo"], spec["hi"]))
        # Two families can share an itemid (metformin_egfr30 and _egfr45 are
        # both creatinine). Keep the UNION of their plausible ranges so one
        # family's narrower bound cannot silently drop the other's rows.
        wanted[spec["itemid"]] = (min(lo, spec["lo"]), max(hi, spec["hi"]))
    for itemid in wanted:
        _LAB_CACHE[(str(src), itemid)] = defaultdict(list)

    kept = scanned = 0
    for r in read_gz(Path(src) / "labevents.csv.gz"):
        scanned += 1
        bounds = wanted.get(r["itemid"])
        if bounds is None or not r["valuenum"]:
            continue
        try:
            v = float(r["valuenum"])
        except ValueError:
            continue
        if bounds[0] <= v <= bounds[1]:
            _LAB_CACHE[(str(src), r["itemid"])][r["subject_id"]].append(
                (r["charttime"], round(v, 2)))
            kept += 1
    print(f"labevents: scanned={scanned} kept={kept} "
          f"itemids={sorted(wanted)}", flush=True)


def load_lab(src, itemid, lo, hi):
    cached = _LAB_CACHE.get((str(src), itemid))
    if cached is not None:
        # The cache holds the union range; re-apply this family's own bounds.
        return {sid: [(t, v) for t, v in ev if lo <= v <= hi]
                for sid, ev in cached.items()
                if any(lo <= v <= hi for _, v in ev)}
    ev = defaultdict(list)
    for r in read_gz(Path(src) / "labevents.csv.gz"):
        if r["itemid"] != itemid or not r["valuenum"]:
            continue
        try:
            v = float(r["valuenum"])
        except ValueError:
            continue
        if lo <= v <= hi:
            ev[r["subject_id"]].append((r["charttime"], round(v, 2)))
    return ev


def pick_pair(events, age, sex, spec, mode="nearest"):
    """Choose one UNSAFE and one SAFE real measurement for this patient."""
    thr = spec["threshold"]
    scored = [(t, v, spec["convert"](v, age, sex)) for t, v in events]
    if spec["op"] == "<":
        lo = [e for e in scored if e[2] < thr]         # UNSAFE
        hi = [e for e in scored if e[2] >= thr]         # SAFE
        pick_extreme_lo = min; pick_extreme_hi = max
        pick_near_lo = max; pick_near_hi = min
    else:
        lo = [e for e in scored if e[2] > thr]          # UNSAFE
        hi = [e for e in scored if e[2] <= thr]          # SAFE
        pick_extreme_lo = max; pick_extreme_hi = min
        pick_near_lo = min; pick_near_hi = max
    if not lo or not hi:
        return None
    if mode == "nearest":
        unsafe = pick_near_lo(lo, key=lambda e: e[2])
        safe = pick_near_hi(hi, key=lambda e: e[2])
    else:
        unsafe = pick_extreme_lo(lo, key=lambda e: e[2])
        safe = pick_extreme_hi(hi, key=lambda e: e[2])
    return unsafe, safe


def pick_control_pair(events, age, sex, spec, exclude=()):
    """
    Two REAL measurements on the SAME side of the threshold, as far apart as
    the patient's own record allows.

    This is the control arm, and MIMIC is the only place in the project where
    it can be built out of nothing but observed data. Every other arm has to
    edit a number to make a control pair; here the patient genuinely had two
    different values and neither crossed the threshold, so a model that flips
    between them is reacting to the prose, not to the clinical rule.

    Returns the widest same-side spread available, because a control pair whose
    two values are nearly identical tests almost nothing -- the prompt barely
    changes. `exclude` keeps the timepoints already used by the causal pair out
    of it, so the two pairs are not the same measurements re-labelled.
    """
    thr = spec["threshold"]
    scored = [(t, v, spec["convert"](v, age, sex)) for t, v in events
              if t not in exclude]
    if spec["op"] == "<":
        safe_side = [e for e in scored if e[2] >= thr]
        unsafe_side = [e for e in scored if e[2] < thr]
    else:
        safe_side = [e for e in scored if e[2] <= thr]
        unsafe_side = [e for e in scored if e[2] > thr]

    best = None
    for side, label in ((safe_side, "SAFE"), (unsafe_side, "UNSAFE")):
        if len(side) < 2:
            continue
        lo = min(side, key=lambda e: e[2])
        hi = max(side, key=lambda e: e[2])
        spread = abs(hi[2] - lo[2])
        if spread <= 0:
            continue
        if best is None or spread > best[0]:
            best = (spread, lo, hi, label)
    if best is None:
        return None
    _, a, b, label = best
    return a, b, label


def crosses(op, thr, u, s):
    return (u < thr <= s) if op == "<" else (u > thr >= s)


def render(age, sex, value, spec):
    """
    Phrasing matters, not just content: src/components.py's SymbolicGate
    extracts facts with LITERAL regexes -- "potassium is X", "INR is X" (see
    src/rules.py's own explicit vignettes) -- not "potassium/INR of X". Renal
    families stay implicit BY DESIGN (creatinine is stated, "eGFR" never is,
    matching the implicit renal families elsewhere in this project), so the
    gate correctly declines on them. Potassium and INR are rendered to MATCH
    the extractor so the gate gets a genuine chance to fire on this arm too,
    rather than silently reading 0% coverage because of a wording mismatch.
    """
    who = "woman" if sex == "female" else "man"
    if spec["lab_name"] == "INR":
        lab_txt = f"Today's INR is {value:.{spec['render_precision']}f}"
        return (f"A {age}-year-old {who} is reviewed. {lab_txt}. "
                f"{spec['drug'].capitalize()} is being considered.")
    if spec["lab_name"] == "potassium":
        lab_txt = f"Serum potassium is {value:.{spec['render_precision']}f} mmol/L"
        return (f"A {age}-year-old {who} is reviewed. {lab_txt}. "
                f"{spec['drug'].capitalize()} is being considered.")
    lab_txt = (f"a serum {spec['lab_name']} of "
              f"{value:.{spec['render_precision']}f} {spec['render_unit']}")
    return (f"A {age}-year-old {who} has {lab_txt}. "
            f"{spec['drug'].capitalize()} is being considered.")


PROMPT = ("You are reviewing a proposed prescription for safety.\n"
          "Answer with exactly one word on the first line: SAFE or UNSAFE.\n\n"
          "{vignette}\n\nIs it safe to prescribe {drug}?")


def build_family_items(fam, spec, pats, mode, eligible_patients=None,
                       src="data/mimic_demo", source="demo"):
    """
    All pairs for one family, restricted to `eligible_patients` if given.

    Returns (items, straddling_patient_ids, skip_counts).
    """
    labs = load_lab(src, spec["itemid"], spec["lo"], spec["hi"])
    items, straddling, skipped = [], set(), defaultdict(int)
    for sid, events in sorted(labs.items()):
        if eligible_patients is not None and sid not in eligible_patients:
            continue
        if sid not in pats:
            skipped["no_demographics"] += 1
            continue
        age, sex = pats[sid]
        if not AGE_RANGE[0] <= age <= AGE_RANGE[1]:
            skipped["age_outside_range"] += 1
            continue
        events = [(t, v) for t, v in events
                  if implausible(**{spec["plausible_key"]: v}) is None]
        if not events:
            skipped["all_values_implausible"] += 1
            continue
        chosen = pick_pair(events, age, sex, spec, mode)
        if chosen is None:
            skipped["no_straddling_pair"] += 1
            continue
        (t_u, v_u, q_u), (t_s, v_s, q_s) = chosen
        q_u_r = spec["convert"](round(v_u, 2), age, sex)
        q_s_r = spec["convert"](round(v_s, 2), age, sex)
        q_u_r, q_s_r = round(q_u_r, 2), round(q_s_r, 2)
        if not crosses(spec["op"], spec["threshold"], q_u_r, q_s_r):
            skipped["rounding_crossed_threshold"] += 1
            continue
        if any(implausible(eGFR=q) is not None for q in (q_u_r, q_s_r)
               if spec["calculator"] == "ckd_epi"):
            skipped["implausible_derived_value"] += 1
            continue

        pair_id = f"{fam}__mimic-{sid}"
        for arm, v, q, t, label in (
                ("unsafe", v_u, q_u_r, t_u, "UNSAFE"),
                ("safe",   v_s, q_s_r, t_s, "SAFE")):
            vig = render(age, sex, v, spec)
            items.append({
                "id": f"{pair_id}__{arm}",
                "pair_id": pair_id,
                "family": fam,
                "held_out": spec["held_out"],
                "arm": arm,
                "drug": spec["drug"],
                "factor": spec["lab_name"] if spec["calculator"] != "ckd_epi"
                          else "eGFR",
                "factor_value": q,
                "label": label,
                "vignette": vig,
                "presentation": "implicit" if spec["calculator"] == "ckd_epi"
                                else "explicit",
                "edited_arm": False,          # THE POINT: no number invented
                "is_control": False,
                "calculator": spec["calculator"],
                "source_calculator": spec["calculator"],
                "note_id": f"mimic-{sid}",
                "subject_id": sid,
                "charttime": t,
                "facts": {"lab_raw": v, "lab_name": spec["lab_name"],
                          "lab_unit": spec["render_unit"] or "ratio",
                          "age": age, "sex": sex.capitalize(), "race": None},
                "prompt": PROMPT.format(vignette=vig, drug=spec["drug"]),
                "provenance": {
                    "source": DATA_SOURCES[source]["item"],
                    "table": "hosp/labevents", "itemid": spec["itemid"],
                    "threshold_source": spec["threshold_source"],
                    "threshold_status": spec["threshold_status"],
                    "threshold_evidence": spec["threshold_evidence"],
                    "both_arms_real": True},
            })
        straddling.add(sid)
        skipped["kept_pairs"] += 1

        # CONTROL PAIR, from the same patient's other real measurements.
        ctrl = pick_control_pair(events, age, sex, spec,
                                 exclude={t_u, t_s})
        if ctrl is None:
            skipped["no_same_side_control_pair"] += 1
            continue
        (t_a, v_a, _), (t_b, v_b, _), c_label = ctrl
        q_a = round(spec["convert"](round(v_a, 2), age, sex), 2)
        q_b = round(spec["convert"](round(v_b, 2), age, sex), 2)
        # Rounding must not push either arm across; if it does this is not a
        # control pair any more and is dropped rather than silently relabelled.
        if crosses(spec["op"], spec["threshold"], q_a, q_b) or \
                crosses(spec["op"], spec["threshold"], q_b, q_a):
            skipped["control_rounding_crossed_threshold"] += 1
            continue
        c_pair = f"{fam}__mimic-{sid}__ctrl"
        for arm, v, q, t in (("ctrl_a", v_a, q_a, t_a),
                             ("ctrl_b", v_b, q_b, t_b)):
            vig = render(age, sex, v, spec)
            items.append({
                "id": f"{c_pair}__{arm}",
                "pair_id": c_pair,
                "family": fam,
                "held_out": spec["held_out"],
                "arm": arm,
                "drug": spec["drug"],
                "factor": spec["lab_name"] if spec["calculator"] != "ckd_epi"
                          else "eGFR",
                "factor_value": q,
                "label": c_label,          # SAME on both arms, by definition
                "vignette": vig,
                "presentation": "implicit" if spec["calculator"] == "ckd_epi"
                                else "explicit",
                "edited_arm": False,
                "is_control": True,
                "calculator": spec["calculator"],
                "source_calculator": spec["calculator"],
                "note_id": f"mimic-{sid}",
                "subject_id": sid,
                "charttime": t,
                "facts": {"lab_raw": v, "lab_name": spec["lab_name"],
                          "lab_unit": spec["render_unit"] or "ratio",
                          "age": age, "sex": sex.capitalize(), "race": None},
                "prompt": PROMPT.format(vignette=vig, drug=spec["drug"]),
                "provenance": {
                    "source": DATA_SOURCES[source]["item"],
                    "table": "hosp/labevents", "itemid": spec["itemid"],
                    "threshold_source": spec["threshold_source"],
                    "threshold_status": spec["threshold_status"],
                    "threshold_evidence": spec["threshold_evidence"],
                    "both_arms_real": True,
                    "control_pair": ("two real measurements on the same side "
                                     "of the threshold; the label does not "
                                     "change, so a flip here is spurious")},
            })
        skipped["kept_control_pairs"] += 1
    return items, straddling, skipped


def split_by_patient(patient_items, fracs, seed):
    sids = sorted(patient_items)
    random.Random(seed).shuffle(sids)
    n = len(sids)
    n_test = max(1, int(round(fracs[0] * n))) if n else 0
    n_calib = max(1, int(round(fracs[1] * n))) if n else 0
    groups = {"test": sids[:n_test],
              "calib": sids[n_test:n_test + n_calib],
              "train": sids[n_test + n_calib:]}
    return {k: [it for s in v for it in patient_items[s]] for k, v in groups.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/mimic_demo")
    ap.add_argument("--out", default="data/mimic")
    ap.add_argument("--source", choices=sorted(DATA_SOURCES), default="demo",
                    help="which MIMIC release --src holds; sets the licence "
                         "and credentialed labels on every emitted item")
    ap.add_argument("--mode", choices=["nearest", "extreme"], default="nearest")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.5)
    ap.add_argument("--calib_frac", type=float, default=0.2)
    args = ap.parse_args()

    pats = load_patients(args.src)
    print(f"patients={len(pats)}  source={args.source}")
    # Refuse a label the data contradicts. A mislabel here is not cosmetic: it
    # is what tells a reader whether the emitted files may be redistributed.
    if args.source == "demo" and len(pats) != DEMO_PATIENTS:
        sys.exit(f"--source demo but {args.src} holds {len(pats)} patients; "
                 f"the Demo has exactly {DEMO_PATIENTS}. Pass --source v3.1.")
    if args.source != "demo" and len(pats) == DEMO_PATIENTS:
        sys.exit(f"--source {args.source} but {args.src} holds the Demo's "
                 f"{DEMO_PATIENTS} patients. Pass --source demo.")

    # One pass over labevents for every itemid the families need, before any
    # family is built. See prime_lab_cache().
    prime_lab_cache(args.src, FAMILIES)

    # Pass 1: the held-out family, unrestricted -- this decides which
    # patients are EXCLUDED from every other family's pool.
    held_specs = {k: v for k, v in FAMILIES.items() if v["held_out"]}
    trainable_specs = {k: v for k, v in FAMILIES.items() if not v["held_out"]}

    heldout_items, heldout_patients = [], set()
    for fam, spec in held_specs.items():
        items, straddling, skipped = build_family_items(fam, spec, pats,
                                                         args.mode,
                                                         src=args.src,
                                                         source=args.source)
        print(f"[heldout] {fam}: {dict(skipped)}")
        heldout_items += items
        heldout_patients |= straddling

    # Pass 2: every other family, EXCLUDING heldout patients globally, so a
    # patient can never appear in both `heldout` and any of train/test/calib.
    pool_items = defaultdict(list)     # subject_id -> items across families
    all_straddling = set()
    for fam, spec in trainable_specs.items():
        # eligible_patients=None on the first call to find who straddles;
        # then explicitly re-run EXCLUDING heldout patients.
        _, straddling_all, _ = build_family_items(fam, spec, pats, args.mode,
                                                  src=args.src,
                                                  source=args.source)
        eligible = straddling_all - heldout_patients
        items, straddling, skipped = build_family_items(
            fam, spec, pats, args.mode, eligible_patients=eligible,
            src=args.src, source=args.source)
        print(f"[trainable] {fam}: eligible_after_excluding_heldout="
              f"{len(eligible)}  {dict(skipped)}")
        for it in items:
            pool_items[it["subject_id"]].append(it)
        all_straddling |= straddling

    print(f"\npatients contributing to train/test/calib: {len(pool_items)}")
    print(f"patients held out entirely (warfarin_inr4): {len(heldout_patients)}")
    overlap = set(pool_items) & heldout_patients
    assert not overlap, f"leakage: {overlap} in both pools"

    splits = split_by_patient(pool_items, (args.test_frac, args.calib_frac),
                              args.seed)
    splits["heldout"] = heldout_items

    # Guard 2 + integrity: both arms share age/sex, opposite labels, patient-
    # disjoint splits.
    seen_split = {}
    for name, rows in splits.items():
        byp = defaultdict(list)
        for it in rows:
            byp[it["pair_id"]].append(it)
            prev = seen_split.setdefault(it["subject_id"], name)
            assert prev == name, (f"subject {it['subject_id']} in both "
                                  f"{prev} and {name}")
        for pid, arms in byp.items():
            assert len(arms) == 2, f"{pid}: {len(arms)} arms"
            a, b = arms
            assert a["facts"]["age"] == b["facts"]["age"], pid
            assert a["facts"]["sex"] == b["facts"]["sex"], pid
            # A causal pair must FLIP; a control pair must NOT. Asserting both
            # separately is what makes a mislabelled control pair a build
            # failure rather than a silently weakened benchmark -- a control
            # arm that crosses the threshold is a causal arm wearing the wrong
            # label, and would drag the spurious-flip rate toward the causal
            # one and make discrimination look better than it is.
            if a.get("is_control") or b.get("is_control"):
                assert a.get("is_control") and b.get("is_control"), (
                    f"{pid}: one arm is a control and the other is not")
                assert a["label"] == b["label"], (
                    f"{pid}: control arms carry different labels, so the "
                    f"value crossed the threshold")
                assert a["facts"]["lab_raw"] != b["facts"]["lab_raw"], (
                    f"{pid}: control arms are the same measurement, so the "
                    f"prompt does not actually differ")
            else:
                assert a["label"] != b["label"], pid

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, rows in splits.items():
        p = out / f"counterfactual_{name}.jsonl"
        with p.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        counts[name] = len(rows)
        by_fam = defaultdict(int)
        for r in rows:
            by_fam[r["family"]] += 1
        print(f"  {p}: {len(rows)} items / {len(rows)//2} pairs  "
              f"{dict(by_fam)}")

    corpus = [
        {"id": "guideline::metformin_egfr30",
         "text": "Metformin is contraindicated in severe renal impairment "
                 "(eGFR below 30 mL/min/1.73 m2). [FDA label, "
                 "CONTRAINDICATIONS -- attested_exact]"},
        {"id": "guideline::metformin_egfr45",
         "text": "Metformin initiation is not recommended in patients with "
                 "an eGFR between 30 and less than 45 mL/min/1.73 m2. "
                 "[FDA label, DOSAGE AND ADMINISTRATION -- attested_exact]"},
        {"id": "guideline::spironolactone_k5_5",
         "text": "Spironolactone should be used with caution in patients "
                 "with elevated serum potassium; a level above 5.5 mEq/L is "
                 "treated here as a contraindication threshold. "
                 "[threshold_status: construct_mismatch -- the FDA label's "
                 "5.0 mEq/L figure is an initiation criterion, not this "
                 "ceiling]"},
        {"id": "guideline::warfarin_inr4",
         "text": "An INR greater than 4 provides no additional therapeutic "
                 "benefit and is associated with a higher risk of bleeding. "
                 "[FDA label, WARNINGS AND CAUTIONS -- attested_exact]"},
    ]
    with (out / "rag_corpus.jsonl").open("w") as fh:
        for d in corpus:
            fh.write(json.dumps(d) + "\n")
    print(f"  {out}/rag_corpus.jsonl: {len(corpus)} passages")

    ds = DATA_SOURCES[args.source]
    meta = {"source": ds["source"],
            "licence": ds["licence"], "credentialed": ds["credentialed"],
            "url": ds["url"],
            "families": {k: {kk: vv for kk, vv in v.items()
                             if kk != "convert"}
                        for k, v in FAMILIES.items()},
            "pair_selection": args.mode, "seed": args.seed,
            "both_arms_real": True,
            "caveat": "The two arms are different TIMEPOINTS in the same "
                      "real patient, so the clinical state genuinely "
                      "differed. The rendered note is minimal (age, sex, "
                      "one lab value, drug) so the prompts differ in "
                      "exactly one number, and both numbers are real "
                      "measurements. metformin_egfr30 and metformin_egfr45 "
                      "share the same underlying creatinine measurements at "
                      "two thresholds of differing clinical severity -- do "
                      "not report them as independent evidence.",
            "counts": counts}
    (out / "build_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out}/build_meta.json")


if __name__ == "__main__":
    main()
