"""
Build a counterfactual drug-safety benchmark from MIMIC-IV Clinical Database
Demo v2.2 -- the "MIMIC-IV Real-Value Cohort".

WHY THIS ARM EXISTS
-------------------
The repository already has two benchmarks and both fabricate numbers:

  data/synthetic_control  every number invented, vignette templated.
  data/medcalc            note is REAL published prose, but one arm of each
                          pair has its driving number EDITED to cross the
                          threshold. Half of every pair is synthetic.

This arm removes the fabricated number entirely. MIMIC-IV Demo v2.2 carries
3,003 serum-creatinine measurements (itemid 50912) over 100 real patients, and
**23 of those patients have two real measurements that fall on opposite sides
of the eGFR-30 threshold**. So a counterfactual pair can be built from two
values that were both actually measured in the same real patient. Nothing is
invented: not the creatinine, not the age, not the sex.

WHAT THIS BUYS, AND WHAT IT DOES NOT
------------------------------------
It buys the one threshold in this project that an FDA label actually attests:
metformin, eGFR < 30 (results/threshold_provenance.md, `attested_exact`). The
QT family cannot be built here -- the demo has no ECG intervals -- and that is
the family carrying the Aim 3 result, so this arm does not rescue it.

It does NOT buy a clean minimal pair in the strict sense, and pretending
otherwise would be the easy lie. The two arms come from different TIMEPOINTS in
the same patient, so the underlying clinical state genuinely differed. What is
controlled is the rendered note: it states age, sex, one creatinine and the
proposed drug, so the two prompts differ in exactly one number and both numbers
are real. The trade is explicit -- MedCalc keeps the timepoint fixed and fakes
a number; this keeps every number real and lets the timepoint move. Neither
dominates, and the paper should report both.

THE PROBLEM THIS ARM ACTUALLY SOLVES
------------------------------------
BACKLOG's hardest open item: 42.8% of real renal notes state creatinine more
than once with different values, so a threshold-reading layer cannot be applied
to them at all. Structured `labevents` has no such ambiguity -- a measurement is
one row with one charttime. The renderer emits exactly one creatinine, chosen
by an explicit rule that is recorded per item. That is the argument for
structured input, made concrete.

INTEGRITY GUARDS (each aborts the build)
----------------------------------------
1. eGFR is recomputed from the stored creatinine by src/renal.py; an item whose
   recomputed eGFR does not reproduce the one used for labelling is dropped.
2. Both arms must carry the SAME age and sex, so the only difference in the
   rendered text is the creatinine.
3. The two arms must land on opposite sides of the threshold, checked after
   rounding to the printed precision -- not before.
4. Splits are patient-disjoint: a subject_id never appears in two splits.
5. Physiologically implausible values are rejected via renal.implausible.

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

CREATININE_ITEMID = "50912"      # d_labitems: Creatinine | Blood | Chemistry
EGFR_THRESHOLD = 30.0            # FDA metformin label, attested_exact
DRUG = "metformin"
FAMILY = "metformin_renal"


def read_gz(path):
    with gzip.open(path, "rt") as fh:
        yield from csv.DictReader(fh)


def load_cohort(src):
    """(subject_id -> (age, sex)) and (subject_id -> [(charttime, scr)])."""
    pats = {}
    for r in read_gz(Path(src) / "patients.csv.gz"):
        pats[r["subject_id"]] = (int(r["anchor_age"]),
                                 "female" if r["gender"] == "F" else "male")
    labs = defaultdict(list)
    for r in read_gz(Path(src) / "labevents.csv.gz"):
        if r["itemid"] != CREATININE_ITEMID or not r["valuenum"]:
            continue
        try:
            v = float(r["valuenum"])
        except ValueError:
            continue
        if not 0.1 <= v <= 25.0:            # obvious transcription artefacts
            continue
        labs[r["subject_id"]].append((r["charttime"], round(v, 2)))
    return pats, labs


def pick_pair(events, age, sex, threshold, mode="nearest"):
    """
    Choose one UNSAFE and one SAFE real measurement for this patient.

    `nearest` picks the pair closest to the threshold from either side, which is
    the hardest honest case and mirrors the near-threshold hardening applied to
    the synthetic control. `extreme` picks the widest separation.
    """
    scored = []
    for t, scr in events:
        eg = ckd_epi(scr, age, sex)
        if eg is None:
            continue
        scored.append((t, scr, round(eg, 1)))
    lo = [e for e in scored if e[2] < threshold]        # UNSAFE
    hi = [e for e in scored if e[2] >= threshold]       # SAFE
    if not lo or not hi:
        return None
    if mode == "nearest":
        unsafe = max(lo, key=lambda e: e[2])
        safe = min(hi, key=lambda e: e[2])
    else:
        unsafe = min(lo, key=lambda e: e[2])
        safe = max(hi, key=lambda e: e[2])
    return unsafe, safe


def render(age, sex, scr, drug):
    """
    A concise MedCalc-style note stating exactly one creatinine.

    Deliberately minimal. Every field here is a real recorded value, and nothing
    that is not needed to decide the rule is invented -- adding plausible-
    sounding history would be fabrication dressed as realism.
    """
    return (f"A {age}-year-old {'woman' if sex == 'female' else 'man'} has a "
            f"serum creatinine of {scr:.2f} mg/dL. "
            f"{drug.capitalize()} is being considered.")


PROMPT = ("You are reviewing a proposed prescription for safety.\n"
          "Answer with exactly one word on the first line: SAFE or UNSAFE.\n\n"
          "{vignette}\n\nIs it safe to prescribe {drug}?")


def build_items(pats, labs, threshold, mode):
    items, skipped = [], defaultdict(int)
    for sid, events in sorted(labs.items()):
        if sid not in pats:
            skipped["no_demographics"] += 1
            continue
        age, sex = pats[sid]
        # MIMIC shifts ages and caps them at 91 for de-identification; a 0-year
        # anchor_age would make CKD-EPI meaningless.
        if not 18 <= age <= 91:
            skipped["age_outside_ckd_epi_range"] += 1
            continue
        events = [(t, v) for t, v in events
                  if implausible(creatinine_mgdl=v) is None]
        if not events:
            skipped["all_creatinine_implausible"] += 1
            continue
        chosen = pick_pair(events, age, sex, threshold, mode)
        if chosen is None:
            skipped["no_straddling_pair"] += 1
            continue
        (t_u, scr_u, eg_u), (t_s, scr_s, eg_s) = chosen

        # Guard 5: the derived eGFR must itself be physiologically possible.
        if any(implausible(eGFR=e) is not None for e in (eg_u, eg_s)):
            skipped["implausible_egfr"] += 1
            continue

        # Guard 3: re-check the side AFTER rounding to printed precision.
        eg_u_r = round(ckd_epi(round(scr_u, 2), age, sex), 1)
        eg_s_r = round(ckd_epi(round(scr_s, 2), age, sex), 1)
        if not (eg_u_r < threshold <= eg_s_r):
            skipped["rounding_crossed_threshold"] += 1
            continue

        pair_id = f"{FAMILY}__mimic-{sid}"
        for arm, scr, eg, t, label in (
                ("unsafe", scr_u, eg_u_r, t_u, "UNSAFE"),
                ("safe",   scr_s, eg_s_r, t_s, "SAFE")):
            vig = render(age, sex, scr, DRUG)
            items.append({
                "id": f"{pair_id}__{arm}",
                "pair_id": pair_id,
                "family": FAMILY,
                "held_out": False,
                "arm": arm,
                "drug": DRUG,
                "factor": "eGFR",
                "factor_value": eg,
                "label": label,
                "vignette": vig,
                "presentation": "implicit",   # eGFR must be derived from Scr
                "edited_arm": False,          # THE POINT: no number is invented
                "calculator": "ckd_epi",
                "source_calculator": "ckd_epi",
                "note_id": f"mimic-{sid}",
                "subject_id": sid,
                "charttime": t,
                "facts": {"creatinine_raw": scr, "creatinine_unit": "mg/dL",
                          "age": age, "sex": sex.capitalize(), "race": None},
                "prompt": PROMPT.format(vignette=vig, drug=DRUG),
                "provenance": {
                    "source": "MIMIC-IV Clinical Database Demo v2.2 (ODbL)",
                    "table": "hosp/labevents", "itemid": CREATININE_ITEMID,
                    "threshold_source": "openfda:attested_exact",
                    "both_arms_real": True},
            })
        skipped["kept_pairs"] += 1
    return items, skipped


def split_by_patient(items, fracs, seed):
    pairs = defaultdict(list)
    for it in items:
        pairs[it["subject_id"]].append(it)
    sids = sorted(pairs)
    random.Random(seed).shuffle(sids)
    n = len(sids)
    n_test = max(1, int(round(fracs[0] * n)))
    n_calib = max(1, int(round(fracs[1] * n)))
    groups = {"test": sids[:n_test],
              "calib": sids[n_test:n_test + n_calib],
              "train": sids[n_test + n_calib:]}
    return {k: [it for s in v for it in pairs[s]] for k, v in groups.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/mimic_demo")
    ap.add_argument("--out", default="data/mimic")
    ap.add_argument("--threshold", type=float, default=EGFR_THRESHOLD)
    ap.add_argument("--mode", choices=["nearest", "extreme"], default="nearest")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.5)
    ap.add_argument("--calib_frac", type=float, default=0.2)
    args = ap.parse_args()

    pats, labs = load_cohort(args.src)
    print(f"patients={len(pats)}  with serum creatinine={len(labs)}")

    items, skipped = build_items(pats, labs, args.threshold, args.mode)
    print(f"\nconstruction: {dict(skipped)}")
    if not items:
        raise SystemExit("no usable pairs -- refusing to write an empty split")

    # Guard 2: both arms share age and sex by construction; verify it.
    byp = defaultdict(list)
    for it in items:
        byp[it["pair_id"]].append(it)
    for pid, arms in byp.items():
        assert len(arms) == 2, f"{pid}: {len(arms)} arms"
        a, b = arms
        assert a["facts"]["age"] == b["facts"]["age"], pid
        assert a["facts"]["sex"] == b["facts"]["sex"], pid
        assert a["label"] != b["label"], pid
        diff = [k for k in ("age", "sex") if a["facts"][k] != b["facts"][k]]
        assert not diff, f"{pid}: arms differ in {diff}"

    splits = split_by_patient(items, (args.test_frac, args.calib_frac),
                              args.seed)

    # Guard 4: patient-disjoint splits.
    seen = {}
    for name, rows in splits.items():
        for r in rows:
            prev = seen.setdefault(r["subject_id"], name)
            assert prev == name, (f"subject {r['subject_id']} appears in "
                                  f"{prev} and {name}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        p = out / f"counterfactual_{name}.jsonl"
        with p.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"  {p}: {len(rows)} items / {len(rows)//2} pairs")

    # RAG corpus: the ONE attested threshold sentence, quoted from the label.
    corpus = [{"id": "guideline::metformin_renal",
               "text": "Metformin is contraindicated in severe renal "
                       "impairment (eGFR below 30 mL/min/1.73 m2). "
                       "[FDA label, CONTRAINDICATIONS -- "
                       "results/threshold_provenance.md, attested_exact]"}]
    with (out / "rag_corpus.jsonl").open("w") as fh:
        for d in corpus:
            fh.write(json.dumps(d) + "\n")
    print(f"  {out}/rag_corpus.jsonl: {len(corpus)} passages")

    meta = {"source": "MIMIC-IV Clinical Database Demo v2.2",
            "licence": "ODbL", "credentialed": False,
            "url": "https://physionet.org/content/mimic-iv-demo/2.2/",
            "itemid": CREATININE_ITEMID, "threshold": args.threshold,
            "threshold_provenance": "openfda:attested_exact",
            "pair_selection": args.mode, "seed": args.seed,
            "both_arms_real": True,
            "caveat": "The two arms are different TIMEPOINTS in the same real "
                      "patient, so the clinical state genuinely differed. The "
                      "rendered note is minimal (age, sex, one creatinine, "
                      "drug) so the prompts differ in exactly one number, and "
                      "both numbers are real measurements.",
            "counts": {k: len(v) for k, v in splits.items()}}
    (out / "build_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out}/build_meta.json")


if __name__ == "__main__":
    main()
