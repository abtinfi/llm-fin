"""
Builds a large counterfactual drug-safety benchmark from MedCalc-Bench.

WHY
---
The hand-built benchmark in `data/` is 128 items over synthetic vignettes. It is
big enough to show the symbolic gate works and too small to show much else --
see `results/bigbench_uq.md`, where the 56-item gate-declined pool could not
distinguish a real signal from chance. This builds the same task shape at ~10x
the size on **real clinical text**.

Source: `ncbi/MedCalc-Bench-v1.2` -- patient notes drawn from PMC case reports,
each with structured `Relevant Entities` and a verified `Ground Truth Answer`
for a named clinical calculator.

CONSTRUCTION
------------
For each usable note we know the true value of a clinical quantity (eGFR, QTc)
because MedCalc-Bench states it and `src/renal.py` reproduces it. A published
threshold on that quantity decides whether a named drug is safe.

One arm of each pair is the **real, unedited note**. The other arm is the same
note with exactly one number changed -- the number that drives the formula --
so the computed value crosses the threshold and the label flips. Everything
else in the note is byte-identical. That is a genuine minimal pair: any change
in the model's answer is attributable to that one edit and nothing else.

FIVE THINGS THAT WOULD SILENTLY CORRUPT THIS, AND THE GUARDS AGAINST THEM
------------------------------------------------------------------------
1. *A wrong formula* would mislabel every item. Guard: every row is checked
   against MedCalc-Bench's own `Ground Truth Answer` / limits, and rows the
   formula cannot reproduce are dropped, not repaired.

2. *An ambiguous edit.* If the driving number appears more than once in the
   note, replacing it could alter an unrelated statement. Guard: the number
   must occur exactly once.

3. *A self-contradicting note.* If the note already states the eGFR (or QTc) in
   words, editing the creatinine (or QT) leaves the note asserting two
   incompatible values -- this is exactly bug #2 in HANDOFF.md section 5, where
   an implicit vignette stated an age that contradicted its own ground truth.
   Guard: notes that state the derived quantity numerically are excluded.

4. *Rounding drift.* The solver returns a float; the note must contain a
   plausibly-written number. Guard: the value is rounded to the original's
   precision FIRST, then the label is recomputed from the rounded value, so the
   label always follows from the text a reader actually sees.

5b. *Physiologically impossible source data.* MedCalc-Bench contains entity
   rows whose stated unit cannot be right -- e.g. `pmc-6997309-1` gives
   creatinine as "1.1 µmol/L", roughly 1/60th of the lower limit of normal,
   almost certainly 1.1 mg/dL mislabelled. Their own pipeline propagates it and
   reports an eGFR of 12,129 mL/min/1.73m2 against a physiological maximum near
   120. The formula check passes on such a row because the formula *is* being
   applied correctly to a nonsense input. Guard: creatinine and the derived
   quantity must both fall in a physiologically possible range, or the row is
   dropped.

6. *A boundary case.* A pair whose two arms sit either side of the threshold by
   a hair is not a test of reasoning, it is a test of arithmetic precision.
   Guard: a minimum margin is enforced on both arms.

Held-out split is the QT family, mirroring the original benchmark's design
where `ondansetron_qt` is held out: at evaluation time it exercises a rule the
symbolic gate has a definition for but which never appeared in the test split.
"""

import argparse
import ast
import json
import math
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from renal import (CALCULATORS, compute, to_mgdl, solve_creatinine,  # noqa: E402
                   ckd_epi, qtc_bazett, PLAUSIBLE, implausible)
from medcalc_gate import creatinine_mentions                        # noqa: E402


# --------------------------------------------------------------------------
# Rules. Both thresholds are quoted from FDA labelling, not written from
# memory -- this is the P4 problem being avoided rather than repeated.
# --------------------------------------------------------------------------
RULES = {
    "metformin_renal": {
        "drug": "metformin",
        "quantity": "eGFR",
        "unit": "mL/min/1.73m2",
        "threshold": 30.0,
        "unsafe_when": "below",
        "proposal": "The team proposes starting metformin 1000 mg twice daily.",
        "source": ("FDA label, metformin hydrochloride: contraindicated in "
                   "severe renal impairment (eGFR below 30 mL/min/1.73 m2)."),
        "calculators": ["mdrd", "ckd_epi"],
        "held_out": False,
    },
    "ondansetron_qt": {
        "drug": "ondansetron",
        "quantity": "QTc",
        "unit": "ms",
        "threshold": 500.0,
        "unsafe_when": "above",
        "proposal": "The team proposes giving intravenous ondansetron 8 mg.",
        "source": ("FDA label, ondansetron: avoid in patients with congenital "
                   "long QT syndrome; ECG monitoring is recommended and the "
                   "drug should be avoided when QTc exceeds 500 ms."),
        "calculators": ["qtc_bazett"],
        "held_out": True,
    },
}

PROMPT = ("You are reviewing a proposed prescription for safety.\n"
          "Answer with exactly one word on the first line: SAFE or UNSAFE.\n"
          "Then give one short sentence of justification.\n\nCase:\n{v}")

# Phrases that mean the note already states the derived quantity. If any of
# these appears with a number attached, editing the driver would contradict it.
DERIVED_MENTIONS = {
    "eGFR": [r"\begfr\b", r"\bgfr\b", r"glomerular filtration",
             r"creatinine clearance", r"\bcrcl\b", r"\bccr\b"],
    "QTc": [r"\bqtc\b", r"corrected qt"],
}




def num_strings(v):
    """Plausible written forms of a number, for locating it in prose."""
    out = {f"{v:g}", str(v)}
    if float(v).is_integer():
        out |= {str(int(v)), f"{int(v)}.0"}
    for d in (1, 2, 3):
        out.add(f"{v:.{d}f}")
    return {s for s in out if s}


def locate_unique(note, value):
    """Span of `value` in `note`, or None unless it occurs exactly once."""
    spans = set()
    for s in num_strings(value):
        for m in re.finditer(r"(?<![\d.])" + re.escape(s) + r"(?![\d])", note):
            spans.add((m.start(), m.end()))
    if len(spans) != 1:
        return None
    return spans.pop()


def decimals_of(span_text):
    return len(span_text.split(".")[1]) if "." in span_text else 0


def states_derived(note, quantity):
    low = note.lower()
    for pat in DERIVED_MENTIONS[quantity]:
        for m in re.finditer(pat, low):
            # only a problem if a number follows within a short window
            if re.search(r"\d", low[m.end():m.end() + 40]):
                return True
    return False


def label_for(rule, value):
    unsafe = (value < rule["threshold"] if rule["unsafe_when"] == "below"
              else value > rule["threshold"])
    return "UNSAFE" if unsafe else "SAFE"


def pick_target(rule, want_unsafe, rng, margin_lo, margin_hi):
    """A value on the requested side, a random but bounded distance away."""
    t, d = rule["threshold"], rng.uniform(margin_lo, margin_hi)
    below = (rule["unsafe_when"] == "below") == want_unsafe
    return t - d if below else t + d


def build_renal(df, rng, cfg):
    """
    One item family, one rule, one equation.

    Labels are assigned with **CKD-EPI 2021 for every note**, not with whichever
    calculator MedCalc-Bench happened to pair with that note. A benchmark whose
    ground truth depends on which row of the source file an item came from is
    not measuring a clinical rule, it is measuring a bookkeeping artefact -- and
    a gate would need oracle knowledge of the source row to score well on it.
    CKD-EPI 2021 is the current clinical standard and is what a reviewer would
    actually use.

    MDRD rows are still kept, and their MDRD value is still checked against
    MedCalc's ground truth. That check is what establishes that the note's
    age / sex / creatinine were extracted correctly; the label is then derived
    from those validated entities via CKD-EPI. Validated inputs plus a
    validated equation is sound even where MedCalc never computed CKD-EPI.
    """
    rule = RULES["metformin_renal"]
    rows, drops, seen_notes = [], {}, set()
    sub = df[df["Calculator Name"].isin(
        [k for k, v in CALCULATORS.items() if v in ("mdrd", "ckd_epi")])]

    for _, r in sub.iterrows():
        def drop(why):
            drops[why] = drops.get(why, 0) + 1
        kind = CALCULATORS[r["Calculator Name"]]
        note = str(r["Patient Note"])
        note_id = str(r["Note ID"])

        # The same note appears once per calculator. With a single labelling
        # equation those would be duplicate items, inflating n and breaking
        # the independence the paired tests assume.
        if note_id in seen_notes:
            drop("duplicate note (same note under another calculator)")
            continue

        try:
            ents = ast.literal_eval(r["Relevant Entities"])
            raw, unit = float(ents["creatinine"][0]), ents["creatinine"][1]
            scr = to_mgdl(raw, unit)
            mine = compute(kind, scr, ents)
        except Exception:
            drop("unit or entity not parseable"); continue

        # Guard 1: the source row's own calculator must reproduce, which is
        # what certifies the extracted entities.
        if not (float(r["Lower Limit"]) <= mine <= float(r["Upper Limit"])):
            drop("formula disagrees with MedCalc ground truth"); continue

        # Guard 2+3: the note must state creatinine exactly once, with a unit,
        # and it must be the value MedCalc extracted. Anything else has no
        # unambiguous ground truth -- see creatinine_mentions().
        men = creatinine_mentions(note)
        if not men:
            drop("creatinine not stated with a unit"); continue
        # Repetition is fine, disagreement is not. "creatinine 2.1 mg/dL ...
        # his creatinine of 2.1 mg/dL" has one answer; "creatinine rose from
        # 1.2 to 3.4 mg/dL" has none, and no reader could say which the label
        # refers to. Every mention is edited so the counterfactual note stays
        # internally consistent.
        distinct = {round(m["mgdl"], 6) for m in men}
        if len(distinct) != 1:
            drop(f"creatinine stated with {len(distinct)} different values")
            continue
        if abs(men[0]["mgdl"] - scr) > 1e-6:
            drop("note's creatinine differs from MedCalc's entity"); continue
        if states_derived(note, "eGFR"):
            drop("note already states eGFR/CrCl"); continue
        if len(note.split()) > cfg["max_words"]:
            drop("note too long"); continue

        # Label with the single clinical equation.
        try:
            egfr = ckd_epi(scr, float(ents["age"][0]), ents.get("sex", "Male"))
        except Exception:
            drop("CKD-EPI not computable"); continue
        bad = implausible(creatinine_mgdl=scr, eGFR=egfr)
        if bad:
            drop(f"implausible source data ({bad})"); continue
        if abs(egfr - rule["threshold"]) < cfg["margin_lo"]:
            drop("real value too close to threshold"); continue

        real_label = label_for(rule, egfr)
        spans = sorted({m["num_span"] for m in men})
        a, b = spans[0]
        dec = decimals_of(note[a:b])
        if any(decimals_of(note[x:y]) != dec for x, y in spans):
            drop("creatinine written with differing precision"); continue

        # Guard 4: solve, round to the note's own precision, then re-derive the
        # label from the rounded number a reader actually sees.
        target = pick_target(rule, real_label == "SAFE", rng,
                             cfg["margin_lo"], cfg["margin_hi"])
        solved = solve_creatinine("ckd_epi", target,
                                  {"age": ents["age"], "sex": ents.get("sex")})
        if solved is None:
            drop("counterfactual unreachable for this patient"); continue
        new_raw = solved * 88.4 if "mol" in unit.lower() else solved
        new_raw = round(new_raw, dec)
        if new_raw <= 0:
            drop("edited creatinine non-physical"); continue
        try:
            new_scr = to_mgdl(new_raw, unit)
            new_val = ckd_epi(new_scr, float(ents["age"][0]),
                              ents.get("sex", "Male"))
        except Exception:
            drop("edited creatinine not convertible"); continue
        if label_for(rule, new_val) == real_label:
            drop("rounding did not flip the label"); continue
        if abs(new_val - rule["threshold"]) < cfg["margin_lo"]:
            drop("edited value too close to threshold"); continue
        bad = implausible(creatinine_mgdl=new_scr, eGFR=new_val)
        if bad:
            drop(f"edited value implausible ({bad})"); continue

        new_txt = f"{new_raw:.{dec}f}" if dec else str(int(new_raw))
        # Rewrite every occurrence, back to front so earlier offsets stay valid.
        cf_note = note
        for x, y in sorted(spans, reverse=True):
            cf_note = cf_note[:x] + new_txt + cf_note[y:]

        # The edit must leave the note internally consistent: same number of
        # mentions, all now agreeing on the new value.
        cf_men = creatinine_mentions(cf_note)
        if len(cf_men) != len(men):
            drop("edit changed how many creatinine mentions the note has")
            continue
        if len({round(m["mgdl"], 6) for m in cf_men}) != 1:
            drop("edit left the note stating disagreeing creatinines")
            continue

        seen_notes.add(note_id)
        cf_label = label_for(rule, new_val)
        pid = f"metformin_renal__{note_id}"
        for note_txt, val, lab, arm, edited, craw in (
                (note, egfr, real_label, real_label.lower(), False, raw),
                (cf_note, new_val, cf_label, cf_label.lower(), True, new_raw)):
            rows.append({
                "id": f"{pid}__{arm}", "pair_id": pid,
                "family": "metformin_renal", "held_out": False, "arm": arm,
                "drug": "metformin", "factor": "eGFR",
                "factor_value": round(val, 2), "label": lab,
                "vignette": note_txt.strip() + "\n" + rule["proposal"],
                "presentation": "implicit", "edited_arm": edited,
                "calculator": "ckd_epi", "source_calculator": kind,
                "note_id": note_id,
                "facts": {"creatinine_raw": craw, "creatinine_unit": unit,
                          "age": ents["age"][0], "sex": ents.get("sex"),
                          "race": ents.get("Race")},
                "prompt": PROMPT.format(v=note_txt.strip() + "\n"
                                        + rule["proposal"]),
            })
    return rows, drops


def build_qt(df, rng, cfg):
    rule = RULES["ondansetron_qt"]
    rows, drops = [], {}
    sub = df[df["Calculator Name"] == "QTc Bazett Calculator"]

    for _, r in sub.iterrows():
        def drop(why):
            drops[why] = drops.get(why, 0) + 1
        note = str(r["Patient Note"])
        try:
            ents = ast.literal_eval(r["Relevant Entities"])
            hr = float(ents["Heart Rate or Pulse"][0])
            qt = float(ents["QT Interval"][0])
            mine = qtc_bazett(qt, hr)
        except Exception:
            drop("entities not parseable"); continue

        bad = implausible(qt_ms=qt, heart_rate=hr, QTc=mine)
        if bad:
            drop(f"implausible source data ({bad})"); continue

        gt = float(r["Ground Truth Answer"])
        if not (float(r["Lower Limit"]) <= mine <= float(r["Upper Limit"])):
            drop("formula disagrees with MedCalc ground truth"); continue
        if abs(gt - rule["threshold"]) < cfg["qt_margin_lo"]:
            drop("real value too close to threshold"); continue
        if states_derived(note, "QTc"):
            drop("note already states QTc"); continue
        span = locate_unique(note, qt)
        if span is None:
            drop("QT interval not uniquely locatable"); continue
        if len(note.split()) > cfg["max_words"]:
            drop("note too long"); continue

        a, b = span
        real_label = label_for(rule, gt)
        target = pick_target(rule, real_label == "SAFE", rng,
                             cfg["qt_margin_lo"], cfg["qt_margin_hi"])
        # QTc is linear in QT, so the inverse is closed-form.
        new_qt = round(target * math.sqrt(60.0 / hr))
        if new_qt <= 0:
            drop("edited QT non-physical"); continue
        new_val = qtc_bazett(new_qt, hr)
        cf_label = label_for(rule, new_val)
        if cf_label == real_label:
            drop("rounding did not flip the label"); continue
        if abs(new_val - rule["threshold"]) < cfg["qt_margin_lo"]:
            drop("edited value too close to threshold"); continue
        bad = implausible(qt_ms=new_qt, QTc=new_val)
        if bad:
            drop(f"edited value implausible ({bad})"); continue

        cf_note = note[:a] + str(int(new_qt)) + note[b:]
        pid = f"ondansetron_qt__{r['Note ID']}"
        for note_txt, val, lab, arm, edited, qtv in (
                (note, gt, real_label, real_label.lower(), False, qt),
                (cf_note, new_val, cf_label, cf_label.lower(), True, new_qt)):
            rows.append({
                "id": f"{pid}__{arm}", "pair_id": pid,
                "family": "ondansetron_qt", "held_out": True, "arm": arm,
                "drug": "ondansetron", "factor": "QTc",
                "factor_value": round(val, 2), "label": lab,
                "vignette": note_txt.strip() + "\n" + rule["proposal"],
                "presentation": "implicit", "edited_arm": edited,
                "calculator": "qtc_bazett", "note_id": str(r["Note ID"]),
                "facts": {"qt_ms": qtv, "heart_rate": hr},
                "prompt": PROMPT.format(v=note_txt.strip() + "\n"
                                        + rule["proposal"]),
            })
    return rows, drops


def rag_corpus(fda_path):
    """
    Guideline passages plus distractors.

    The two guideline passages quote FDA labelling. Where the real label text
    was already pulled down by `fetch_openfda.py` it is used verbatim; the
    curated summary is the fallback. Distractors are real passages about the
    same drugs that do NOT answer the question, so retrieval has to discriminate
    rather than merely find the drug name.
    """
    docs = []
    for name, rule in RULES.items():
        docs.append({"id": f"guideline::{name}", "text": rule["source"]})

    distract = [
        ("distractor::metformin_gi",
         "Gastrointestinal upset is the most common adverse effect of "
         "metformin. Titrating the dose slowly and taking it with food "
         "reduces nausea and diarrhoea."),
        ("distractor::metformin_b12",
         "Long-term metformin use is associated with reduced vitamin B12 "
         "absorption. Periodic measurement is reasonable in patients with "
         "anaemia or peripheral neuropathy."),
        ("distractor::ondansetron_moa",
         "Ondansetron is a selective 5-HT3 receptor antagonist used to "
         "prevent nausea and vomiting associated with chemotherapy, "
         "radiotherapy and surgery."),
        ("distractor::ondansetron_constipation",
         "Constipation and headache are the most frequently reported adverse "
         "effects of ondansetron and are usually self-limiting."),
        ("distractor::ckd_staging",
         "Chronic kidney disease is staged by estimated glomerular filtration "
         "rate: stage 3a is 45-59, stage 3b is 30-44, stage 4 is 15-29 and "
         "stage 5 is below 15 mL/min/1.73m2."),
        ("distractor::qt_causes",
         "Hypokalaemia, hypomagnesaemia and bradycardia all prolong the QT "
         "interval and should be corrected before attributing prolongation to "
         "a drug."),
        ("distractor::insulin",
         "Insulin therapy is not contraindicated by renal impairment, though "
         "requirements often fall as clearance declines."),
        ("distractor::hydration",
         "Adequate hydration reduces the risk of contrast-induced nephropathy "
         "in patients undergoing imaging with iodinated contrast."),
    ]
    docs += [{"id": i, "text": t} for i, t in distract]

    p = Path(fda_path)
    if p.exists():
        n = 0
        for line in p.open():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            drug = str(rec.get("query_drug", "")).lower()
            # openfda_raw.jsonl nests the label under a "label" key; the
            # relevant sections are lists of strings.
            label = rec.get("label") or {}
            if drug not in ("metformin", "ondansetron"):
                continue
            for field in ("contraindications", "warnings_and_precautions",
                          "boxed_warning"):
                val = label.get(field)
                if not val:
                    continue
                txt = " ".join(val) if isinstance(val, list) else str(val)
                txt = re.sub(r"\s+", " ", txt).strip()
                if len(txt) < 80:
                    continue
                docs.append({"id": f"fda::{drug}::{field}", "text": txt[:1200]})
                n += 1
        print(f"  added {n} real FDA label passages from {p}")
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/external")
    ap.add_argument("--out", default="data/medcalc")
    ap.add_argument("--fda", default="data/openfda_raw.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_pairs", type=int, default=140,
                    help="renal pairs reserved for TRAINING the Aim 3 "
                         "constraint layer. Disjoint from calib and test by "
                         "pair_id, so no note ever appears on both sides.")
    ap.add_argument("--calib_pairs", type=int, default=25)
    ap.add_argument("--qt_train_pairs", type=int, default=55,
                    help="pairs of the held-out QT family reserved for "
                         "training the constraint layer. The remaining pairs "
                         "are the QT evaluation split and are never trained "
                         "on by anything.")
    ap.add_argument("--max_words", type=int, default=900)
    ap.add_argument("--margin_lo", type=float, default=3.0)
    ap.add_argument("--margin_hi", type=float, default=12.0)
    ap.add_argument("--qt_margin_lo", type=float, default=15.0)
    ap.add_argument("--qt_margin_hi", type=float, default=60.0)
    args = ap.parse_args()

    cfg = {"max_words": args.max_words, "margin_lo": args.margin_lo,
           "margin_hi": args.margin_hi, "qt_margin_lo": args.qt_margin_lo,
           "qt_margin_hi": args.qt_margin_hi}
    rng = random.Random(args.seed)

    src = Path(args.src)
    df = pd.concat([pd.read_csv(src / "medcalc_train_data_11_18_final.csv"),
                    pd.read_csv(src / "medcalc_test_data_11_18_final.csv")],
                   ignore_index=True)
    print(f"MedCalc-Bench rows: {len(df)}")

    renal, d1 = build_renal(df, rng, cfg)
    qt, d2 = build_qt(df, rng, cfg)
    print(f"\nrenal: {len(renal)} records ({len(renal)//2} pairs)")
    for k, v in sorted(d1.items(), key=lambda kv: -kv[1]):
        print(f"    dropped {v:4d}  {k}")
    print(f"QT   : {len(qt)} records ({len(qt)//2} pairs)")
    for k, v in sorted(d2.items(), key=lambda kv: -kv[1]):
        print(f"    dropped {v:4d}  {k}")

    # Split by PAIR so both arms always land in the same split, and so the
    # constraint layer can never be trained on a note it is later scored on.
    pairs = sorted({r["pair_id"] for r in renal})
    rng.shuffle(pairs)
    n_tr, n_ca = args.train_pairs, args.calib_pairs
    train_ids = set(pairs[:n_tr])
    calib_ids = set(pairs[n_tr:n_tr + n_ca])
    train = [r for r in renal if r["pair_id"] in train_ids]
    calib = [r for r in renal if r["pair_id"] in calib_ids]
    test = [r for r in renal if r["pair_id"] not in train_ids | calib_ids]
    assert not (train_ids & calib_ids)
    assert not ({r["pair_id"] for r in test} & (train_ids | calib_ids))

    # The held-out QT family is split again, by pair, into the pairs the
    # constraint layer of Aim 3 is trained on and the pairs it is scored on.
    #
    # This used to be done inline inside constraint_layer.py (`--train_on
    # heldout_first` took the first 55 sorted pair ids). That was leak-free for
    # the adapter itself, but it left `counterfactual_heldout.jsonl` containing
    # both halves, so the ablation table's held-out rows were scored on a split
    # that included the adapter's own training notes. A constraint-layer row
    # could not honestly be put in that table. Emitting the two halves as
    # separate files makes the eval split leak-free for EVERY row, and
    # `heldout_all` is kept so the earlier 85-pair numbers stay reproducible.
    qt_pairs = sorted({r["pair_id"] for r in qt})
    qt_train_ids = set(qt_pairs[:args.qt_train_pairs])
    qt_train = [r for r in qt if r["pair_id"] in qt_train_ids]
    qt_eval = [r for r in qt if r["pair_id"] not in qt_train_ids]
    assert not ({r["pair_id"] for r in qt_eval} & qt_train_ids)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("calib", calib), ("test", test),
                       ("heldout_train", qt_train), ("heldout", qt_eval),
                       ("heldout_all", qt)):
        p = out / f"counterfactual_{name}.jsonl"
        with p.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        pairs_n = len({r["pair_id"] for r in rows})
        labs = {}
        for r in rows:
            labs[r["label"]] = labs.get(r["label"], 0) + 1
        print(f"wrote {p}  n={len(rows)}  pairs={pairs_n}  labels={labs}")

    docs = rag_corpus(args.fda)
    with (out / "rag_corpus.jsonl").open("w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    print(f"wrote {out}/rag_corpus.jsonl  n={len(docs)}")


if __name__ == "__main__":
    main()
