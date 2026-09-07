"""
results/COMPARISON_BASE_VS_PROPOSED.md -- the baseline against the proposed
system, on every benchmark and every model that has been run.

This is the supervisor's question, answered in one table: what does the full
contributed system do that the base model does not?

Not to be confused with `results/rerun_fixes/COMPARISON.md`, which compares
the CODEBASE before and after the B1/B2/B3 layer-indexing fixes. That file
answers "did the defects change a conclusion". This one answers "do the
contributions help".

Everything is read from `preds_*.jsonl` and recomputed here -- bootstrap CIs
over items, exact McNemar with Holm correction across the reported
comparisons -- so no number in this file is copied from another file that
might be stale.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from make_table import LABELS, correct_flags, load_preds
from metrics import bootstrap_ci, holm_bonferroni, mcnemar, score

# (tag, split, human name). The tag is how run_eval.py namespaces the arm.
# Which dataset each tag was evaluated on, needed to recover `is_control`.
TAG_DATA = {
    "":          "data/synthetic_control",
    "_medcalc":  "data/medcalc",
    "_medcalc2": "data/medcalc_v2",
    "_mimic":    "data/mimic",
}

ARMS = [
    ("",          "test",    "synthetic control, test"),
    ("",          "heldout", "synthetic control, held-out"),
    ("_medcalc",  "test",    "real notes (MedCalc), test"),
    ("_medcalc",  "heldout", "real notes (MedCalc), held-out QT"),
    ("_medcalc2", "test",    "real notes v2 (+2 families, +controls), test"),
    ("_medcalc2", "heldout", "real notes v2, held-out QT"),
    ("_mimic",    "test",    "MIMIC-IV real values, test"),
    ("_mimic",    "heldout", "MIMIC-IV real values, held-out warfarin"),
]

# The single contribution each isolating row adds to the base model.
ISOLATING = [("sym", "Symbolic gate"), ("uq", "UQ engine"),
             ("cl", "Constraint layer"), ("rag", "RAG")]

PROPOSED = "nsai_uq_cl"      # all four contributions
FALLBACK = "nsai_uq"         # the proposal's row (4) where row (8) is absent


# Prediction files written before 2026-09-07 do not carry `is_control`, and
# re-running a 448-item ablation to recover one boolean per row would be
# absurd. The flag is a property of the DATASET, so it is joined back on `id`.
_CTRL_CACHE = {}


def control_ids(data_dir, split):
    key = (str(data_dir), split)
    if key in _CTRL_CACHE:
        return _CTRL_CACHE[key]
    p = Path(data_dir) / f"counterfactual_{split}.jsonl"
    ids = set()
    if p.is_file():
        for line in p.open():
            r = json.loads(line)
            if r.get("is_control"):
                ids.add(r["id"])
    _CTRL_CACHE[key] = ids
    return ids


def enrich(recs, data_dir, split):
    """Restore `is_control` from the dataset when the preds predate it."""
    if not recs or "is_control" in recs[0]:
        return recs
    ids = control_ids(data_dir, split)
    if not ids:
        return recs
    for r in recs:
        r["is_control"] = r["id"] in ids
    return recs


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def signed(x, nd=3):
    return "—" if x is None else f"{x:+.{nd}f}"


def arm_rows(R, tag, split, seed=0):
    """Everything one arm contributes, or None if it was not run."""
    data_dir = TAG_DATA.get(tag, "data/medcalc")
    try:
        base = enrich(load_preds(R, split, "base", seed, tag), data_dir, split)
    except FileNotFoundError:
        return None
    prop_name = PROPOSED
    try:
        prop = enrich(load_preds(R, split, prop_name, seed, tag),
                      data_dir, split)
    except FileNotFoundError:
        try:
            prop = enrich(load_preds(R, split, FALLBACK, seed, tag),
                          data_dir, split)
            prop_name = FALLBACK
        except FileNotFoundError:
            return None

    sb, sp = score(base), score(prop)
    fb, fp = correct_flags(base), correct_flags(prop)
    mc = mcnemar(fb, fp)
    out = {
        "n": sb["n"], "proposed_variant": prop_name,
        "base": sb, "proposed": sp,
        "acc_ci_base": bootstrap_ci(fb), "acc_ci_prop": bootstrap_ci(fp),
        "cc_ci_base": bootstrap_ci(sb["_cc_flags"]),
        "cc_ci_prop": bootstrap_ci(sp["_cc_flags"]),
        "mcnemar": mc,
        "isolating": {},
    }
    for v, label in ISOLATING:
        try:
            recs = enrich(load_preds(R, split, v, seed, tag),
                          data_dir, split)
        except FileNotFoundError:
            continue
        s = score(recs)
        f = correct_flags(recs)
        out["isolating"][v] = {
            "label": label, "score": s, "mcnemar": mcnemar(fb, f),
            "d_acc": s["accuracy"] - sb["accuracy"],
            "d_cc": s["causal_consistency"] - sb["causal_consistency"],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results",
                    help="BioMistral's artifact root")
    ap.add_argument("--models", nargs="*", default=[],
                    help="extra roots as slug=path, e.g. "
                         "llama3-openbiollm-8b=results/models/llama3-openbiollm-8b")
    ap.add_argument("--base_model", default="BioMistral/BioMistral-7B")
    ap.add_argument("--out", default="results/COMPARISON_BASE_VS_PROPOSED.md")
    args = ap.parse_args()

    roots = [(args.base_model, args.results)]
    for spec in args.models:
        slug, _, path = spec.partition("=")
        roots.append((slug, path))

    collected = []
    for model, root in roots:
        for tag, split, name in ARMS:
            r = arm_rows(root, tag, split)
            if r:
                r.update(model=model, root=root, arm=name, tag=tag, split=split)
                collected.append(r)
    if not collected:
        raise SystemExit("no prediction files found under the given roots")

    # Holm across every base-vs-proposed test reported in this file, which is
    # the family of comparisons a reader sees. Correcting per-table instead
    # would understate the multiplicity.
    pv = {f"{c['model']}|{c['arm']}": c["mcnemar"]["p_value"] for c in collected}
    holm = holm_bonferroni(pv)

    L = [
        "# Baseline against the proposed system",
        "",
        "*Generated by `src/make_comparison.py` from the prediction files. Do "
        "not edit the numbers by hand -- rerun the script.*",
        "",
        "**This is not `results/rerun_fixes/COMPARISON.md`.** That file "
        "compares the codebase before and after the B1/B2/B3 layer-indexing "
        "fixes and answers \"did the defects change a conclusion\". This file "
        "answers \"do the contributions help\".",
        "",
        "Row (1) is the base LLM alone. The proposed system is row (8), all "
        "four contributions together, falling back to row (4) on arms where "
        "row (8) was not run. Both columns are the same items, same decoding, "
        "same seed, so the McNemar test is paired.",
        "",
        "## 1. Headline",
        "",
        "| model | benchmark | n | acc base | acc proposed | Δ acc | "
        "CC base | CC proposed | Δ CC | viol base | viol proposed | "
        "coverage proposed | p (Holm) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in collected:
        b, p = c["base"], c["proposed"]
        L.append(
            f"| `{c['model']}` | {c['arm']} | {c['n']} | "
            f"{fmt(b['accuracy'])} | {fmt(p['accuracy'])} | "
            f"{signed(p['accuracy'] - b['accuracy'])} | "
            f"{fmt(b['causal_consistency'])} | {fmt(p['causal_consistency'])} | "
            f"{signed(p['causal_consistency'] - b['causal_consistency'])} | "
            f"{fmt(b['violation_rate'])} | {fmt(p['violation_rate'])} | "
            f"{fmt(p['coverage'])} | "
            f"{holm[c['model'] + '|' + c['arm']]:.3g} |")

    L += [
        "",
        "`viol` is the violation rate: an UNSAFE prescription called SAFE "
        "without abstaining. It is the number that matters clinically and it "
        "does not always move the same way as accuracy.",
        "",
        "## 2. 95% bootstrap CI over items",
        "",
        "| model | benchmark | accuracy base | accuracy proposed | "
        "CC base | CC proposed |",
        "|---|---|---|---|---|---|",
    ]
    for c in collected:
        ab, ap_, cb, cp = (c["acc_ci_base"], c["acc_ci_prop"],
                           c["cc_ci_base"], c["cc_ci_prop"])
        L.append(f"| `{c['model']}` | {c['arm']} | "
                 f"{fmt(c['base']['accuracy'])} [{fmt(ab[0])}, {fmt(ab[1])}] | "
                 f"{fmt(c['proposed']['accuracy'])} [{fmt(ap_[0])}, {fmt(ap_[1])}] | "
                 f"{fmt(c['base']['causal_consistency'])} [{fmt(cb[0])}, {fmt(cb[1])}] | "
                 f"{fmt(c['proposed']['causal_consistency'])} [{fmt(cp[0])}, {fmt(cp[1])}] |")

    L += [
        "",
        "## 3. Where the difference comes from",
        "",
        "Each row adds exactly ONE contribution to the base model, so the "
        "delta is attributable. A cumulative ladder cannot do this: on real "
        "notes RAG makes causal consistency *worse*, so any gain measured on "
        "top of RAG is measured from a damaged starting point.",
        "",
        "| model | benchmark | contribution | Δ acc | Δ CC | "
        "B01 (base wrong→right) | B10 (base right→wrong) | p (exact) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in collected:
        for v, d in c["isolating"].items():
            m = d["mcnemar"]
            L.append(f"| `{c['model']}` | {c['arm']} | {d['label']} | "
                     f"{signed(d['d_acc'])} | {signed(d['d_cc'])} | "
                     f"{m['b01']} | {m['b10']} | {m['p_value']:.3g} |")

    # ---- control pairs, where the arm has them --------------------------
    ctrl = [c for c in collected
            if c["base"].get("spurious_flip_rate") is not None]
    L += ["", "## 4. Is the consistency real? (control pairs)", ""]
    if not ctrl:
        L += ["No arm in this run carries control pairs, so every Causal "
              "Consistency number above conflates causal sensitivity with "
              "plain prompt sensitivity. `data/medcalc_v2` adds them; run "
              "`run_eval.py --data data/medcalc_v2 --tag _medcalc2` to fill "
              "this section in.", ""]
    else:
        L += [
            "A control pair moves the driving value by a comparable amount "
            "**without crossing the threshold**, so the label does not "
            "change. A model that reacts to any prompt edit flips on these "
            "too. Discrimination is the causal flip rate minus the spurious "
            "one; on 8,000 MCQ items this model scored −0.013 "
            "[−0.037, +0.013] (`results/mcqpairs.md`), which is why the "
            "column exists.",
            "",
            "| model | benchmark | variant | causal flip | spurious flip | "
            "discrimination | n control pairs |",
            "|---|---|---|---|---|---|---|",
        ]
        for c in ctrl:
            for tagname, s in (("base", c["base"]),
                               (c["proposed_variant"], c["proposed"])):
                L.append(f"| `{c['model']}` | {c['arm']} | {tagname} | "
                         f"{fmt(s['causal_flip_rate'])} | "
                         f"{fmt(s['spurious_flip_rate'])} | "
                         f"{signed(s['discrimination'])} | "
                         f"{s['n_control_pairs']} |")
        L += [""]

    L += [
        "## 5. Two things that must travel with this table",
        "",
        "**The gate's accuracy is partly circular.** On the real-notes "
        "benchmark the symbolic gate applies the same rule and threshold the "
        "labels were generated from, and the dataset was filtered with the "
        "same extractor that runs inside the gate. The number that is not "
        "circular is its **coverage**: how often it can fire at all. See "
        "`results/SUMMARY.md` and the per-family coverage in the manuscript.",
        "",
        "**`ondansetron_qt` is ungrounded at both ends.** Its 500 ms "
        "threshold is absent from the FDA labels, and the family is absent "
        "from MED-RT, so the causal knowledge graph licenses no "
        "contraindication path for it. It is the family carrying the Aim 3 "
        "constraint-layer result, so that result rests on curation at both "
        "ends.",
        "",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}  ({len(collected)} arm(s) across "
          f"{len(roots)} model root(s))")


if __name__ == "__main__":
    main()
