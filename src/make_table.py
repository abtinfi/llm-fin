"""
Builds the deliverable: the ablation table, each contribution measured against
the BASELINE, plus paired significance tests.

  python src/make_table.py --split test

WHAT CHANGED AND WHY
--------------------
1. **Every row is compared against the base model**, not only against the row
   above it. A cumulative ladder cannot attribute an effect to a component:
   on real clinical notes RAG moves Causal Consistency from 0.023 to 0.000, so
   the symbolic gate's "+0.986" was measured from a damaged intermediate state
   rather than from the baseline. Both comparisons are now reported -- delta
   vs. base in the main table, and the consecutive-step tests below it.

2. **Isolated rows.** `+ Symbolic Gate only` and `+ UQ only` add one component
   to the base model directly. That is the number that answers "what does this
   contribution buy me", which is what the supervisor asked for.

3. **No fake error bars.** Decoding is greedy, so re-running with a different
   seed reproduces the run exactly and a seed-to-seed standard deviation is
   0.000 by construction, not by measurement. Printing `± 0.000` implied a
   variance estimate that was never made. With a single seed the table now
   prints the point estimate and a 95% bootstrap CI **over items**, which is
   the uncertainty that actually exists here. `±` reappears only when more than
   one seed is present.

4. **The gate's coverage is a column, not a footnote.** Its accuracy is
   partly circular -- the same extraction logic filters the benchmark and runs
   inside the gate -- so the honest headline is how often it can decide at all.
   A per-row breakdown over the gate-fired and gate-declined subsets is printed
   underneath, which is the part of the gate's contribution that is NOT
   circular.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from metrics import bootstrap_ci, holm_bonferroni, mcnemar

# display order; rows absent from the summary are skipped
# display order = the numbering in LABELS, so the table reads 1..8
VARIANTS = ["base", "rag", "nsai", "nsai_uq",
            "sym", "uq", "cl", "nsai_uq_cl"]
LABELS = {
    "base":       "(1) Base LLM",
    "rag":        "(2) + RAG",
    "nsai":       "(3) + Symbolic Gate (NS-AI)",
    "nsai_uq":    "(4) + UQ Engine (NS-AI+UQ)",
    "sym":        "(5) Base + Symbolic Gate only",
    "uq":         "(6) Base + UQ only",
    "cl":         "(7) Base + Constraint Layer only",
    "nsai_uq_cl": "(8) All four contributions",
}
# Base / RAG / Sym / UQ / Constraint layer
COMPONENTS = {
    "base":       ("YES", "-", "-", "-", "-"),
    "rag":        ("YES", "YES", "-", "-", "-"),
    "nsai":       ("YES", "YES", "YES", "-", "-"),
    "nsai_uq":    ("YES", "YES", "YES", "YES", "-"),
    "sym":        ("YES", "-", "YES", "-", "-"),
    "uq":         ("YES", "-", "-", "YES", "-"),
    "cl":         ("YES", "-", "-", "-", "YES"),
    "nsai_uq_cl": ("YES", "YES", "YES", "YES", "YES"),
}
# the cumulative ladder of section 4.6, for the step-by-step tests
LADDER = ["base", "rag", "nsai", "nsai_uq"]


def load_preds(results, split, variant, seed, tag=""):
    p = Path(results) / f"preds_{split}_{variant}_seed{seed}{tag}.jsonl"
    return [json.loads(l) for l in p.open()]


def restore_control_flag(recs, data_dir, split):
    """Preds written before 2026-09-07 carry no `is_control`; it is a property
    of the dataset, so join it back on `id` (as make_comparison.enrich does --
    not imported from there, which imports this module)."""
    if not recs or "is_control" in recs[0] or not data_dir:
        return recs
    p = Path(data_dir) / f"counterfactual_{split}.jsonl"
    if not p.is_absolute() and not p.is_file():
        p = Path(__file__).resolve().parent.parent / p
    if not p.is_file():
        return recs
    ctrl = {r["id"] for r in map(json.loads, p.open()) if r.get("is_control")}
    for r in recs:
        r["is_control"] = r["id"] in ctrl
    return recs


def correct_flags(recs):
    return [(not r["abstained"]) and r["pred"] == r["label"] for r in recs]


def pair_flags(recs):
    # Control pairs are excluded, exactly as in metrics.score: without this the
    # "Δ CC vs base" column and the CI table measured a different quantity
    # from the Causal Consistency column they sit next to (on mimic_v3b base,
    # 0.080 here against 0.000 there -- a constant answer gets control pairs
    # right for free).
    pairs = defaultdict(list)
    for r in recs:
        pairs[r["pair_id"]].append(r)
    return [all((not a["abstained"]) and a["pred"] == a["label"] for a in arms)
            for arms in pairs.values()
            if len(arms) == 2 and not any(a.get("is_control") for a in arms)]


def fmt(rows, key, multi_seed):
    arr = np.array([r[key] for r in rows], dtype=float)
    if multi_seed:
        return f"{arr.mean():.3f} ± {arr.std(ddof=1):.3f}"
    return f"{arr.mean():.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results/table.md")
    ap.add_argument("--tag", default="",
                    help="filename suffix matching run_eval.py --tag, so an "
                         "alternative-UQ-signal run can be tabulated without "
                         "overwriting the reported one.")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    summary = json.load(
        open(Path(args.results) / f"summary_{args.split}{args.tag}.json"))
    by_variant = defaultdict(list)
    for s in summary:
        by_variant[s["variant"]].append(s)
    seeds = sorted({s["seed"] for s in summary})
    multi_seed = len(seeds) > 1
    model = summary[0]["model"]
    present = [v for v in VARIANTS if by_variant.get(v)]

    # everything below the main table is computed on the first seed
    preds = {}
    for v in present:
        try:
            preds[v] = restore_control_flag(
                load_preds(args.results, args.split, v, seeds[0], args.tag),
                by_variant[v][0].get("data"), args.split)
        except FileNotFoundError:
            pass

    lines = []
    uq_sig = summary[0].get("uq_signal", "entropy")
    lines.append(args.title or
                 f"## Ablation matrix -- split=`{args.split}`, "
                 f"model=`{model}`, seeds={seeds}\n")
    _uq_note = {
        "entropy": "  (the proposal's Eq. (2) as written -- measured at AUROC "
                   "0.525 [0.511, 0.539] over 6,456 items, i.e. chance; see "
                   "`results/bigbench_uq.md`)\n",
        "decision_entropy": "  (Eq. (2) restricted to the decision tokens, the "
                            "respecified 4.7 term: AUROC 0.687 [0.675, 0.700] "
                            "over 6,456 items. Rank-equivalent to "
                            "`logit_margin` for K=2.)\n",
    }
    lines.append(f"UQ engine defers on **`{uq_sig}`**"
                 + _uq_note.get(uq_sig, "  (decision-level signal; see "
                                        "`results/bigbench_uq.md`)\n"))
    if model == "mock":
        lines.append("> **WARNING: mock backend. Plumbing check only, "
                     "not a scientific result.**\n")
    if not multi_seed:
        lines.append("Single seed. Decoding is greedy and therefore "
                     "deterministic: another seed reproduces this run "
                     "exactly, so no seed-to-seed spread is reported. The "
                     "uncertainty that does exist is over items, and is given "
                     "as a bootstrap CI in the next table.\n")

    base_cc = (np.mean(pair_flags(preds["base"])) if "base" in preds else None)
    base_acc = (np.mean(correct_flags(preds["base"])) if "base" in preds
                else None)

    lines.append("| Variant | Base | RAG | Sym | UQ | CL | Accuracy (strict) | "
                 "Δ Acc vs base | Causal Consistency | Δ CC vs base | "
                 "Violation Rate | Coverage | Gate fired |")
    lines.append("|---|:-:|:-:|:-:|:-:|:-:|---|---|---|---|---|---|---|")
    for v in present:
        rows = by_variant[v]
        c = COMPONENTS[v]
        acc = np.mean([r["accuracy"] for r in rows])
        cc = np.mean([r["causal_consistency"] for r in rows])
        d_acc = "—" if base_acc is None else f"{acc - base_acc:+.3f}"
        d_cc = "—" if base_cc is None else f"{cc - base_cc:+.3f}"
        gate = np.mean([r.get("gate_fired_rate", 0.0) for r in rows])
        lines.append(
            f"| {LABELS[v]} | {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} | "
            f"{fmt(rows, 'accuracy', multi_seed)} | {d_acc} | "
            f"{fmt(rows, 'causal_consistency', multi_seed)} | {d_cc} | "
            f"{fmt(rows, 'violation_rate', multi_seed)} | "
            f"{fmt(rows, 'coverage', multi_seed)} | {gate:.3f} |")

    # ---- bootstrap CIs over items -------------------------------------
    lines.append("\n### 95% bootstrap CI over items (seed "
                 f"{seeds[0]})\n")
    lines.append("| Variant | Accuracy | 95% CI | Causal Consistency | 95% CI |")
    lines.append("|---|---|---|---|---|")
    for v in present:
        if v not in preds:
            continue
        af, pf = correct_flags(preds[v]), pair_flags(preds[v])
        alo, ahi = bootstrap_ci(af, seed=0)
        clo, chi = bootstrap_ci(pf, seed=0)
        lines.append(f"| {LABELS[v]} | {np.mean(af):.3f} | "
                     f"[{alo:.3f}, {ahi:.3f}] | {np.mean(pf):.3f} | "
                     f"[{clo:.3f}, {chi:.3f}] |")

    # ---- McNemar of every variant against the BASELINE ----------------
    lines.append("\n### Paired McNemar of each contribution **against the "
                 f"baseline** (seed {seeds[0]}, item-level correctness)\n")
    lines.append("This is the supervisor's question: what does adding this "
                 "contribution to the base model do?\n")
    lines.append("| Variant vs base | B01 (base wrong→right) | "
                 "B10 (base right→wrong) | p (exact) | p (Holm) |")
    lines.append("|---|---|---|---|---|")
    raw_p, cells = {}, {}
    if "base" in preds:
        af = correct_flags(preds["base"])
        for v in present:
            if v == "base" or v not in preds:
                continue
            assert [x["id"] for x in preds["base"]] == \
                   [x["id"] for x in preds[v]], "variants scored on different items"
            m = mcnemar(af, correct_flags(preds[v]))
            raw_p[f"base→{v}"] = m["p_value"]
            cells[f"base→{v}"] = m
    adj = holm_bonferroni(raw_p) if raw_p else {}
    for key, m in cells.items():
        lines.append(f"| {key} | {m['b01']} | {m['b10']} | "
                     f"{m['p_value']:.4g} | {adj[key]:.4g} |")

    # ---- McNemar along the cumulative ladder --------------------------
    ladder = [v for v in LADDER if v in preds]
    if len(ladder) > 1:
        lines.append(f"\n### Paired McNemar along the cumulative ladder "
                     f"(seed {seeds[0]})\n")
        lines.append("| Comparison | B01 | B10 | p (exact) | p (Holm) |")
        lines.append("|---|---|---|---|---|")
        raw_p2, cells2 = {}, {}
        for prev, cur in zip(ladder, ladder[1:]):
            m = mcnemar(correct_flags(preds[prev]), correct_flags(preds[cur]))
            raw_p2[f"{prev}→{cur}"] = m["p_value"]
            cells2[f"{prev}→{cur}"] = m
        adj2 = holm_bonferroni(raw_p2)
        for key, m in cells2.items():
            lines.append(f"| {key} | {m['b01']} | {m['b10']} | "
                         f"{m['p_value']:.4g} | {adj2[key]:.4g} |")

    # ---- the non-circular part of the gate's contribution -------------
    gate_rows = [v for v in present
                 if v in preds and any(r["gate_fired"] for r in preds[v])]
    if gate_rows and "base" in preds:
        lines.append("\n### Where the gate's contribution comes from "
                     "(seed %d)\n" % seeds[0])
        lines.append("The gate's accuracy on items it fires on is partly "
                     "circular: it applies the same rule and threshold the "
                     "labels were generated from, and on the MedCalc benchmark "
                     "the same extractor that filters the data runs inside the "
                     "gate. Splitting the split by whether the gate fired "
                     "separates the circular part from the part that is not: "
                     "on gate-declined items the row IS the neural pathway, so "
                     "any difference there is real.\n")
        lines.append("| Variant | subset | n | accuracy | base accuracy on "
                     "the same subset |")
        lines.append("|---|---|---|---|---|")
        base_by_id = {r["id"]: r for r in preds["base"]}
        for v in gate_rows:
            for fired in (True, False):
                sub = [r for r in preds[v] if r["gate_fired"] is fired]
                if not sub:
                    continue
                bsub = [base_by_id[r["id"]] for r in sub]
                lines.append(
                    f"| {LABELS[v]} | gate {'fired' if fired else 'declined'} "
                    f"| {len(sub)} | {np.mean(correct_flags(sub)):.3f} | "
                    f"{np.mean(correct_flags(bsub)):.3f} |")

    # ---- explain a degenerate UQ row instead of leaving it looking broken --
    degenerate = [v for v in present
                  if by_variant[v][0].get("coverage", 1.0) == 0.0]
    if degenerate:
        r = by_variant[degenerate[0]][0]
        lines.append("\n### Why a UQ row can read 0.000 coverage\n")
        lines.append(
            f"Split conformal picks the largest uncertainty threshold whose "
            f"error rate on the calibration split is at most alpha = "
            f"{r.get('calib_target_alpha', 0.1):.2f}. On this calibration "
            f"split of {r.get('calib_n', 0)} items no threshold reaches that "
            f"target, because the model it is governing is near chance. The "
            f"method then falls back to its most conservative threshold, which "
            f"retains {r.get('calib_coverage_at_tau', 0.0):.1%} of the "
            f"calibration items, and on the test split retains none. "
            f"**That is the method behaving correctly, not a failure to run**: "
            f"a 10% error target is unreachable for a model at this accuracy, "
            f"so the only way to honour it is to answer nothing. It is also "
            f"the exact situation Adaptive Conformal Inference exists for -- "
            f"see `results/uq_coverage_*.md`, where the threshold is allowed "
            f"to move.\n")

    lines.append("\n### Notes\n")
    lines.append("- Causal Consistency is pair-level: both counterfactual arms "
                 "must be correct. A constant-answer model scores 0.")
    lines.append("- Abstentions count as failures for accuracy and causal "
                 "consistency, and as non-violations for violation rate. "
                 "Coverage is reported so this trade-off is visible.")
    lines.append("- **Accuracy (strict)** counts an abstention as wrong and is "
                 "the row-comparable number. Selective accuracy is in "
                 "`summary_*.json`; it rises trivially as coverage falls and "
                 "must never be compared across rows without coverage.")
    lines.append("- Rows (5)-(7) add ONE contribution to the base model. Rows "
                 "(1)-(4) are the cumulative ladder of the proposal's section "
                 "4.6. Row (8) is everything at once.")
    lines.append("- **The gate's accuracy is an upper bound.** It evaluates "
                 "the same constraint the ground truth was generated from, so "
                 "on items it fires on it cannot be wrong unless extraction "
                 "is. The figure to quote next to it is its firing rate here, "
                 "and its coverage on unfiltered real notes (42.7%, "
                 "`results/medcalc_ablation.md`).")
    lines.append("- The constraint layer is scored through the identical "
                 "readout as every other row (generate, then parse the first "
                 "decision word), not by argmax over answer logits, so the "
                 "numbers are comparable down the column.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
