"""
The section 4.6 ablation ladder for the corrected MIMIC-IV v3.1 arm, one table
across the three models.

  python src/make_ladder_mimic3b.py            # -> results/mimic_v3b/LADDER_MIMIC3B.md

WHAT IT REPORTS, AND WHY THIS WAY
----------------------------------
* PRIMARY: the proposal's four tiers -- Base, + RAG, NS-AI (+ gate), NS-AI+UQ
  -- per model and split. `sym` (base + gate) and `uq` (base + UQ) answer a
  different question and sit in a supplementary table.
* CIs resample DISTINCT PROMPTS (accuracy) and pair prompt-tuples (CC), via
  report_mimic3.metrics: 166,264 test items are ~13,900 distinct prompts, and
  an item-level bootstrap would count ~12 copies of one answer as 12 draws.
* CIRCULARITY. The gate applies the rule the labels were generated from, so
  where it fires, final accuracy is an identity check. Every row therefore
  also carries the MODEL'S OWN ANSWER (parsed from its generation, before the
  gate overrides it and before UQ defers it) and how often the model already
  agreed with the gate. On the held-out family the gate fires on every item:
  there, the model's own answer is the only accuracy that measures anything.
* UQ. Where conformal calibration cannot reach alpha it abstains on
  everything; a flat 0.000 hides what the signal can do. The risk-coverage
  summary gives, per model, the error at full coverage, the area under the
  curve, and the largest coverage at which alpha is reachable on these items.
"""

import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
CSAI = Path("/home/asosoft/abtin/paper/csai")
sys.path.insert(0, str(HERE / "src"))
from report_mimic3 import cluster_ci, fmt, fmt_ci, load_items, metrics  # noqa: E402
sys.path.insert(0, str(CSAI / "src"))
from make_table import model_correct_flags, risk_coverage, curve_steps  # noqa: E402
from model import parse_answer  # noqa: E402

DATA = HERE / "data/mimic_v3b"
RES = HERE / "results/mimic_v3b"
TAG = "_mimic3b"
MODELS = ["biomistral-7b", "llama3-openbiollm-8b", "mistral-7b-instruct-v0-2"]
LADDER = [("base", "(1) Base LLM"), ("rag", "(2) + RAG"),
          ("nsai", "(3) NS-AI (+ gate)"), ("nsai_uq", "(4) NS-AI + UQ")]
SUPP = [("sym", "(5) Base + gate only"), ("uq", "(6) Base + UQ only")]
SPLITS = [("test", "test (metformin ×2, spironolactone)"),
          ("heldout", "held-out family (warfarin INR > 4)")]


def load(model, split, variant):
    p = RES / model / f"preds_{split}_{variant}_seed0{TAG}.jsonl"
    return [json.loads(l) for l in p.open()] if p.is_file() else None


def summary_row(model, split, variant):
    p = RES / model / f"summary_{split}{TAG}.json"
    if not p.is_file():
        return {}
    return next((r for r in json.load(p.open()) if r["variant"] == variant), {})


def row_stats(preds, items):
    m = metrics(preds, items)
    own = model_correct_flags(preds)
    m["own"] = float(np.mean(own))
    m["own_ci"] = cluster_ci([items[p["id"]]["prompt"] for p in preds],
                             [float(x) for x in own])
    unsafe = [p for p in preds if p["label"] == "UNSAFE"]
    m["viol"] = (float(np.mean([(not p["abstained"]) and p["pred"] == "SAFE"
                                for p in unsafe])) if unsafe else None)
    fired = [p for p in preds if p.get("gate_fired")]
    m["gate"] = len(fired) / len(preds)
    m["agree"] = (float(np.mean([parse_answer(p.get("raw") or "") == p["pred"]
                                 for p in fired])) if fired else None)
    return m


def table(lines, rows, split, items):
    lines.append("| Model | Tier | Accuracy (strict) [95% CI] | Model's own "
                 "answer [95% CI] | Causal Consistency [95% CI] | Violation "
                 "| Coverage | Gate fired | Model agrees with gate |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        for v, label in rows:
            preds = load(model, split, v)
            if preds is None:
                lines.append(f"| {model} | {label} | *not run yet* | | | | | | |")
                continue
            m = row_stats(preds, items)
            dag = " †" if m["gate"] > 0 else ""
            lines.append(
                f"| {model} | {label} | {fmt(m['acc'])}{dag} {fmt_ci(m['acc_ci'])} "
                f"| {fmt(m['own'])} {fmt_ci(m['own_ci'])} "
                f"| {fmt(m['cc'])}{dag} {fmt_ci(m['cc_ci'])} "
                f"| {fmt(m['viol'])} | {fmt(m['cov'])} | {m['gate']:.3f} "
                f"| {fmt(m['agree'])} |")


def uq_summary(lines, split):
    lines.append("| Model | Row | Error at full coverage | AURC | Lowest error "
                 "at ≥1% coverage | α reachable up to coverage | τ "
                 "(calibration) | Coverage at τ, calibration | Coverage at "
                 "τ, here |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        for v, label in (("uq", "(6) Base + UQ only"),
                         ("nsai_uq", "(4) NS-AI + UQ")):
            preds = load(model, split, v)
            if preds is None:
                continue
            curve = risk_coverage(preds)
            s = summary_row(model, split, v)
            tau_s = "—" if s.get("tau") is None else f"{s['tau']:.4g}"
            if not curve:
                lines.append(f"| {model} | {label} | — (gate decides every "
                             f"item; nothing for UQ to govern) | | | | "
                             f"{tau_s} | | {fmt(s.get('coverage'))} |")
                continue
            alpha = s.get("calib_target_alpha", 0.1)
            aurc = float(np.mean([p["error"] for p in curve_steps(curve)]))
            floor = min((p for p in curve if p["coverage"] >= 0.01),
                        key=lambda p: p["error"], default=None)
            reach = [p["coverage"] for p in curve if p["error"] <= alpha]
            tau = s.get("tau")
            lines.append(
                f"| {model} | {label} | {curve[-1]['error']:.3f} | {aurc:.3f} "
                f"| {fmt(floor['error'] if floor else None)} "
                f"| {f'{max(reach):.3f}' if reach else 'never'} "
                f"| {'—' if tau is None else f'{tau:.4g}'} "
                f"| {fmt(s.get('calib_coverage_at_tau'))} "
                f"| {fmt(s.get('coverage'))} |")


def main():
    lines = ["# MIMIC-IV v3.1 (corrected arm, `mimic_v3b`): the section 4.6 "
             "ablation ladder across three models\n",
             "Generated by `src/make_ladder_mimic3b.py`. Single seed, greedy "
             "decoding. CIs resample distinct prompts (accuracy) and pair "
             "prompt-tuples (CC), not items.\n",
             "**† Identity, not measurement.** On items where the gate fires, "
             "it applies the same rule and threshold the label was generated "
             "from, so the final answer is correct by construction. For what "
             "the model itself knows, read **Model's own answer**: the "
             "decision parsed from its generation, before the gate overrides "
             "it and before UQ defers it. **Model agrees with gate** is the "
             "share of gate-fired items on which the model had already "
             "reached the guideline's answer.\n"]
    for split, name in SPLITS:
        items, _, _ = load_items(DATA, split)
        safe = np.mean([r["label"] == "SAFE" for r in items.values()])
        lines.append(f"\n## {name}\n")
        lines.append(f"{len(items):,} items. The label is SAFE on "
                     f"{safe:.1%} of them, so an always-SAFE answer scores "
                     f"accuracy {safe:.3f} and an always-UNSAFE one "
                     f"{1 - safe:.3f}; both score CC 0.\n")
        lines.append("### Primary: the four tiers\n")
        table(lines, LADDER, split, items)
        lines.append("\n### Supplementary: one contribution added to the base "
                     "model\n")
        table(lines, SUPP, split, items)
        lines.append("\n### UQ: risk-coverage instead of a flat collapse\n")
        lines.append("Over the items UQ governs (gate-decided items are never "
                     "deferred). Error is the model's own answer's, on the "
                     "items kept at each threshold; one operating point per "
                     "distinct uncertainty value. α = 0.10. Per-threshold "
                     "curves: `<model>/table_mimic3b_<split>_riskcov_<row>.csv`.\n")
        uq_summary(lines, split)
    out = RES / "LADDER_MIMIC3B.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
