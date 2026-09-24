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
SUPP = [("sym", "(5) Base + gate only"), ("uq", "(6) Base + UQ only"),
        ("cl", "(7) Base + constraint layer (internal)"),
        ("nsai_uq_cl", "(8) NS-AI + UQ + constraint layer")]
CL_TRAIN = HERE / "data/mimic_v3b_cl/counterfactual_train.jsonl"
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


_STATS = {}


def cached_stats(model, split, v, items):
    """row_stats once per (model, split, variant): the ladder and the
    consolidated table read the same rows, and each costs a bootstrap."""
    if (model, split, v) not in _STATS:
        preds = load(model, split, v)
        _STATS[model, split, v] = None if preds is None else row_stats(preds, items)
    return _STATS[model, split, v]


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
            m = cached_stats(model, split, v, items)
            if m is None:
                lines.append(f"| {model} | {label} | *not run yet* | | | | | | |")
                continue
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
                         ("nsai_uq", "(4) NS-AI + UQ"),
                         ("nsai_uq_cl", "(8) NS-AI + UQ + CL")):
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


def unseen_by_adapter(items):
    """Pair ids none of whose prompts the adapter trained on. The split is
    patient-disjoint, not prompt-disjoint, so cl on ALL test pairs is partly
    scored on prompts it was fitted to; this is the view that is not."""
    if not CL_TRAIN.exists():
        return None
    seen_prompts = {json.loads(l)["prompt"] for l in CL_TRAIN.open()}
    seen = collections.defaultdict(bool)
    for r in items.values():
        seen[r["pair_id"]] |= r["prompt"] in seen_prompts
    return {pid for pid, s in seen.items() if not s}


def shuffled_control_lines(lines):
    """The adapter against the same adapter trained on shuffled labels, on the
    training script's own readout (argmax of the two answer logits)."""
    lines.append("| Model | Adapter | test acc | test CC | held-out acc "
                 "| held-out CC |")
    lines.append("|---|---|---|---|---|---|")
    for model in MODELS:
        for name, f in (("rule labels", "constraint_mimic3b.json"),
                        ("shuffled labels (control)",
                         "constraint_mimic3b_shuffled.json")):
            p = RES / model / f
            if not p.is_file():
                lines.append(f"| {model} | {name} | *not run yet* | | | |")
                continue
            d = json.load(p.open())
            for tag, r in (("frozen base", d["base"]), (name, d["adapted"])):
                if tag == "frozen base" and "shuffled" in name:
                    continue
                lines.append(f"| {model} | {tag} | {fmt(r.get('test_acc'))} "
                             f"| {fmt(r.get('test_cc'))} "
                             f"| {fmt(r.get('heldout_acc'))} "
                             f"| {fmt(r.get('heldout_cc'))} |")


CONSOLIDATED = [("base", "Base LLM", "tier 1"), ("rag", "+ RAG", "tier 2"),
                ("nsai", "NS-AI (+ gate)", "tier 3"),
                ("nsai_uq", "NS-AI + UQ", "tier 4"),
                ("sym", "Base + gate (`sym`)", "ablation"),
                ("uq", "Base + UQ (`uq`)", "ablation"),
                ("cl", "Base + constraint layer (`cl`, internal)", "ablation"),
                ("nsai_uq_cl", "NS-AI + UQ + CL", "ablation")]


def consolidated():
    """One table for the manuscript: the four tiers and the single-component
    ablations, five columns, every model and split."""
    lines = ["# MIMIC-IV v3.1 (`mimic_v3b`): consolidated verification table\n",
             "Generated by `src/make_ladder_mimic3b.py`. Single seed, greedy "
             "decoding. **Clean accuracy** is strict accuracy over all items "
             "(a deferral counts as wrong). **Model's own** is the decision "
             "parsed from the model's generation before the gate overrides it "
             "or UQ defers it. **CC** excludes control pairs. **Violation** is "
             "the share of UNSAFE items answered SAFE. † marks a row where the "
             "gate fired: there, clean accuracy and CC are partly an identity "
             "with the labelling rule, not a measurement. `cl` is the "
             "Aim 3 residual adapter h' = h + α·P_causal(h) (rank 32, layer 30, "
             "α = 1), trained on 1,000 train pairs; *unseen* restricts it to "
             "test pairs none of whose prompts it trained on.\n"]
    for split, name in SPLITS:
        items, _, _ = load_items(DATA, split)
        unseen = unseen_by_adapter(items) if split == "test" else None
        lines.append(f"\n## {name}\n")
        lines.append("| Model | Row | Kind | Clean accuracy | Model's own "
                     "| CC | Coverage | Violation |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for model in MODELS:
            rows = [(v, label, kind, None) for v, label, kind in CONSOLIDATED]
            if unseen:
                rows += [(v, label + ", unseen pairs", "ablation", unseen)
                         for v, label, _ in CONSOLIDATED
                         if v in ("cl", "nsai_uq_cl")]
            for v, label, kind, only in rows:
                if only is None:
                    m = cached_stats(model, split, v, items)
                else:
                    preds = load(model, split, v)
                    m = None if preds is None else row_stats(
                        [p for p in preds if p["pair_id"] in only], items)
                if m is None:
                    lines.append(f"| {model} | {label} | {kind} | *not run "
                                 f"yet* | | | | |")
                    continue
                dag = " †" if m["gate"] > 0 else ""
                lines.append(
                    f"| {model} | {label} | {kind} | {fmt(m['acc'])}{dag} "
                    f"{fmt_ci(m['acc_ci'])} | {fmt(m['own'])} "
                    f"| {fmt(m['cc'])}{dag} {fmt_ci(m['cc_ci'])} "
                    f"| {fmt(m['cov'])} | {fmt(m['viol'])} |")
    out = RES / "CONSOLIDATED_MIMIC3B.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


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
    lines.append("\n## Constraint layer against its shuffled-label control\n")
    lines.append("From `constraint_layer.py` on its own evaluation (argmax over "
                 "the two answer logits, the 1,000-pair training sample's "
                 "splits), not the generation-parsed rows above. If the "
                 "shuffled adapter gains as much, the gain is from touching "
                 "the model, not from the constraint.\n")
    shuffled_control_lines(lines)
    out = RES / "LADDER_MIMIC3B.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    consolidated()


if __name__ == "__main__":
    main()
