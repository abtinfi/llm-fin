"""
Publication figures for the corrected MIMIC-IV v3.1 arm (`mimic_v3b`) and the
Aim 1/2 feature-attribution audit, as vector PDF and 300-dpi PNG.

  /usr/bin/python3 src/generate_paper_plots.py   # -> results/mimic_v3b/figures/

(The pipeline's conda env has no matplotlib; the system python has it plus
numpy and scipy, which is all this reads with. Nothing here touches a GPU.)

WHAT EACH FIGURE READS, AND WHY
-------------------------------
fig_risk_coverage   `<model>/table_mimic3b_<split>_riskcov_<row>.csv`, the
                    curves make_table.py writes: one operating point per
                    distinct uncertainty value, over the items UQ governs (the
                    gate-decided ones are never deferred), error = the MODEL'S
                    OWN answer's. The deployed point is where the calibrated
                    threshold tau (summary row) falls on that curve.
fig_sae_contraction the top-25 L20 TopK features, knock-out effect against
                    the old control (one feature drawn uniformly over the
                    dictionary, dead ones included) and the S1-fixed control
                    (mean of 5 live features matched on firing rate). Values
                    are read from the two json files and nothing is typed in:
                    the headline numbers in the annotations are computed here.
fig_option_order    base on the test split with the answer options in the
                    original order and swapped; accuracy and CC with CIs that
                    resample distinct prompts (report_mimic3.metrics), and the
                    share of UNSAFE answers, which is the position bias itself.

Colour: models take categorical slots 1-3 in fixed order (validated all-pairs,
light surface; slot 3 sits below 3:1 contrast, so every mark is also labelled
or in the legend). Rows of the same model differ by line style, not colour.
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
CSAI = Path("/home/asosoft/abtin/paper/csai")
sys.path.insert(0, str(HERE / "src"))
from report_mimic3 import load_items, metrics  # noqa: E402

RES = HERE / "results/mimic_v3b"
OUT = RES / "figures"
TAG = "_mimic3b"
MODELS = [("biomistral-7b", "BioMistral-7B"),
          ("llama3-openbiollm-8b", "OpenBioLLM-8B"),
          ("mistral-7b-instruct-v0-2", "Mistral-7B-Instruct")]
SPLITS = [("test", "Test (metformin ×2, spironolactone)"),
          ("heldout", "Held-out family (warfarin, INR > 4)")]

# Reference palette, light mode: categorical slots 1-3, text and chrome inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
NEUTRAL = "#b9b7b0"
SURFACE = "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf / .png")


def summary_row(model, split, variant, tag=TAG):
    p = RES / model / f"summary_{split}{tag}.json"
    if not p.is_file():
        return None
    return next((r for r in json.load(p.open()) if r["variant"] == variant),
                None)


def read_curve(model, split, row):
    p = RES / model / f"table{TAG}_{split}_riskcov_{row}.csv"
    if not p.is_file():
        return None
    with p.open() as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


# --------------------------------------------------------------------------
# 1. Risk-coverage (Aim 4)
# --------------------------------------------------------------------------
UQ_ROWS = [("uq", "Base + UQ", "-"), ("nsai_uq", "NS-AI + UQ", "--"),
           ("nsai_uq_cl", "NS-AI + UQ + CL", ":")]


def operating_point(curve, tau):
    """Where the calibrated threshold lands on the curve: keep u <= tau."""
    if tau is None:
        return None
    if tau == -math.inf:
        return 0.0, "none"
    if tau < curve[0]["threshold"]:
        return 0.0, "below"
    kept = [p for p in curve if p["threshold"] <= tau]
    return kept[-1]["coverage"], kept[-1]["error"]


def fig_risk_coverage():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    used_rows = set()
    for ax, (split, title) in zip(axes, SPLITS):
        alpha = 0.10
        ops = []
        for (model, name), color in zip(MODELS, SERIES):
            for row, rlab, ls in UQ_ROWS:
                curve = read_curve(model, split, row)
                if not curve:
                    continue
                s = summary_row(model, split, row) or {}
                alpha = s.get("calib_target_alpha", alpha)
                cov = [0.0] + [p["coverage"] for p in curve]
                err = [curve[0]["error"]] + [p["error"] for p in curve]
                ax.plot(cov, err, color=color, lw=1.6, ls=ls,
                        drawstyle="steps-pre", solid_capstyle="round")
                used_rows.add(row)
                op = operating_point(curve, s.get("tau"))
                if op is None:
                    continue
                c, e = op
                ax.plot([c], [e if isinstance(e, float) else 0.0],
                        marker="o", ms=5.5, color=color, mec=SURFACE,
                        mew=1.2, zorder=5)
                why = {"none": " (τ = −∞)",
                       "below": " (τ below all u)"}
                ops.append(f"{name}, {rlab}: defers {1 - c:.1%}"
                           + (why[e] if isinstance(e, str)
                              else f", risk {e:.3f}"))
        ax.axhline(alpha, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.text(1.0, alpha + 0.01, f"target risk α = {alpha:.2f}",
                ha="right", va="bottom", fontsize=7, color=INK2)
        if split == "heldout":
            ax.text(0.98, 0.97, "no NS-AI rows: the gate decides every\n"
                    "held-out item, leaving nothing to defer",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.8, color=MUTED)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.0)
        ax.text(0.0, -0.27, "Deployed τ (markers):\n" + "\n".join(ops),
                transform=ax.transAxes, ha="left", va="top", fontsize=6.6,
                color=INK2, linespacing=1.35)
    axes[0].set_ylabel("Selective risk\n(error of the model's own answer)")
    fig.supxlabel("Coverage (share of UQ-governed items answered)",
                  fontsize=8.5, color=INK2, y=-0.04)
    handles = [Line2D([], [], color=c, lw=2, label=n)
               for (_, n), c in zip(MODELS, SERIES)]
    handles += [Line2D([], [], color=INK2, lw=1.4, ls=ls, label=lab)
                for row, lab, ls in UQ_ROWS if row in used_rows]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               fontsize=7.2, bbox_to_anchor=(0.5, 1.07))
    save(fig, "fig_risk_coverage")


# --------------------------------------------------------------------------
# 2. SAE feature causal effect, before and after the matched control (Aims 1-2)
# --------------------------------------------------------------------------
SAE_PAIRS = [
    ("test", CSAI / "results/pre_s1s3_20260908/sae_topk_L20_fis.json",
     CSAI / "results/sae/sae_topk_L20_fis.json"),
    ("heldout", CSAI / "results/pre_s1s3_20260908/sae_heldout_topk_L20_fis.json",
     CSAI / "results/sae_heldout/sae_topk_L20_fis.json"),
]
HIGHLIGHT = 10721


def sae_rows(old_path, new_path):
    """Per feature: raw knock-out, excess over the uniform control, excess over
    the matched mean (+ its sd). None if the matched run is not on disk."""
    if not (old_path.is_file() and new_path.is_file()):
        return None
    old, new = json.load(old_path.open()), json.load(new_path.open())
    if new["causal"].get("control_mode") != "matched":
        return None
    top = old["features"][:25]
    rows = []
    for f in top:
        k = str(f["feature"])
        o, n = old["causal"]["per_feature"][k], new["causal"]["per_feature"].get(k)
        if n is None:
            continue
        rows.append({"feature": f["feature"], "concept": f["concept"],
                     "raw": n["mean_abs_delta_logit"],
                     "old_excess": o["excess"], "new_excess": n["excess"],
                     "new_sd": n.get("control_sd", 0.0),
                     "old_s": max(o["excess"], 0.0),
                     "new_s": max(n["excess"], 0.0)})
    return {"rows": rows, "n_items": new["causal"]["n_items"],
            "n_controls": new["causal"].get("n_controls")}


def contraction(rows, concept=None):
    rs = [r for r in rows if concept is None or r["concept"] == concept]
    o, n = sum(r["old_s"] for r in rs), sum(r["new_s"] for r in rs)
    return (1 - n / o) if o > 0 else None, o, n


def fig_sae_contraction():
    panels = [(split, sae_rows(o, n)) for split, o, n in SAE_PAIRS]
    panels = [(s, d) for s, d in panels if d and d["rows"]]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.2, 2.9 * len(panels)),
                             squeeze=False)
    for ax, (split, d) in zip(axes[:, 0], panels):
        rows = sorted(d["rows"], key=lambda r: (r["concept"], -r["raw"]))
        x = np.arange(len(rows))
        w = 0.27
        ax.bar(x - w, [r["raw"] for r in rows], w * 0.92, color=NEUTRAL,
               label="raw knock-out |Δ margin|")
        ax.bar(x, [r["old_excess"] for r in rows], w * 0.92, color=SERIES[1],
               label="excess over 1 uniform control (old)")
        ax.bar(x + w, [r["new_excess"] for r in rows], w * 0.92,
               color=SERIES[0], yerr=[r["new_sd"] for r in rows],
               error_kw={"elinewidth": 0.7, "ecolor": INK2, "capsize": 0},
               label=f"excess over {d['n_controls']} firing-rate-matched "
                     f"controls (S1 fix)")
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"#{r['feature']}\n{r['concept']}" for r in rows],
                           fontsize=6, rotation=90)
        for t, r in zip(ax.get_xticklabels(), rows):
            if r["feature"] == HIGHLIGHT:
                t.set_color(INK)
                t.set_fontweight("bold")
        hi = next((i for i, r in enumerate(rows)
                   if r["feature"] == HIGHLIGHT), None)
        if hi is not None:
            r = rows[hi]
            ax.annotate(f"#{HIGHLIGHT}: S_causal {r['old_s']:.4f} → "
                        f"{r['new_s']:.4f}",
                        (hi + w, max(r["new_excess"], 0)),
                        xytext=(-120, 14), textcoords="offset points",
                        fontsize=7, color=INK,
                        arrowprops={"arrowstyle": "-", "color": INK2,
                                    "lw": 0.6})
        tot, o, n = contraction(rows)
        parts = []
        for c in ("age", "heart_rate"):
            cc, _, _ = contraction(rows, c)
            if cc is not None:
                parts.append(f"{c} {cc:.0%}")
        label = {"test": "Test split", "heldout": "Held-out split"}[split]
        ax.set_title(
            f"{label} ({d['n_items']} items): Σ S_causal over the top 25 "
            f"{o:.3f} → {n:.3f}"
            + (f", −{tot:.1%}" if tot is not None else "")
            + (f"  ({', '.join(parts)})" if parts else ""),
            loc="left", color=INK)
        ax.set_ylabel("Decision-margin change (logits)")
        ax.set_xlim(-0.6, len(rows) - 0.4)
        top = max(max(r["raw"], r["old_excess"], r["new_excess"] + r["new_sd"])
                  for r in rows)
        ax.set_ylim(top=top * 1.3)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, 1.06))
    if not any(s == "heldout" for s, _ in panels):
        fig.text(0.01, -0.02, "Held-out: the matched-control knock-out has "
                 "not been run, so no held-out panel is shown.",
                 fontsize=7, color=MUTED)
    fig.tight_layout()
    save(fig, "fig_sae_contraction")


# --------------------------------------------------------------------------
# 3. Option-order invariance
# --------------------------------------------------------------------------
def load_preds(model, split, variant, tag):
    p = RES / model / f"preds_{split}_{variant}_seed0{tag}.jsonl"
    return [json.loads(l) for l in p.open()] if p.is_file() else None


def fig_option_order():
    items = {"orig": load_items(HERE / "data/mimic_v3b", "test")[0],
             "swap": load_items(HERE / "data/mimic_v3b_swap", "test")[0]}
    stats = {}
    for model, _ in MODELS:
        for key, tag in (("orig", TAG), ("swap", "_mimic3bswap")):
            preds = load_preds(model, "test", "base", tag)
            if preds is None:
                continue
            m = metrics(preds, items[key])
            answered = [p for p in preds if not p["abstained"]]
            m["unsafe"] = float(np.mean([p["pred"] == "UNSAFE"
                                         for p in answered]))
            stats[model, key] = m
    safe_rate = float(np.mean([r["label"] == "SAFE"
                               for r in items["orig"].values()]))
    panels = [("acc", "Accuracy", True), ("cc", "Causal Consistency", True),
              ("unsafe", "Share of answers UNSAFE", False)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    x = np.arange(len(MODELS))
    w = 0.36
    for ax, (k, title, has_ci) in zip(axes, panels):
        for j, (key, lab, col) in enumerate(
                (("orig", "SAFE listed first (original)", SERIES[0]),
                 ("swap", "UNSAFE listed first (swapped)", SERIES[1]))):
            vals, lo, hi = [], [], []
            for model, _ in MODELS:
                m = stats.get((model, key))
                v = m[k] if m and m[k] is not None else np.nan
                vals.append(v)
                ci = m.get(f"{k}_ci") if (m and has_ci) else None
                lo.append(v - ci[0] if ci else 0)
                hi.append(ci[1] - v if ci else 0)
            xs = x + (j - 0.5) * w
            ax.bar(xs, vals, w * 0.9, color=col, label=lab,
                   yerr=[lo, hi] if has_ci else None,
                   error_kw={"elinewidth": 0.7, "ecolor": INK2, "capsize": 0})
            for xi, v, h in zip(xs, vals, hi):
                if not np.isnan(v):
                    ax.text(xi, v + h + 0.015, f"{v:.2f}", ha="center",
                            va="bottom", fontsize=6.3, color=INK2)
        if k == "acc":
            ax.axhline(safe_rate, color=INK2, lw=0.8, ls=(0, (4, 3)),
                       label=f"always-SAFE accuracy ({safe_rate:.3f})")
        if k == "unsafe":
            ax.axhline(1 - safe_rate, color=MUTED, lw=0.8, ls=(0, (1, 2)),
                       label=f"true UNSAFE rate ({1 - safe_rate:.3f})")
        ax.set_title(title, loc="left", color=INK)
        ax.set_ylim(0, 1.08)
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in MODELS], fontsize=6.8, rotation=15)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    h2, l2 = axes[2].get_legend_handles_labels()
    handles += [h for h, l in zip(h2, l2) if l.startswith("true")]
    labels += [l for l in l2 if l.startswith("true")]
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7.2,
               bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    save(fig, "fig_option_order")
    return stats


def main():
    fig_risk_coverage()
    fig_sae_contraction()
    stats = fig_option_order()
    for (model, key), m in sorted(stats.items()):
        print(f"  option order  {model:26s} {key}: acc {m['acc']:.3f} "
              f"cc {m['cc']:.3f} unsafe {m['unsafe']:.3f}")


if __name__ == "__main__":
    main()
