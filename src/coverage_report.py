"""
The two UQ guarantees the proposal names and the pipeline did not have:
**Adaptive Conformal Inference** for the abstention threshold, and
**conditional coverage** across subgroups.

  python src/coverage_report.py --split test --variant nsai_uq
  python src/coverage_report.py --split heldout --tag _medcalc --variant uq

WHY THIS IS A SEPARATE PASS
---------------------------
It runs entirely off the saved `preds_*.jsonl`, because everything it needs --
the per-item uncertainty and whether the answer was right -- is already
recorded there. No GPU, no regeneration, and it can therefore be applied to
runs that were made before this file existed.

WHAT IT ANSWERS
---------------
1. Does the frozen split-conformal threshold hold its promised error rate on a
   split that is NOT exchangeable with the calibration split? (The held-out
   family is a different drug, quantity and threshold, so it is not.)
2. Does ACI recover the promise there, and at what cost in coverage?
3. Is the coverage the system delivers evenly spread, or is it bought by
   abstaining on one subgroup? A system that answers every SAFE case and
   abstains on every UNSAFE one has good marginal numbers and is dangerous.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from metrics import adaptive_conformal, conditional_coverage


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).open()]


def correct(r):
    return r["pred"] == r["label"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--split", default="test")
    ap.add_argument("--variant", default="nsai_uq")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = (Path(args.results) /
            f"preds_{args.split}_{args.variant}_seed{args.seed}{args.tag}.jsonl")
    recs = read_jsonl(path)
    sig = recs[0].get("uq_signal", "entropy")

    # Uncertainty as run_eval recorded it. A None means the generation never
    # produced a decision token, which is maximal uncertainty, not missing data.
    unc = [r["uq_uncertainty"] if r["uq_uncertainty"] is not None
           else float("inf") for r in recs]
    ok = [correct(r) for r in recs]

    lines = [f"# UQ coverage report -- `{args.split}{args.tag}`, "
             f"variant `{args.variant}`\n",
             f"Signal: **`{sig}`**. Target error rate on retained items: "
             f"alpha = {args.alpha}. n = {len(recs)}.\n"]

    # ---- 1. what the frozen threshold actually delivered ------------------
    answered = [r for r in recs if not r["abstained"]]
    frozen_cov = len(answered) / len(recs)
    frozen_err = (1 - np.mean([correct(r) for r in answered])
                  if answered else float("nan"))
    lines.append("## 1. The frozen split-conformal threshold, as shipped\n")
    lines.append("| | coverage | error on retained items | target |")
    lines.append("|---|---|---|---|")
    lines.append(f"| frozen tau (calibration split) | {frozen_cov:.3f} | "
                 f"{frozen_err:.3f} | {args.alpha:.2f} |")
    verdict = ("holds" if frozen_err <= args.alpha + 1e-9 else
               "**violated**")
    lines.append(f"\nTarget {verdict} on this split.\n")
    if frozen_err > args.alpha:
        lines.append("A frozen threshold is only valid while the test stream "
                     "is exchangeable with the calibration split. This split "
                     "is a different rule family, so it is not, and the "
                     "guarantee does not transfer. That is not a bug in the "
                     "calibration -- it is the documented limit of split "
                     "conformal, and the reason the proposal names the "
                     "adaptive variant.\n")

    # ---- 2. adaptive conformal inference ---------------------------------
    aci = adaptive_conformal(unc, ok, alpha=args.alpha, gamma=args.gamma)
    lines.append("## 2. Adaptive Conformal Inference on the same items\n")
    lines.append(f"alpha_t updated after every item with gamma = "
                 f"{args.gamma}; the threshold is the running "
                 f"(1 - alpha_t) quantile of uncertainties seen so far.\n")
    lines.append("| | coverage | error on retained items |")
    lines.append("|---|---|---|")
    lines.append(f"| frozen tau | {frozen_cov:.3f} | {frozen_err:.3f} |")
    lines.append(f"| **ACI** | **{aci['coverage']:.3f}** | "
                 f"**{aci['selective_error']:.3f}** |")
    lines.append(f"\nalpha_t ends at {aci['alpha_t'][-1]:.4f} "
                 f"(started at {args.alpha}). ")
    if aci["selective_error"] <= args.alpha + 1e-9:
        lines.append("ACI holds the target on this split.\n")
    else:
        lines.append("ACI does not reach the target within this many items: "
                     "the guarantee is a long-run average and this split is "
                     f"only {len(recs)} items long. The direction of travel is "
                     "the thing to read, not the endpoint.\n")

    # ---- 3. conditional coverage ------------------------------------------
    lines.append("## 3. Conditional coverage by subgroup\n")
    lines.append("Marginal coverage can hide a subgroup the system has quietly "
                 "stopped answering. `violation rate` is the share of truly "
                 "UNSAFE cases in that subgroup that were answered SAFE.\n")
    cond = conditional_coverage(recs)
    for key, rows in cond.items():
        lines.append(f"\n**by `{key}`**\n")
        lines.append("| value | n | coverage | selective accuracy | "
                     "violation rate |")
        lines.append("|---|---|---|---|---|")
        for name, m in rows.items():
            sa = ("—" if np.isnan(m["selective_accuracy"])
                  else f"{m['selective_accuracy']:.3f}")
            flag = "  ⚠️ n<5" if m["n"] < 5 else ""
            lines.append(f"| {name} | {m['n']}{flag} | {m['coverage']:.3f} | "
                         f"{sa} | {m['violation_rate']:.3f} |")

    covs = [m["coverage"] for rows in cond.values() for m in rows.values()
            if m["n"] >= 5]
    if covs:
        lines.append(f"\nWidest coverage gap between subgroups of n>=5: "
                     f"**{max(covs) - min(covs):.3f}** "
                     f"(min {min(covs):.3f}, max {max(covs):.3f}). Marginal "
                     f"coverage is {frozen_cov:.3f}.\n")

    out = Path(args.out or (f"{args.results}/uq_coverage_{args.split}"
                            f"{args.tag}_{args.variant}.md"))
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
