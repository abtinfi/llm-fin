"""
Does the uncertainty signal actually carry information?

Run this BEFORE trusting any row of the ablation that involves the UQ engine.
It answers one question: among the cases the symbolic gate does NOT decide, do
wrong answers look more uncertain than right ones? If they do not, the UQ row is
measuring nothing and must not be presented as a contribution.

Signals (--signal), all converted to a "higher = more uncertain" convention so
AUROC is directly comparable across them:

  entropy         mean token-level predictive entropy over the generated tokens.
                  The proposal's Eq. (2) signal. Uncertainty = entropy.
  max_entropy     max token entropy over the generation. Uncertainty = value.
  logit_margin    |logit(SAFE) - logit(UNSAFE)| at the decision token.
                  Uncertainty = -|margin|: a margin near zero means the model is
                  torn between the two answers. Tied to the binary decision
                  rather than the whole vocabulary.
  answer_logprob  log-probability of the emitted answer token.
                  Uncertainty = -logprob.

Outputs
  - AUROC of the signal as a wrongness detector (0.5 = no signal)
  - a risk-coverage curve: error rate among retained cases as coverage falls
  - the coverage/error pair at each retained fraction, so a single operating
    point can be chosen honestly instead of reverse-engineered from the result

  python src/uq_diagnostic.py --preds results/preds_test_nsai_seed0.jsonl
  python src/uq_diagnostic.py --preds results/preds_test_nsai_seed0.jsonl \
      --signal logit_margin
  python src/uq_diagnostic.py --preds results/preds_test_nsai_seed0.jsonl --all
"""

import argparse
import json
from pathlib import Path

import numpy as np

# name -> (record field, transform to "higher = more uncertain")
SIGNALS = {
    "entropy":        ("entropy",        lambda v: v),
    "max_entropy":    ("max_entropy",    lambda v: v),
    "logit_margin":   ("logit_margin",   lambda v: -abs(v)),
    "answer_logprob": ("answer_logprob", lambda v: -v),
}


def auroc(scores, labels):
    """Probability a random positive outranks a random negative. Rank-based."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for i, c in enumerate(counts):
        if c > 1:
            m = inv == i
            ranks[m] = ranks[m].mean()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auroc_ci(unc, wrong, n_boot=5000, seed=0):
    """
    Percentile bootstrap CI for AUROC. With n in the tens, a point estimate of
    0.56 is not distinguishable from chance; the CI is what decides whether the
    P1 acceptance criterion is actually met.
    """
    rng = np.random.default_rng(seed)
    n = len(unc)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        w = wrong[idx]
        if w.all() or not w.any():
            continue
        stats.append(auroc(unc[idx], w))
    if not stats:
        return float("nan"), float("nan")
    return (float(np.percentile(stats, 2.5)),
            float(np.percentile(stats, 97.5)))


def paired_delta(pool, sig_a, sig_b, n_boot=5000, seed=0):
    """
    Paired bootstrap of AUROC(sig_a) - AUROC(sig_b) on the SAME items.

    Comparing two independent CIs is not a test of the difference: they can
    overlap while the paired difference is reliably non-zero, and vice versa.
    P1 asks whether a decision-level signal beats whole-vocabulary entropy, so
    the difference is the quantity of interest and it must be resampled paired.

    Restricted to records where both signals are present, so the two AUROCs are
    computed over an identical item set.
    """
    fa, ta = SIGNALS[sig_a]
    fb, tb = SIGNALS[sig_b]
    usable = [r for r in pool
              if r.get(fa) is not None and r.get(fb) is not None]
    if len(usable) < 10:
        return None

    ua = np.array([ta(r[fa]) for r in usable], float)
    ub = np.array([tb(r[fb]) for r in usable], float)
    wrong = np.array([r["pred"] != r["label"] for r in usable], bool)
    if wrong.all() or not wrong.any():
        return None

    obs = auroc(ua, wrong) - auroc(ub, wrong)
    rng = np.random.default_rng(seed)
    n = len(usable)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        w = wrong[idx]
        if w.all() or not w.any():
            continue
        deltas.append(auroc(ua[idx], w) - auroc(ub[idx], w))
    deltas = np.asarray(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # two-sided bootstrap p for delta == 0
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {"n": n, "delta": obs, "lo": float(lo), "hi": float(hi),
            "p": float(min(1.0, p)),
            "auroc_a": auroc(ua, wrong), "auroc_b": auroc(ub, wrong)}


def analyse(pool, signal, verbose=True):
    """
    Returns (auroc, n_used, ci_lo, ci_hi) for one signal over `pool`, or nans if
    the signal is missing/None on too many records to be worth scoring.
    """
    field, to_uncertainty = SIGNALS[signal]
    usable = [r for r in pool if r.get(field) is not None]
    n_missing = len(pool) - len(usable)

    if verbose and n_missing:
        print(f"  {n_missing} of {len(pool)} records have no {field} "
              f"(no answer token found in the generation); excluded")
    if len(usable) < 10:
        if verbose:
            print(f"  too few usable records for {signal}")
        return float("nan"), len(usable), float("nan"), float("nan")

    unc = np.array([to_uncertainty(r[field]) for r in usable], float)
    wrong = np.array([r["pred"] != r["label"] for r in usable], bool)
    if wrong.all() or not wrong.any():
        if verbose:
            print(f"  all records are {'wrong' if wrong.all() else 'correct'}; "
                  f"AUROC undefined for {signal}")
        return float("nan"), len(usable), float("nan"), float("nan")

    a = auroc(unc, wrong)
    lo, hi = auroc_ci(unc, wrong)

    if verbose:
        print(f"\n=== signal: {signal} ({field}, n={len(usable)}) ===")
        print(f"error rate at full coverage: {wrong.mean():.3f}")
        print(f"mean uncertainty  correct: {unc[~wrong].mean():.4f}   "
              f"wrong: {unc[wrong].mean():.4f}")
        print(f"AUROC ({signal} detects wrongness): {a:.3f} "
              f"[95% CI {lo:.3f}, {hi:.3f}]")
        if lo <= 0.5 <= hi:
            print("  -> CI covers 0.5: not distinguishable from chance at this "
                  "sample size, whatever the point estimate suggests.")
        if a < 0.55:
            print("  -> essentially no signal. The UQ row cannot be claimed as a "
                  "contribution; report this as a negative result.")
        elif a < 0.65:
            print("  -> weak signal. Report the risk-coverage curve, not a single "
                  "operating point.")
        else:
            print("  -> usable signal.")

        print("\nrisk-coverage curve (threshold sweep, keep lowest uncertainty)")
        print(f"{'coverage':>9} {'error':>8} {'tau':>10}")
        order = unc.argsort()
        for frac in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
            k = max(1, int(round(frac * len(usable))))
            keep = order[:k]
            print(f"{k/len(usable):9.3f} {wrong[keep].mean():8.3f} "
                  f"{unc[keep].max():10.4f}")

    return a, len(usable), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--signal", choices=sorted(SIGNALS), default="entropy")
    ap.add_argument("--all", action="store_true",
                    help="score every signal and print a comparison table")
    ap.add_argument("--paired", nargs=2, metavar=("SIG_A", "SIG_B"),
                    help="paired bootstrap of AUROC(SIG_A) - AUROC(SIG_B) on "
                         "the same items, e.g. --paired logit_margin entropy")
    ap.add_argument("--include_gated", action="store_true",
                    help="also score cases the symbolic gate decides "
                         "(their uncertainty is not what governs them)")
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.preds).open()]
    pool = recs if args.include_gated else [r for r in recs if not r.get("gate_fired")]
    print(f"{len(pool)} of {len(recs)} cases are governed by the neural pathway")
    if len(pool) < 10:
        print("too few to analyse")
        return

    if args.paired:
        a, b = args.paired
        for s in (a, b):
            if s not in SIGNALS:
                raise SystemExit(f"unknown signal {s!r}; "
                                 f"choose from {sorted(SIGNALS)}")
        res = paired_delta(pool, a, b)
        if res is None:
            print("too few usable records for a paired comparison")
            return
        print(f"\n=== paired: {a} vs {b} (n={res['n']}, same items) ===")
        print(f"  AUROC {a}: {res['auroc_a']:.3f}")
        print(f"  AUROC {b}: {res['auroc_b']:.3f}")
        print(f"  delta:  {res['delta']:+.3f}  "
              f"[95% CI {res['lo']:+.3f}, {res['hi']:+.3f}]  "
              f"bootstrap p={res['p']:.4f}")
        if res["lo"] > 0:
            print(f"  -> {a} is reliably the better wrongness detector on this "
                  f"population.")
        elif res["hi"] < 0:
            print(f"  -> {b} is reliably the better wrongness detector on this "
                  f"population.")
        else:
            print("  -> the difference is not distinguishable from zero at this "
                  "sample size.")
        return

    if not args.all:
        analyse(pool, args.signal)
        return

    results = {}
    for name in ["entropy", "max_entropy", "logit_margin", "answer_logprob"]:
        results[name] = analyse(pool, name)

    print("\n=== comparison (AUROC as wrongness detector, 0.5 = no signal) ===")
    print(f"{'signal':>16} {'n':>5} {'AUROC':>8} {'95% CI':>18}")
    for name, (a, n, lo, hi) in results.items():
        print(f"{name:>16} {n:5d} {a:8.3f}   [{lo:.3f}, {hi:.3f}]")

    best_name = max((k for k, v in results.items() if not np.isnan(v[0])),
                    key=lambda k: results[k][0], default=None)
    if best_name is None:
        print("\nno signal could be scored")
        return
    a, n, lo, hi = results[best_name]
    print(f"\nbest: {best_name} at AUROC {a:.3f} [95% CI {lo:.3f}, {hi:.3f}]")

    # P1 acceptance criterion: clearing 0.55 on a point estimate is not enough
    # when the CI still covers chance -- that is exactly the "tune until a
    # number looks good" failure the handoff forbids.
    if a < 0.55:
        print("Per the P1 acceptance criterion, no signal clears 0.55 -- stop "
              "here and report the negative result. Do not tune thresholds.")
    elif lo <= 0.5:
        print(f"{best_name} clears 0.55 on the point estimate but its CI still "
              f"covers chance (lower bound {lo:.3f}).\nThat is not evidence of "
              f"a usable signal at n={n}. Report the negative result; a larger "
              f"evaluation set is\nwhat would settle it, not a threshold sweep.")
    else:
        print(f"{best_name} clears 0.55 with a CI excluding chance. This is a "
              f"real signal and is worth a variant row of its own.")


if __name__ == "__main__":
    main()
