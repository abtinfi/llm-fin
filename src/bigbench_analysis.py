"""
Scores the external-validity run produced by `bigbench_uq.py`.

The question P1 left open was not "is entropy a good signal" -- it was "was
n=56 simply too small to tell?" Every confidence interval in
`results/p1_uq_signals.md` covered chance, which means the honest reading of
that report is "undetermined", not "no signal". This script re-asks the same
question at n in the thousands, where the CIs are narrow enough that
undetermined is no longer an available answer.

WHAT IS REPORTED
----------------
  1. AUROC of each uncertainty signal as a wrongness detector, per dataset and
     pooled, with percentile bootstrap CIs. 0.5 = no signal.
  2. Paired bootstrap contrasts against the proposal's Eq. (2) entropy, on
     identical items, Holm-corrected across the family. This is the decisive
     comparison: it asks whether restricting entropy to the decision beats
     spreading it over the whole vocabulary.
  3. Risk-coverage curves and AURC -- what deferral actually buys.
  4. Expected calibration error of `option_prob_top`, which is a genuine
     probability and so can be asked whether it is calibrated, not merely
     whether it ranks.

No threshold is tuned. Operating points are printed across the whole sweep so
one can be chosen deliberately rather than reverse-engineered from the result.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uq_diagnostic import auroc as _auroc_ref      # noqa: E402
from metrics import holm_bonferroni                 # noqa: E402


def auroc(scores, labels):
    """
    Rank-based AUROC, tie-corrected, O(n log n).

    `uq_diagnostic.auroc` is the reference implementation and is numerically
    identical to this one (asserted in `_selftest` below), but it resolves ties
    with a Python loop that is O(n_unique * n). At n=56 that is invisible; at
    n=4183 x 5000 bootstrap resamples it does not finish. Same estimator, same
    numbers, different cost.
    """
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks_sorted = np.empty(len(scores), float)

    # Average ranks within each run of equal values (vectorised).
    start = 0
    boundaries = np.flatnonzero(np.diff(s_sorted)) + 1
    for end in np.append(boundaries, len(s_sorted)):
        ranks_sorted[start:end] = (start + end + 1) / 2.0
        start = end

    ranks = np.empty(len(scores), float)
    ranks[order] = ranks_sorted
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def _selftest(n=400, seed=0):
    """Assert the fast AUROC equals the reference on tie-heavy data."""
    rng = np.random.default_rng(seed)
    for scale in (1, 3, 50):          # scale=1 forces many ties after rounding
        sc = np.round(rng.normal(size=n) * scale) / scale
        lb = rng.random(n) < 0.4
        a, b = auroc(sc, lb), _auroc_ref(sc, lb)
        assert abs(a - b) < 1e-9, f"AUROC mismatch: {a} vs {b}"
    return True


# name -> (field, transform to "higher = more uncertain", short description)
SIGNALS = {
    "entropy":         ("entropy",         lambda v: v,
                        "Eq.(2) mean token entropy over whole vocabulary"),
    "max_entropy":     ("max_entropy",     lambda v: v,
                        "max token entropy over the generation"),
    "logit_margin":    ("logit_margin",    lambda v: -abs(v),
                        "top1-top2 logit gap among the answer options"),
    "answer_logprob":  ("answer_logprob",  lambda v: -v,
                        "log P of the emitted answer token"),
    "option_entropy":  ("option_entropy",  lambda v: v,
                        "entropy over the K answer options only"),
    "option_prob_top": ("option_prob_top", lambda v: -v,
                        "renormalised P of the chosen option"),
}

ORDER = ["entropy", "max_entropy", "answer_logprob", "logit_margin",
         "option_entropy", "option_prob_top"]


def load(paths):
    pools = {}
    for p in paths:
        recs = [json.loads(l) for l in Path(p).open()]
        # Records with no parsed prediction cannot be scored for correctness
        # in a way that reflects uncertainty; they are counted and excluded.
        usable = [r for r in recs if r.get("pred") is not None]
        name = Path(p).stem.replace("bigbench_", "").rsplit("_seed", 1)[0]
        pools[name] = {"recs": usable,
                       "n_raw": len(recs),
                       "n_dropped": len(recs) - len(usable)}
    return pools


def boot_auroc_ci(unc, wrong, n_boot=5000, seed=0):
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
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def vectors(recs, signal):
    field, to_unc, _ = SIGNALS[signal]
    usable = [r for r in recs if r.get(field) is not None]
    if len(usable) < 10:
        return None, None
    unc = np.array([to_unc(r[field]) for r in usable], float)
    wrong = np.array([r["pred"] != r["label"] for r in usable], bool)
    return unc, wrong


def score_signal(recs, signal, n_boot=5000):
    unc, wrong = vectors(recs, signal)
    if unc is None or wrong.all() or not wrong.any():
        return None
    a = auroc(unc, wrong)
    lo, hi = boot_auroc_ci(unc, wrong, n_boot=n_boot)
    return {"auroc": a, "lo": lo, "hi": hi, "n": len(unc),
            "err": float(wrong.mean())}


def paired_delta(recs, sig_a, sig_b, n_boot=5000, seed=0):
    """Paired bootstrap of AUROC(a) - AUROC(b) over identical items."""
    fa, ta, _ = SIGNALS[sig_a]
    fb, tb, _ = SIGNALS[sig_b]
    usable = [r for r in recs
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
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        w = wrong[idx]
        if w.all() or not w.any():
            continue
        d.append(auroc(ua[idx], w) - auroc(ub[idx], w))
    d = np.asarray(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"n": n, "delta": obs, "lo": float(lo), "hi": float(hi),
            "p": float(min(1.0, p)),
            "auroc_a": auroc(ua, wrong), "auroc_b": auroc(ub, wrong)}


def risk_coverage(recs, signal):
    """Error among retained items as coverage falls. Also returns AURC."""
    unc, wrong = vectors(recs, signal)
    if unc is None:
        return None, None
    order = unc.argsort()
    n = len(unc)
    rows, risks = [], []
    for frac in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        k = max(1, int(round(frac * n)))
        e = float(wrong[order[:k]].mean())
        rows.append((k / n, e))
    # AURC over the full sweep (mean risk across all coverage levels)
    for k in range(1, n + 1):
        risks.append(wrong[order[:k]].mean())
    return rows, float(np.mean(risks))


def ece(recs, n_bins=10):
    """Expected calibration error of option_prob_top against correctness."""
    usable = [r for r in recs if r.get("option_prob_top") is not None]
    if len(usable) < 10:
        return None
    conf = np.array([r["option_prob_top"] for r in usable], float)
    acc = np.array([r["pred"] == r["label"] for r in usable], float)
    edges = np.linspace(conf.min(), conf.max() + 1e-9, n_bins + 1)
    total, rows = 0.0, []
    for i in range(n_bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        if not m.any():
            continue
        gap = abs(acc[m].mean() - conf[m].mean())
        total += m.mean() * gap
        rows.append((edges[i], edges[i + 1], int(m.sum()),
                     float(conf[m].mean()), float(acc[m].mean())))
    return {"ece": float(total), "bins": rows, "n": len(usable),
            "mean_conf": float(conf.mean()), "acc": float(acc.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--baseline", default="entropy",
                    help="signal all others are contrasted against")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    _selftest()
    print("fast AUROC verified identical to uq_diagnostic reference "
          "implementation on tie-heavy data")

    pools = load(args.preds)
    pooled = [r for v in pools.values() for r in v["recs"]]
    if len(pools) > 1:
        pools["POOLED"] = {"recs": pooled, "n_raw": sum(
            v["n_raw"] for v in pools.values()), "n_dropped": 0}

    report = {}

    for name, info in pools.items():
        recs = info["recs"]
        n = len(recs)
        acc = sum(r["pred"] == r["label"] for r in recs) / n
        nlab = len(recs[0].get("labels", []) or [1])
        chance = 1.0 / nlab if nlab else float("nan")
        print(f"\n{'='*72}\n{name}   n={n}"
              + (f"  (dropped {info['n_dropped']} unparsable)"
                 if info.get("n_dropped") else "")
              + f"\naccuracy={acc:.3f}   error={1-acc:.3f}"
              + (f"   chance={chance:.3f}" if name != "POOLED" else "")
              + f"\n{'='*72}")

        print(f"\n{'signal':>16} {'n':>6} {'AUROC':>7} {'95% CI':>18} "
              f"{'width':>7}")
        res = {}
        for s in ORDER:
            r = score_signal(recs, s, n_boot=args.n_boot)
            res[s] = r
            if r is None:
                print(f"{s:>16} {'--':>6}   not scoreable")
                continue
            print(f"{s:>16} {r['n']:6d} {r['auroc']:7.3f}   "
                  f"[{r['lo']:.3f}, {r['hi']:.3f}] {r['hi']-r['lo']:7.3f}")

        # Paired contrasts against the proposal's signal, Holm-corrected.
        base = args.baseline
        contrasts, pvals = {}, {}
        for s in ORDER:
            if s == base or res.get(s) is None:
                continue
            d = paired_delta(recs, s, base, n_boot=args.n_boot)
            if d:
                contrasts[s] = d
                pvals[s] = d["p"]
        adj = holm_bonferroni(pvals) if pvals else {}

        if contrasts:
            print(f"\npaired contrasts vs '{base}' (same items, "
                  f"Holm-corrected across {len(pvals)})")
            print(f"{'signal':>16} {'delta':>8} {'95% CI':>20} "
                  f"{'p_raw':>9} {'p_holm':>9}")
            for s, d in sorted(contrasts.items(),
                               key=lambda kv: -kv[1]["delta"]):
                star = ""
                if adj.get(s, 1) < 0.05:
                    star = " *" if d["delta"] > 0 else " * (worse)"
                print(f"{s:>16} {d['delta']:+8.3f}   "
                      f"[{d['lo']:+.3f}, {d['hi']:+.3f}] "
                      f"{d['p']:9.4f} {adj.get(s, float('nan')):9.4f}{star}")

        # Best signal + verdict against the P1 acceptance criterion.
        scored = {k: v for k, v in res.items() if v}
        if scored:
            best = max(scored, key=lambda k: scored[k]["auroc"])
            b = scored[best]
            print(f"\nbest signal: {best}  AUROC {b['auroc']:.3f} "
                  f"[{b['lo']:.3f}, {b['hi']:.3f}]")
            if b["lo"] > 0.5 and b["auroc"] >= 0.55:
                print("  -> clears the P1 bar (>=0.55, CI excludes chance). "
                      "This is a real signal.")
            elif b["lo"] > 0.5:
                print("  -> CI excludes chance but the point estimate is under "
                      "0.55: real but too weak to act on.")
            else:
                print("  -> CI still covers chance: no usable signal.")

            rows, aurc = risk_coverage(recs, best)
            print(f"\nrisk-coverage using {best}   (AURC {aurc:.4f}; "
                  f"error at full coverage {1-acc:.3f})")
            print(f"{'coverage':>9} {'error':>8} {'vs full':>9}")
            for cov, err in rows:
                print(f"{cov:9.2f} {err:8.3f} {err-(1-acc):+9.3f}")

        e = ece(recs)
        if e:
            print(f"\ncalibration of option_prob_top: ECE={e['ece']:.4f}  "
                  f"mean confidence {e['mean_conf']:.3f} vs "
                  f"accuracy {e['acc']:.3f} "
                  f"({'over' if e['mean_conf']>e['acc'] else 'under'}"
                  f"confident by {abs(e['mean_conf']-e['acc']):.3f})")

        report[name] = {
            "n": n, "accuracy": acc,
            "auroc": {k: v for k, v in res.items()},
            "contrasts": contrasts,
            "holm": adj,
            "ece": e["ece"] if e else None,
        }

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
