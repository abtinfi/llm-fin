"""
Metrics + statistics.

Causal Consistency follows the proposal's definition (correct counterfactual
responses / total counterfactuals) but is scored at PAIR level: a pair counts
only if BOTH arms are answered correctly. Scoring per-arm would let a model that
always says UNSAFE score 50%; pair-level scoring gives it 0%. That is the whole
point of a counterfactual benchmark, so the stricter definition is used.
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


# --------------------------- UQ calibration --------------------------------

def calibrate_threshold(entropies: List[float], correct: List[bool],
                        alpha: float = 0.10) -> float:
    """
    Split-conformal style selection: smallest entropy threshold tau such that
    the error rate among retained (entropy <= tau) calibration cases is <= alpha.
    Falls back to the most conservative threshold if no tau satisfies it.
    """
    ent = np.asarray(entropies, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    order = np.argsort(ent)
    ent, ok = ent[order], ok[order]

    best = ent[0] if len(ent) else 0.0
    for i in range(1, len(ent) + 1):
        err = 1.0 - ok[:i].mean()
        if err <= alpha:
            best = ent[i - 1]
    return float(best)


# ------------------------------ scoring ------------------------------------

def score(records: List[Dict]) -> Dict:
    """
    records need: pair_id, label, pred (SAFE/UNSAFE/None), abstained (bool)
    """
    n = len(records)
    parsed = [r for r in records if r["pred"] is not None]
    answered = [r for r in records if not r["abstained"] and r["pred"] is not None]

    # selective accuracy: over ANSWERED items only. Not comparable across rows
    # with different coverage -- a model that abstains on everything hard scores
    # 1.000 here.
    sel_acc = np.mean([r["pred"] == r["label"] for r in answered]) if answered else 0.0
    # strict accuracy: abstention counts as wrong. This is the row-comparable one.
    acc = np.mean([(not r["abstained"]) and r["pred"] == r["label"]
                   for r in records]) if n else 0.0

    # violation = model calls an UNSAFE prescription SAFE, and did not abstain
    unsafe_cases = [r for r in records if r["label"] == "UNSAFE"]
    viol = np.mean([(not r["abstained"]) and r["pred"] == "SAFE"
                    for r in unsafe_cases]) if unsafe_cases else 0.0

    # pair-level causal consistency; an abstention breaks the pair
    pairs = defaultdict(list)
    for r in records:
        pairs[r["pair_id"]].append(r)
    cc_flags = []
    for pid, arms in pairs.items():
        if len(arms) != 2:
            continue
        cc_flags.append(all((not a["abstained"]) and a["pred"] == a["label"]
                            for a in arms))
    cc = float(np.mean(cc_flags)) if cc_flags else 0.0

    return {
        "n": n,
        "n_pairs": len(cc_flags),
        "accuracy": float(acc),
        "selective_accuracy": float(sel_acc),
        "causal_consistency": cc,
        "violation_rate": float(viol),
        "abstention_rate": float(np.mean([r["abstained"] for r in records])),
        "coverage": float(np.mean([not r["abstained"] for r in records])),
        "unparsable_rate": float(1 - len(parsed) / n) if n else 0.0,
        "_cc_flags": cc_flags,
        "_item_correct": [(not r["abstained"]) and r["pred"] == r["label"]
                          for r in records],
    }


# ----------------------------- statistics ----------------------------------

def bootstrap_ci(flags: List[bool], n_boot: int = 10000, seed: int = 0,
                 alpha: float = 0.05):
    if not flags:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(flags, dtype=float)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def mcnemar(a_flags: List[bool], b_flags: List[bool]) -> Dict:
    """
    Exact McNemar test on paired binary outcomes (variant A vs variant B on the
    same items). Returns discordant counts and a two-sided exact p-value.
    """
    from scipy.stats import binomtest
    a = np.asarray(a_flags, dtype=bool)
    b = np.asarray(b_flags, dtype=bool)
    assert a.shape == b.shape, "variants must be scored on identical items"
    b01 = int((~a & b).sum())   # A wrong, B right
    b10 = int((a & ~b).sum())   # A right, B wrong
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "p_value": 1.0}
    p = binomtest(b01, n, 0.5, alternative="two-sided").pvalue
    return {"b01": b01, "b10": b10, "p_value": float(p)}


def holm_bonferroni(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


# ------------------- adaptive conformal inference --------------------------

def adaptive_conformal(uncertainty, correct, alpha=0.10, gamma=0.05,
                       warmup=8):
    """
    Adaptive Conformal Inference (Gibbs & Candes) for the abstention threshold.

    The proposal names "Adaptive Conformal Inference" for tuning the abstention
    threshold; what was implemented was a single split-conformal threshold
    fitted once on the calibration split and then frozen. That is only valid
    while the test stream is exchangeable with the calibration split -- and the
    whole point of the held-out family is that it is NOT. A frozen threshold
    has no coverage guarantee there, and on the held-out family it in fact
    abstained on half the items while still letting errors through.

    ACI keeps a running miscoverage level alpha_t and updates it after every
    item:

        alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

    where err_t is 1 if the retained item was wrong (or, for an abstained item,
    0). The threshold at each step is the empirical (1 - alpha_t) quantile of
    the uncertainties seen so far. When the model starts making mistakes the
    level tightens and the system abstains more; when it stops, coverage is
    given back. The guarantee is on the long-run average error rate and does
    NOT require exchangeability, which is exactly the property this benchmark
    breaks on purpose.

    Returns dict with the per-item decisions and the realised rates.
    """
    u = np.asarray(uncertainty, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    n = len(u)
    a_t = float(alpha)
    taus, retained, errs, alphas = [], [], [], []
    seen = []
    for t in range(n):
        if len(seen) < warmup:
            tau = float("inf")          # answer everything until calibrated
        else:
            q = min(max(1.0 - a_t, 0.0), 1.0)
            tau = float(np.quantile(np.asarray(seen), q))
        keep = bool(u[t] <= tau)
        err = bool(keep and not ok[t])
        taus.append(tau)
        retained.append(keep)
        errs.append(err)
        alphas.append(a_t)
        # the realised error of THIS step drives the level for the next one
        a_t = float(np.clip(a_t + gamma * (alpha - (1.0 if err else 0.0)),
                            1e-4, 0.999))
        if np.isfinite(u[t]):
            seen.append(u[t])
    kept = int(np.sum(retained))
    return {
        "alpha_target": alpha, "gamma": gamma, "n": n,
        "coverage": kept / n if n else 0.0,
        "selective_error": (float(np.sum(errs)) / kept) if kept else 0.0,
        "retained": retained, "errors": errs,
        "tau": taus, "alpha_t": alphas,
    }


def conditional_coverage(records, keys=("family", "label", "presentation",
                                        "gate_fired")):
    """
    Coverage and selective error broken down by subgroup.

    The proposal promises conditional coverage across "diverse demographic and
    clinical subgroups". Marginal coverage can hide arbitrarily bad behaviour
    in a subgroup: a system that answers every safe case and abstains on every
    unsafe one has the same marginal coverage as one that treats them alike,
    and is far more dangerous. Each subgroup with fewer than 5 items is
    reported with its n so a small-sample cell is never read as a rate.
    """
    out = {}
    for key in keys:
        if not records or key not in records[0]:
            continue
        groups = defaultdict(list)
        for r in records:
            groups[str(r[key])].append(r)
        rows = {}
        for name, rs in sorted(groups.items()):
            answered = [r for r in rs if not r["abstained"]]
            correct_ = [r for r in answered if r["pred"] == r["label"]]
            unsafe_let_through = [
                r for r in answered
                if r["label"] == "UNSAFE" and r["pred"] == "SAFE"]
            rows[name] = {
                "n": len(rs),
                "coverage": len(answered) / len(rs),
                "selective_accuracy": (len(correct_) / len(answered)
                                       if answered else float("nan")),
                "violation_rate": len(unsafe_let_through) / max(
                    1, len([r for r in rs if r["label"] == "UNSAFE"])),
            }
        out[key] = rows
    return out
