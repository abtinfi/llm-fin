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
from scipy.stats import beta


# --------------------------- UQ calibration --------------------------------

def clopper_pearson_upper(k: int, n: int, delta: float = 0.10) -> float:
    """
    Exact one-sided upper confidence limit on a binomial rate.

    `k` failures out of `n` trials: returns U such that the true rate is <= U
    with probability at least 1 - delta. Exact (Clopper-Pearson), so it is
    valid at every n rather than asymptotically -- which is the whole point
    here, since the calibration pools in this pipeline are 12-124 items.
    """
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, k + 1, n - k))


def min_certifiable_n(alpha: float = 0.10, delta: float = 0.10,
                      limit: int = 100000) -> int:
    """
    Fewest retained items that could ever certify `alpha`, i.e. the smallest n
    with clopper_pearson_upper(0, n, delta) <= alpha.

    With a PERFECT prefix -- zero errors -- a threshold still cannot be
    certified below this n. At alpha=0.10, delta=0.10 it is 22. Reported
    alongside a degenerate tau so "abstains on everything" is legible as a
    sample-size fact rather than a bug.
    """
    for n in range(1, limit + 1):
        if clopper_pearson_upper(0, n, delta) <= alpha:
            return n
    return limit + 1


def calibrate_threshold(entropies: List[float], correct: List[bool],
                        alpha: float = 0.10, delta: float = 0.10,
                        rule: str = "conformal") -> float:
    """
    Abstention threshold tau: retain an item iff its uncertainty is <= tau.

    WHAT IS GUARANTEED (rule="conformal", the default since 2026-09-08)
    -------------------------------------------------------------------
    The LARGEST tau whose SELECTIVE ERROR RATE -- the error rate among the
    calibration items it retains -- is at most `alpha` with confidence
    1 - `delta`, certified by an exact Clopper-Pearson upper bound.

    This is a bound on error among ANSWERED items. It is deliberately NOT the
    conformal COVERAGE guarantee: the textbook `ceil((n+1)(1-alpha))/n`
    quantile of the calibration scores would make `alpha` an abstention budget
    (answer 90% of items however wrong they are), which is a different
    quantity from the one `alpha` denotes everywhere else in this pipeline --
    `adaptive_conformal` below, and `calib_target_alpha` in run_eval, both
    read it as a target ERROR rate. Changing the meaning of alpha in one place
    only would make the two paths silently incomparable.

    RETURNS -inf WHEN NOTHING CAN BE CERTIFIED. That means "abstain on
    everything", the only conservative answer available. It is a real outcome
    rather than a failure: at alpha=0.10, delta=0.10 no threshold is
    certifiable until 22 consecutive correct items are retained, so a
    14-item calibration pool ALWAYS returns -inf however accurate the model
    is. `min_certifiable_n()` gives that floor and run_eval reports it, so a
    zero-coverage row explains itself. The fix for such a row is a bigger
    calibration split, not a looser threshold.

    WHY NOT THE PREVIOUS RULE (rule="legacy" reproduces it exactly).
    It selected the raw empirical optimum -- the largest tau whose calibration
    error POINT ESTIMATE was <= alpha -- with no finite-sample correction, so
    it overfitted the calibration split and undercovered on test. It is
    visible in the artifacts it produced: `summary_test.json` records the
    `uq` row calibrating to error 0.25 against a target of 0.10, and
    `summary_test_medcalc.json` records 0.50. Its no-solution fallback was
    also inverted -- it returned the SMALLEST observed uncertainty, which
    still answers the single item known to be wrong, where the docstring
    claimed the most conservative threshold.

    Ties are handled on distinct values, not on prefix positions: retention is
    `u <= tau`, so every item sharing the winning value is counted as retained
    before the bound is computed. Scoring a prefix instead would certify a set
    smaller than the one actually retained at test time.
    """
    ent = np.asarray(entropies, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    if ent.size == 0:
        # 0.0 is a RETAINING threshold for a non-negative signal; -inf is the
        # empty-input answer that abstains.
        return -float("inf")

    order = np.argsort(ent, kind="stable")
    ent, ok = ent[order], ok[order]

    if rule == "legacy":
        best = ent[0]
        for i in range(1, len(ent) + 1):
            err = 1.0 - ok[:i].mean()
            if err <= alpha:
                best = ent[i - 1]
        return float(best)
    if rule != "conformal":
        raise ValueError(f"unknown calibration rule {rule!r}")

    cum_err = np.cumsum(~ok)
    uniq, counts = np.unique(ent, return_counts=True)
    last = np.cumsum(counts) - 1          # last index of each distinct value

    best = -float("inf")
    for v, j in zip(uniq, last):
        n_ret = int(j) + 1
        k_err = int(cum_err[j])
        if clopper_pearson_upper(k_err, n_ret, delta) <= alpha:
            best = float(v)
    return best


def calibration_diagnostics(entropies: List[float], correct: List[bool],
                            alpha: float = 0.10, delta: float = 0.10,
                            rule: str = "conformal") -> Dict:
    """
    What the chosen tau actually achieved on the calibration split, plus the
    sample-size facts needed to read a degenerate one.

    `certifiable` False with `n` below `min_prefix_needed` means the pool is
    too small to certify `alpha` at this confidence -- no threshold exists,
    independent of the model.
    """
    ent = np.asarray(entropies, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    tau = calibrate_threshold(ent, ok, alpha=alpha, delta=delta, rule=rule)
    keep = ent <= tau
    n_ret = int(keep.sum())
    k_err = int((~ok[keep]).sum())
    return {
        "n": int(ent.size),
        "tau": tau,
        "retained": n_ret,
        "errors": k_err,
        "coverage": (n_ret / ent.size) if ent.size else 0.0,
        "selective_error": (k_err / n_ret) if n_ret else float("nan"),
        "cp_upper_at_tau": clopper_pearson_upper(k_err, n_ret, delta),
        "certifiable": bool(np.isfinite(tau)),
        "min_prefix_needed": min_certifiable_n(alpha, delta),
        "alpha": alpha, "delta": delta, "rule": rule,
    }


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
    cc_flags, causal_flip, spurious_flip = [], [], []
    for pid, arms in pairs.items():
        if len(arms) != 2:
            continue
        # CONTROL PAIRS. Both arms carry the SAME label: the driving value
        # moves, by a comparable amount, but does not cross the threshold. They
        # are excluded from Causal Consistency -- there is no flip to be
        # consistent with -- and instead measure how often the model changes
        # its answer when the truth did not change.
        #
        # Why this exists: on 8,000 MCQ items this model's discrimination was
        # -0.013 [-0.037, +0.013] (results/mcqpairs.md), i.e. it flipped at the
        # same rate whether or not the truth changed. Without control pairs a
        # Causal Consistency number cannot tell causal sensitivity from plain
        # prompt sensitivity. Records with no `is_control` key are treated as
        # causal pairs, so every number computed before control pairs existed
        # reproduces exactly.
        is_ctrl = any(a.get("is_control") for a in arms)
        answered_both = all(not a["abstained"] and a["pred"] is not None
                            for a in arms)
        flipped = (answered_both and arms[0]["pred"] != arms[1]["pred"])
        if is_ctrl:
            if answered_both:
                spurious_flip.append(flipped)
            continue
        if answered_both:
            causal_flip.append(flipped)
        cc_flags.append(all((not a["abstained"]) and a["pred"] == a["label"]
                            for a in arms))
    cc = float(np.mean(cc_flags)) if cc_flags else 0.0
    sf = float(np.mean(spurious_flip)) if spurious_flip else None
    cf = float(np.mean(causal_flip)) if causal_flip else None

    return {
        "n": n,
        "n_pairs": len(cc_flags),
        "accuracy": float(acc),
        "selective_accuracy": float(sel_acc),
        "causal_consistency": cc,
        # None (not 0.0) when the split carries no control pairs: "not
        # measured" and "measured at zero" are different claims and a table
        # must not print the second when it means the first.
        "spurious_flip_rate": sf,
        "causal_flip_rate": cf,
        "discrimination": (None if (sf is None or cf is None) else cf - sf),
        "n_control_pairs": len(spurious_flip),
        "violation_rate": float(viol),
        "abstention_rate": float(np.mean([r["abstained"] for r in records])),
        "coverage": float(np.mean([not r["abstained"] for r in records])),
        "unparsable_rate": float(1 - len(parsed) / n) if n else 0.0,
        "_cc_flags": cc_flags,
        "_spurious_flip_flags": spurious_flip,
        "_causal_flip_flags": causal_flip,
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
