"""
Unit tests for the two estimators fixed on 2026-09-08.

These are the first tests in this repository. They exist because both defects
they cover were docstring-vs-code mismatches: the prose promised a statistical
property the code did not deliver, and nothing executable checked. A test that
pins the PROPERTY (not the current output) is what makes that class of defect
fail loudly next time.

    python -m pytest tests/ -v

No GPU, no model, no data files -- everything here is pure numpy/scipy.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import (calibrate_threshold, calibration_diagnostics,
                     clopper_pearson_upper, min_certifiable_n)
from sae import matched_control_features


# ---------------------------------------------------------------------------
# S3: calibrate_threshold
# ---------------------------------------------------------------------------

def test_all_correct_retains_everything_when_pool_is_large_enough():
    """A clean pool well above the certifiable floor keeps every item."""
    n = 100
    ent = np.linspace(0.0, 1.0, n)
    tau = calibrate_threshold(ent, [True] * n, alpha=0.10, delta=0.10)
    assert np.isfinite(tau)
    assert tau == pytest.approx(ent[-1])
    assert (ent <= tau).all()


def test_no_certifiable_threshold_returns_negative_infinity():
    """
    THE REGRESSION TEST for the inverted fallback.

    Every item is wrong, so no threshold can be certified. The old code
    returned ent[0] -- the SMALLEST uncertainty -- which still answers the
    single item known to be wrong. -inf is the only conservative answer.
    """
    ent = [0.1, 0.2, 0.3, 0.4]
    tau = calibrate_threshold(ent, [False] * 4, alpha=0.10, delta=0.10)
    assert tau == -np.inf
    assert not (np.asarray(ent) <= tau).any(), "must retain nothing"


def test_small_clean_pool_still_cannot_certify():
    """
    n=14 all-correct returns -inf at alpha=0.10, delta=0.10.

    This is the documented consequence of the fix, asserted so it reads as
    intended behaviour rather than a bug: the exact bound cannot certify a 10%
    error rate from 14 observations however clean they are. The nsai_uq /
    nsai_uq_cl rows have calibration pools of 12-16 items and therefore go to
    zero coverage. The fix for those rows is a bigger calibration split.
    """
    tau = calibrate_threshold(np.linspace(0, 1, 14), [True] * 14,
                              alpha=0.10, delta=0.10)
    assert tau == -np.inf
    assert min_certifiable_n(0.10, 0.10) == 22


def test_returned_threshold_actually_satisfies_the_bound():
    """The certified property holds on the calibration set at the chosen tau."""
    rng = np.random.default_rng(0)
    ent = np.sort(rng.uniform(0, 1, 200))
    ok = rng.uniform(0, 1, 200) > (0.02 + 0.3 * ent)   # error grows with unc.
    tau = calibrate_threshold(ent, ok, alpha=0.10, delta=0.10)
    assert np.isfinite(tau)
    keep = ent <= tau
    assert clopper_pearson_upper(int((~ok[keep]).sum()), int(keep.sum()),
                                 0.10) <= 0.10


def test_ties_are_scored_on_the_retained_set_not_the_prefix():
    """
    Retention is `u <= tau`, so items sharing the winning value are all
    retained. Scoring a prefix would certify a smaller set than the one that
    actually gets answered.
    """
    ent = [0.5] * 30 + [0.9]
    ok = [True] * 30 + [False]
    tau = calibrate_threshold(ent, ok, alpha=0.10, delta=0.10)
    keep = np.asarray(ent) <= tau
    assert keep.sum() == 30, "all tied items must be counted together"
    assert clopper_pearson_upper(0, 30, 0.10) <= 0.10


def test_stricter_delta_is_never_more_permissive():
    """Monotonicity: more confidence demanded can only lower tau."""
    rng = np.random.default_rng(1)
    ent = np.sort(rng.uniform(0, 1, 120))
    ok = rng.uniform(0, 1, 120) > 0.05
    taus = [calibrate_threshold(ent, ok, alpha=0.10, delta=d)
            for d in (0.20, 0.10, 0.05)]
    assert taus[0] >= taus[1] >= taus[2]


def test_empty_input_abstains():
    """0.0 would RETAIN everything for a non-negative signal."""
    assert calibrate_threshold([], [], alpha=0.10) == -np.inf


def test_legacy_rule_reproduces_the_old_behaviour():
    """
    The pre-2026-09-08 numbers must stay reproducible, including the branch
    where the old rule overshot its own target.
    """
    ent = [0.1, 0.2, 0.3, 0.4, 0.5]
    ok = [True, True, False, True, False]
    # old code: largest prefix whose POINT-ESTIMATE error <= alpha
    assert calibrate_threshold(ent, ok, alpha=0.50, rule="legacy") \
        == pytest.approx(0.5)
    # and its inverted fallback: smallest observed value, not -inf
    assert calibrate_threshold(ent, [False] * 5, alpha=0.10,
                               rule="legacy") == pytest.approx(0.1)


def test_unknown_rule_raises():
    with pytest.raises(ValueError):
        calibrate_threshold([0.1], [True], rule="nonsense")


def test_diagnostics_explain_a_degenerate_row():
    d = calibration_diagnostics(np.linspace(0, 1, 14), [True] * 14,
                                alpha=0.10, delta=0.10)
    assert d["certifiable"] is False
    assert d["retained"] == 0
    assert d["coverage"] == 0.0
    assert d["n"] < d["min_prefix_needed"]


def test_clopper_pearson_is_an_upper_bound_on_the_point_estimate():
    for k, n in ((0, 10), (1, 50), (5, 100), (20, 400)):
        assert clopper_pearson_upper(k, n, 0.10) >= k / n
    assert clopper_pearson_upper(7, 7, 0.10) == 1.0
    assert clopper_pearson_upper(0, 0, 0.10) == 1.0


# ---------------------------------------------------------------------------
# S1: matched_control_features
# ---------------------------------------------------------------------------

def _dictionary(seed=0, n_feat=16384, dead_frac=0.42):
    """A firing-rate vector with this project's realised dead fraction."""
    rng = np.random.default_rng(seed)
    n_active = np.zeros(n_feat, dtype=np.int64)
    live = rng.permutation(n_feat)[int(dead_frac * n_feat):]
    n_active[live] = rng.lognormal(4.0, 2.0, live.size).astype(np.int64) + 1
    return n_active


def test_dead_features_are_never_drawn():
    """THE REGRESSION TEST for S1: a dead control measures exactly zero."""
    n_active = _dictionary()
    live = np.flatnonzero(n_active > 0)
    rng = np.random.default_rng(0)
    for f in live[:40]:
        ctrl = matched_control_features(int(f), n_active, n_controls=5,
                                        rng=rng)
        assert (n_active[ctrl] > 0).all(), f"drew a dead control for #{f}"


def test_target_is_never_its_own_control():
    n_active = _dictionary()
    rng = np.random.default_rng(0)
    for f in np.flatnonzero(n_active > 0)[:40]:
        assert int(f) not in set(
            matched_control_features(int(f), n_active, rng=rng).tolist())


def test_controls_are_closer_in_firing_rate_than_a_uniform_draw():
    """The property the old code claimed and did not have."""
    n_active = _dictionary()
    live = np.flatnonzero(n_active > 0)
    rng = np.random.default_rng(0)
    matched_gap, uniform_gap = [], []
    for f in live[:200]:
        lf = np.log(n_active[int(f)] + 1.0)
        ctrl = matched_control_features(int(f), n_active, n_controls=5,
                                        rng=rng)
        matched_gap.append(np.abs(np.log(n_active[ctrl] + 1.0) - lf).mean())
        unif = rng.integers(0, n_active.size, 5)
        uniform_gap.append(np.abs(np.log(n_active[unif] + 1.0) - lf).mean())
    assert np.mean(matched_gap) < 0.25 * np.mean(uniform_gap)


def test_extreme_frequency_features_still_get_a_full_pool():
    """Nearest-K never returns an empty pool, unlike a multiplicative band."""
    n_active = _dictionary()
    live = np.flatnonzero(n_active > 0)
    for f in (live[np.argmin(n_active[live])], live[np.argmax(n_active[live])]):
        ctrl = matched_control_features(int(f), n_active, n_controls=5,
                                        rng=np.random.default_rng(0))
        assert ctrl.size == 5


def test_draws_are_deterministic_under_a_fixed_seed():
    n_active = _dictionary()
    a = matched_control_features(100, n_active, rng=np.random.default_rng(7))
    b = matched_control_features(100, n_active, rng=np.random.default_rng(7))
    assert a.tolist() == b.tolist()


def test_controls_are_distinct_within_a_draw():
    n_active = _dictionary()
    ctrl = matched_control_features(
        int(np.flatnonzero(n_active > 0)[0]), n_active, n_controls=5,
        rng=np.random.default_rng(3))
    assert len(set(ctrl.tolist())) == ctrl.size


def test_degenerate_dictionary_returns_empty_rather_than_raising():
    """One live feature which IS the target: nothing valid to draw."""
    n_active = np.zeros(100, dtype=np.int64)
    n_active[5] = 42
    assert matched_control_features(5, n_active).size == 0


def test_pool_size_caps_the_candidate_set():
    n_active = np.arange(1, 501, dtype=np.int64)     # all live
    ctrl = matched_control_features(250, n_active, n_controls=5, pool_k=10,
                                    rng=np.random.default_rng(0))
    # the 10 nearest to index 250 in log space are its immediate neighbours
    assert (np.abs(ctrl - 250) <= 6).all()
