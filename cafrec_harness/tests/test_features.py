"""Unit tests for causal session-context features (RON-16/17/18/19, RON-60).

These assert the properties that matter for the dissertation:
  * the numbers are correct on hand-computable inputs,
  * features are CAUSAL (row i depends only on rows before i in its session) so
    the held-out target never leaks into its own context,
  * sessions are cut on the 30-minute gap,
  * `prefix_policy_flag` reflects an is_rand transition seen in the prefix,
  * the z-scaler is fit on TRAIN rows only.
"""
import numpy as np
import pandas as pd
import pytest

from cafrec.features.context import (
    BINARY,
    CONTINUOUS,
    ENTROPY_BINS,
    SESSION_GAP_MS,
    compute_context_features,
    dwell_ratio,
    segment_sessions,
    standardize_context,
)

MIN = 60 * 1000  # one minute in ms
RAW_FEATURES = CONTINUOUS + BINARY


def _one_user(dwell, category, times=None, is_rand=None, user_id=1):
    n = len(dwell)
    if times is None:
        times = np.arange(n) * MIN  # 1-min apart -> single session
    if is_rand is None:
        is_rand = np.zeros(n, dtype=int)
    return pd.DataFrame(
        {"user_id": np.full(n, user_id, int), "time_ms": np.asarray(times),
         "dwell": dwell, "category": category, "is_rand": is_rand}
    )


# --------------------------------------------------------------------------- #
#  dwell_ratio
# --------------------------------------------------------------------------- #
def test_dwell_ratio_clips_and_guards_zero_duration():
    r = dwell_ratio([50, 200, 30], [100, 100, 0])
    assert r.tolist() == [0.5, 1.0, 0.0]   # 0.5 kept; 2.0 clipped; /0 -> 0


# --------------------------------------------------------------------------- #
#  segment_sessions
# --------------------------------------------------------------------------- #
def test_segment_sessions_cuts_on_30min_gap():
    t = np.array([0, 1 * MIN, 1 * MIN + 40 * MIN, 1 * MIN + 41 * MIN])
    assert segment_sessions(t).tolist() == [0, 0, 1, 1]


def test_segment_sessions_boundary_is_strict():
    # a gap of exactly 30 min is NOT a cut (> gap, not >=); 30 min + 1 ms IS.
    t = np.array([0, SESSION_GAP_MS, 2 * SESSION_GAP_MS + 1])
    assert segment_sessions(t).tolist() == [0, 0, 1]


def test_segment_sessions_empty():
    assert segment_sessions(np.array([], dtype=np.int64)).tolist() == []


# --------------------------------------------------------------------------- #
#  prefix_session_len  (raw counts, log)
# --------------------------------------------------------------------------- #
def test_prefix_session_len_is_causal_position():
    out = compute_context_features(_one_user([0.5] * 4, [1, 1, 1, 1]))
    assert out["prefix_session_len"].tolist() == [0, 1, 2, 3]
    assert np.allclose(out["prefix_session_len_log"].to_numpy(),
                       np.log1p([0, 1, 2, 3]))


def test_prefix_session_len_resets_each_session():
    times = np.array([0, MIN, MIN + 40 * MIN, MIN + 41 * MIN])
    out = compute_context_features(_one_user([0.5] * 4, [1, 1, 1, 1], times=times))
    assert out["prefix_session_len"].tolist() == [0, 1, 0, 1]


# --------------------------------------------------------------------------- #
#  prefix_category_drift
# --------------------------------------------------------------------------- #
def test_cat_drift_all_same_category_is_zero():
    out = compute_context_features(_one_user([0.5] * 4, [7, 7, 7, 7]))
    assert out["prefix_category_drift"].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_cat_drift_alternating_categories_approaches_one():
    out = compute_context_features(_one_user([0.5] * 4, [1, 2, 3, 4]))
    assert out["prefix_category_drift"].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_cat_drift_half_changes_is_prefix_only():
    # cats [5,5,6,6]: drift[3] uses only rows 0..2 (5,5,6): 1 change / 2 pairs.
    # The target's own category (row 3) must NOT enter its drift (no leakage).
    out = compute_context_features(_one_user([0.5] * 4, [5, 5, 6, 6]))
    assert out["prefix_category_drift"].to_numpy()[3] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
#  prefix_dwell_entropy  (un-normalised base-2 entropy)
# --------------------------------------------------------------------------- #
def test_dwell_entropy_zero_when_history_single_bin():
    out = compute_context_features(_one_user([0.05] * 5, [1] * 5))
    assert np.allclose(out["prefix_dwell_entropy"].to_numpy(), 0.0)


def test_dwell_entropy_maximal_when_history_uniform_over_bins():
    # one item per bin -> once the prefix covers all 10 bins uniformly the
    # base-2 entropy equals log2(10).
    ratios = [(b + 0.5) / ENTROPY_BINS for b in range(ENTROPY_BINS)] + [0.05]
    out = compute_context_features(_one_user(ratios, [1] * (ENTROPY_BINS + 1)))
    assert out["prefix_dwell_entropy"].to_numpy()[-1] == pytest.approx(np.log2(ENTROPY_BINS))


# --------------------------------------------------------------------------- #
#  prefix_policy_flag  (is_rand transition in the prefix)
# --------------------------------------------------------------------------- #
def test_policy_flag_zero_when_all_organic():
    out = compute_context_features(_one_user([0.5] * 4, [1] * 4, is_rand=[0, 0, 0, 0]))
    assert out["prefix_policy_flag"].tolist() == [0, 0, 0, 0]


def test_policy_flag_fires_after_transition_in_prefix():
    # is_rand [0,0,1,1]: the 0->1 transition happens AT row 2; only row 3's
    # prefix contains it, so the flag is [0,0,0,1] (strictly causal).
    out = compute_context_features(_one_user([0.5] * 4, [1] * 4, is_rand=[0, 0, 1, 1]))
    assert out["prefix_policy_flag"].tolist() == [0, 0, 0, 1]


# --------------------------------------------------------------------------- #
#  inter_session_gap_log + is_first_session
# --------------------------------------------------------------------------- #
def test_first_session_flag_and_gap():
    # two sessions split by a 40-min gap. Rows in the first session are flagged
    # is_first_session=1; the second session's gap is log1p of the real gap.
    times = np.array([0, MIN, MIN + 40 * MIN, MIN + 41 * MIN])
    out = compute_context_features(_one_user([0.5] * 4, [1] * 4, times=times))
    assert out["is_first_session"].tolist() == [1, 1, 0, 0]
    gap_ms = (MIN + 40 * MIN) - MIN            # start(s1) - end(s0)
    assert out["inter_session_gap_log"].to_numpy()[2] == pytest.approx(np.log1p(gap_ms))


def test_first_row_of_session_has_zero_prefix_context():
    out = compute_context_features(_one_user([0.9, 0.1, 0.5], [1, 2, 3]))
    row0 = out.iloc[0]
    assert row0["prefix_session_len"] == 0
    assert row0["prefix_dwell_entropy"] == 0.0
    assert row0["prefix_category_drift"] == 0.0
    assert row0["prefix_policy_flag"] == 0


# --------------------------------------------------------------------------- #
#  Causality / no-leakage (RON-60)
# --------------------------------------------------------------------------- #
def test_features_are_causal_appending_a_row_does_not_change_earlier_rows():
    base = _one_user([0.2, 0.8, 0.4, 0.6], [1, 2, 2, 3])
    extended = _one_user([0.2, 0.8, 0.4, 0.6, 0.1], [1, 2, 2, 3, 9])
    ob = compute_context_features(base)
    oe = compute_context_features(extended)
    for col in RAW_FEATURES:
        assert np.allclose(ob[col].to_numpy(), oe[col].to_numpy()[:4]), col


def test_multiple_users_are_independent():
    d1 = _one_user([0.5] * 3, [1, 2, 3], user_id=1)
    d2 = _one_user([0.5] * 2, [9, 9], user_id=2)
    out = compute_context_features(pd.concat([d1, d2], ignore_index=True))
    u2 = out[out["user_id"] == 2].sort_values("time_ms")
    assert u2["prefix_session_len"].tolist() == [0, 1]
    assert u2["prefix_category_drift"].tolist() == [0.0, 0.0]


def test_missing_columns_raise():
    with pytest.raises(KeyError):
        compute_context_features(pd.DataFrame({"user_id": [1], "time_ms": [0]}))


def test_held_out_target_context_is_prefix_only():
    rng = np.random.default_rng(0)
    rows = []
    for u in range(1, 6):
        n = int(rng.integers(3, 9))
        for k in range(n):
            rows.append((u, k * MIN, float(rng.random()),
                         int(rng.integers(1, 4)), 0))
    df = pd.DataFrame(rows, columns=["user_id", "time_ms", "dwell",
                                     "category", "is_rand"])
    out = compute_context_features(df)
    for u, grp in out.groupby("user_id"):
        grp = grp.sort_values("time_ms")
        target = grp.iloc[-1]
        same_sess = grp[grp["session_id"] == target["session_id"]]
        assert target["prefix_session_len"] == len(same_sess) - 1, u


# --------------------------------------------------------------------------- #
#  standardize_context: train-only fit
# --------------------------------------------------------------------------- #
def test_standardize_fits_on_train_rows_only():
    out = compute_context_features(_one_user([0.5] * 6, list(range(6))))
    # hold out the last two rows (leave-one-out): fit the scaler on rows [0..3]
    train_mask = np.array([True, True, True, True, False, False])
    scaled, params = standardize_context(out.copy(), train_mask)

    # every continuous feature gains a *_z column; train rows are ~zero-mean.
    for c in CONTINUOUS:
        assert c + "_z" in scaled.columns
        assert scaled.loc[train_mask, c + "_z"].mean() == pytest.approx(0.0, abs=1e-5)
    assert params["fit_rows"] == 4
    assert len(params["mean"]) == len(CONTINUOUS)
    assert params["features"] == CONTINUOUS
