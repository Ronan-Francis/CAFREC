"""Unit tests for causal session-context features (RON-16/17/18, RON-60).

These assert the three properties that matter for the dissertation:
  * the numbers are correct on hand-computable inputs,
  * features are CAUSAL (row i depends only on rows before i) so the held-out
    target never leaks into its own context,
  * sessions are cut on the 30-minute gap.
"""
import numpy as np
import pandas as pd
import pytest

from cafrec.features.context import (
    CTX_FIELDS,
    ENTROPY_BINS,
    SESSION_GAP_MS,
    compute_context_features,
    dwell_ratio,
    segment_sessions,
)

MIN = 60 * 1000  # one minute in ms


# --------------------------------------------------------------------------- #
#  dwell_ratio
# --------------------------------------------------------------------------- #
def test_dwell_ratio_clips_and_guards_zero_duration():
    r = dwell_ratio([50, 200, 30], [100, 100, 0])
    # 0.5 kept; 2.0 clipped to 1.0; zero-duration -> 0.0
    assert r.tolist() == [0.5, 1.0, 0.0]


# --------------------------------------------------------------------------- #
#  segment_sessions
# --------------------------------------------------------------------------- #
def test_segment_sessions_cuts_on_30min_gap():
    # gaps: 1min, 40min (cut), 1min  ->  sessions [0, 0, 1, 1]
    t = np.array([0, 1 * MIN, 1 * MIN + 40 * MIN, 1 * MIN + 41 * MIN])
    assert segment_sessions(t).tolist() == [0, 0, 1, 1]


def test_segment_sessions_boundary_is_strict():
    # a gap of exactly 30 min is NOT a cut (> gap, not >=); 30 min + 1 ms IS.
    t = np.array([0, SESSION_GAP_MS, 2 * SESSION_GAP_MS + 1])
    assert segment_sessions(t).tolist() == [0, 0, 1]


def test_segment_sessions_empty():
    assert segment_sessions(np.array([], dtype=np.int64)).tolist() == []


# --------------------------------------------------------------------------- #
#  session_len
# --------------------------------------------------------------------------- #
def _one_user(dwell, category, times=None):
    n = len(dwell)
    if times is None:
        times = np.arange(n) * MIN  # 1-min apart -> single session
    return pd.DataFrame(
        {"user_id": np.ones(n, int), "time_ms": times,
         "dwell": dwell, "category": category}
    )


def test_session_len_is_causal_position_normalised():
    df = _one_user([0.5] * 4, [1, 1, 1, 1])
    out = compute_context_features(df, cap=50)
    # positions 0,1,2,3 over history -> /50
    assert np.allclose(out["session_len"].to_numpy(), np.array([0, 1, 2, 3]) / 50)


def test_session_len_resets_each_session():
    # two items, big gap, two items -> lengths 0,1,0,1
    times = np.array([0, MIN, MIN + 40 * MIN, MIN + 41 * MIN])
    df = _one_user([0.5] * 4, [1, 1, 1, 1], times=times)
    out = compute_context_features(df, cap=50)
    assert (out["session_len"].to_numpy() * 50).tolist() == [0, 1, 0, 1]


# --------------------------------------------------------------------------- #
#  cat_drift
# --------------------------------------------------------------------------- #
def test_cat_drift_all_same_category_is_zero():
    df = _one_user([0.5] * 4, [7, 7, 7, 7])
    out = compute_context_features(df)
    assert out["cat_drift"].to_numpy().tolist() == [0.0, 0.0, 0.0, 0.0]


def test_cat_drift_alternating_categories_approaches_one():
    df = _one_user([0.5] * 4, [1, 2, 3, 4])   # every consecutive pair changes
    out = compute_context_features(df)
    # row0: no history -> 0; row1: hist=[1] 0 pairs -> 0; row2: hist pairs=(1,2)=1 change/1=1;
    # row3: pairs among (1,2),(2,3)=2 changes/2=1
    assert out["cat_drift"].to_numpy().tolist() == [0.0, 0.0, 1.0, 1.0]


def test_cat_drift_half_changes():
    # categories [5,5,6,6]. drift[3] uses only the CAUSAL history (rows 0..2 =
    # cats 5,5,6): pairs (5,5) no, (5,6) yes -> 1 change / 2 pairs = 0.5.
    # The target's own category (row 3) must NOT enter its drift (no leakage).
    df = _one_user([0.5] * 4, [5, 5, 6, 6])
    out = compute_context_features(df)
    drift = out["cat_drift"].to_numpy()
    assert drift[3] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
#  dwell_entropy
# --------------------------------------------------------------------------- #
def test_dwell_entropy_zero_when_history_single_bin():
    # all dwell ratios land in the same bin -> zero entropy
    df = _one_user([0.05] * 5, [1] * 5)
    out = compute_context_features(df)
    assert np.allclose(out["dwell_entropy"].to_numpy(), 0.0)


def test_dwell_entropy_maximal_when_history_uniform_over_bins():
    # one item in each of the 10 bins -> entropy == 1.0 once all bins populated
    ratios = [(b + 0.5) / ENTROPY_BINS for b in range(ENTROPY_BINS)]
    df = _one_user(ratios, [1] * ENTROPY_BINS)
    out = compute_context_features(df)
    # the LAST row's history is the first 9 distinct bins (uniform) -> high, <1;
    # append one more row so history covers all 10 bins uniformly:
    ratios2 = ratios + [ratios[0]]
    df2 = _one_user(ratios2, [1] * (ENTROPY_BINS + 1))
    out2 = compute_context_features(df2)
    # history of the last row = 10 items, one per bin -> perfectly uniform -> 1.0
    assert out2["dwell_entropy"].to_numpy()[-1] == pytest.approx(1.0)


def test_first_row_of_session_has_zero_context():
    df = _one_user([0.9, 0.1, 0.5], [1, 2, 3])
    out = compute_context_features(df)
    row0 = out.iloc[0]
    assert row0["session_len"] == 0.0
    assert row0["dwell_entropy"] == 0.0
    assert row0["cat_drift"] == 0.0


# --------------------------------------------------------------------------- #
#  Causality / no-leakage (RON-60)
# --------------------------------------------------------------------------- #
def test_features_are_causal_appending_a_row_does_not_change_earlier_rows():
    base = _one_user([0.2, 0.8, 0.4, 0.6], [1, 2, 2, 3])
    extended = _one_user([0.2, 0.8, 0.4, 0.6, 0.1], [1, 2, 2, 3, 9])
    ob = compute_context_features(base)
    oe = compute_context_features(extended)
    for col in CTX_FIELDS:
        # the first 4 rows must be identical whether or not a 5th exists
        assert np.allclose(ob[col].to_numpy(), oe[col].to_numpy()[:4]), col


def test_multiple_users_are_independent():
    d1 = _one_user([0.5] * 3, [1, 2, 3])
    d2 = _one_user([0.5] * 2, [9, 9])
    d2["user_id"] = 2
    out = compute_context_features(pd.concat([d1, d2], ignore_index=True))
    u2 = out[out["user_id"] == 2].sort_values("time_ms")
    # user 2 has its own fresh session: lengths 0,1 and zero drift (same cat)
    assert (u2["session_len"].to_numpy() * 50).tolist() == [0, 1]
    assert u2["cat_drift"].to_numpy().tolist() == [0.0, 0.0]


def test_missing_columns_raise():
    with pytest.raises(KeyError):
        compute_context_features(pd.DataFrame({"user_id": [1], "time_ms": [0]}))


# --------------------------------------------------------------------------- #
#  RON-60 split integrity: the held-out (last) interaction never leaks into its
#  own context. Mirrors the T1.1 notebook's leakage assertion in the harness.
# --------------------------------------------------------------------------- #
def test_held_out_target_context_is_prefix_only():
    rng = np.random.default_rng(0)
    rows = []
    for u in range(1, 6):
        n = int(rng.integers(3, 9))
        for k in range(n):
            rows.append((u, k * MIN, float(rng.random()), int(rng.integers(1, 4))))
    df = pd.DataFrame(rows, columns=["user_id", "time_ms", "dwell", "category"])
    out = compute_context_features(df)

    # For each user the LOO test target is the last (time-ordered) interaction.
    for u, grp in out.groupby("user_id"):
        grp = grp.sort_values("time_ms")
        target = grp.iloc[-1]
        # its session_len must equal the number of prior items in its session
        same_sess = grp[grp["session_id"] == target["session_id"]]
        expected_len = len(same_sess) - 1
        assert target["session_len"] * 50 == pytest.approx(expected_len), u
