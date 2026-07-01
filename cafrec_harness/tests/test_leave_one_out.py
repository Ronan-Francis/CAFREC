"""Tests for the leave-one-out protocol (RON-13)."""
import numpy as np
import pytest

from cafrec.data.leave_one_out import (
    leave_one_out_split,
    seen_items,
    sample_negatives,
    build_candidate_matrix,
    save_eval_cache,
    load_eval_cache,
    POSITIVE_COL,
)
from cafrec.eval.metrics import ranks_from_scores, per_user_metrics


# --- toy interaction log: user -> items in time order ----------------------- #
#   user 1: 10, 11, 12, 13   (test=13, valid=12, train=[10,11])
#   user 2: 20, 21, 22       (test=22, valid=21, train=[20])
#   user 3: 30, 31           (test=31, valid=None, train=[30])
#   user 4: 40               (dropped: length 1)
def _toy_log():
    rows = [
        (1, 10, 0), (1, 11, 1), (1, 12, 2), (1, 13, 3),
        (2, 20, 0), (2, 21, 1), (2, 22, 2),
        (3, 30, 5), (3, 31, 6),
        (4, 40, 0),
    ]
    u, i, t = zip(*rows)
    return np.array(u), np.array(i), np.array(t)


def test_split_targets_and_order():
    u, i, t = _toy_log()
    split = leave_one_out_split(u, i, t)

    assert set(split) == {1, 2, 3}                 # user 4 dropped (too short)
    assert split[1]["test"] == 13
    assert split[1]["valid"] == 12
    assert split[1]["train"].tolist() == [10, 11]
    assert split[3]["valid"] is None               # only 2 interactions
    assert split[3]["train"].tolist() == [30]


def test_split_respects_time_not_input_order():
    # shuffle input rows; timestamps must still drive the ordering
    u = np.array([1, 1, 1, 1])
    i = np.array([13, 10, 12, 11])
    t = np.array([3, 0, 2, 1])
    split = leave_one_out_split(u, i, t)
    assert split[1]["test"] == 13
    assert split[1]["train"].tolist() == [10, 11]


def test_negatives_exclude_seen_and_have_right_count():
    u, i, t = _toy_log()
    split = leave_one_out_split(u, i, t)
    seen = seen_items(split)
    catalogue = np.arange(0, 60)                    # ids 0..59
    negs = sample_negatives(seen, catalogue, n_neg=20, seed=7)

    for user, arr in negs.items():
        assert arr.shape == (20,)
        assert len(set(arr.tolist())) == 20         # distinct
        assert seen[user].isdisjoint(set(arr.tolist()))  # no leakage


def test_negatives_are_reproducible():
    u, i, t = _toy_log()
    seen = seen_items(leave_one_out_split(u, i, t))
    cat = np.arange(0, 60)
    a = sample_negatives(seen, cat, n_neg=20, seed=123)
    b = sample_negatives(seen, cat, n_neg=20, seed=123)
    c = sample_negatives(seen, cat, n_neg=20, seed=999)
    for user in a:
        assert np.array_equal(a[user], b[user])     # same seed -> identical
    # different seed -> at least one user differs
    assert any(not np.array_equal(a[user], c[user]) for user in a)


def test_popularity_strategy_runs_and_excludes_seen():
    u, i, t = _toy_log()
    seen = seen_items(leave_one_out_split(u, i, t))
    cat = np.arange(0, 60)
    pop = np.linspace(1, 10, num=60)                # non-uniform weights
    negs = sample_negatives(seen, cat, n_neg=15, seed=1,
                            strategy="popularity", popularity=pop)
    for user, arr in negs.items():
        assert arr.shape == (15,)
        assert seen[user].isdisjoint(set(arr.tolist()))


def test_popularity_requires_weights():
    with pytest.raises(ValueError):
        sample_negatives({1: set()}, np.arange(10), n_neg=3, strategy="popularity")


def test_candidate_matrix_layout_and_rank_integration():
    u, i, t = _toy_log()
    split = leave_one_out_split(u, i, t)
    seen = seen_items(split)
    cat = np.arange(0, 60)
    negs = sample_negatives(seen, cat, n_neg=5, seed=42)
    users, cand = build_candidate_matrix(split, negs, target="test")

    # column 0 is the held-out positive for each user
    for row, user in zip(cand, users):
        assert row[POSITIVE_COL] == split[user]["test"]
    assert cand.shape == (3, 6)                      # 3 users, 1 pos + 5 neg

    # If the model scored the positive highest for every user, every rank == 1.
    scores = np.zeros_like(cand, dtype=float)
    scores[:, POSITIVE_COL] = 1.0
    ranks = ranks_from_scores(scores, target_col=POSITIVE_COL)
    assert ranks.tolist() == [1, 1, 1]
    assert per_user_metrics(ranks, k=10)["HR@10"].tolist() == [1.0, 1.0, 1.0]


def test_cache_roundtrip(tmp_path):
    u, i, t = _toy_log()
    split = leave_one_out_split(u, i, t)
    seen = seen_items(split)
    negs = sample_negatives(seen, np.arange(0, 60), n_neg=5, seed=3)
    users, cand = build_candidate_matrix(split, negs)

    path = tmp_path / "negatives_seed3.npz"
    save_eval_cache(path, users, cand, seed=3, strategy="uniform")
    u2, c2, meta = load_eval_cache(path)

    assert np.array_equal(users, u2)
    assert np.array_equal(cand, c2)
    assert meta == {"seed": 3, "strategy": "uniform"}
