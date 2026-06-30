import numpy as np


def per_user_metrics(ranks, k=10):
    """ranks: 1-indexed position of the held-out item per user.
    Returns per-user metric arrays — these ARE the paired values for testing."""
    ranks = np.asarray(ranks, dtype=float)
    hit = (ranks <= k).astype(float)
    ndcg = np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0)
    rr = 1.0 / ranks
    return {f"HR@{k}": hit, f"NDCG@{k}": ndcg, "MRR": rr}


def aggregate(per_user):
    """Mean over users -> the headline numbers."""
    return {m: float(v.mean()) for m, v in per_user.items()}


def ranks_from_scores(scores, target_col=0):
    """scores: [n_users, n_candidates] with the positive in `target_col`
    (under uni100, lay out candidates as [pos, neg_1..neg_99]).
    Rank = 1 + (#candidates scoring strictly higher than the positive)."""
    scores = np.asarray(scores)
    target = scores[:, target_col][:, None]
    return 1 + (scores > target).sum(axis=1)