"""Leave-one-out evaluation protocol (RON-13).

A framework-independent implementation of the standard SASRec-style
leave-one-out protocol:

    * the LAST interaction per user (time-ordered) is the test target,
    * the 2nd-last is the validation target,
    * everything earlier is training history,
    * each target is scored against 1 positive + 99 negatives ("uni100").

The RecBole runner does this internally via `eval_args: {mode: uni100}`, but
that couples evaluation to a full training run and re-samples negatives per
run. This module exists so that:

    1. the protocol can be unit-tested against published SASRec numbers in
       isolation (no RecBole, no GPU),
    2. the 99 negatives are sampled ONCE, with a fixed seed, and CACHED to
       disk so that HRNN, SASRec and CAFREC are all ranked against the exact
       same candidate set (the decision locked in the Cycle-3 design quiz),
    3. cached model scores can be turned into per-user ranks that feed
       straight into `cafrec.eval.metrics.ranks_from_scores`.

Design decisions (Cycle-3 quiz):
    * negatives are sampled UNIFORMLY, excluding each user's seen items, so the
      protocol reproduces the published SASRec HR@10/NDCG@10 reference numbers.
      A popularity-weighted variant is provided as an optional robustness
      protocol only — it will NOT match the published numbers, so keep it
      secondary.
    * candidates are laid out as [positive, neg_1 .. neg_99] with the positive
      in column 0, matching `ranks_from_scores(target_col=0)`.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

POSITIVE_COL = 0  # candidate-matrix convention shared with eval.metrics


# --------------------------------------------------------------------------- #
#  Splitting
# --------------------------------------------------------------------------- #
def leave_one_out_split(user_ids, item_ids, timestamps, min_len=2):
    """Chronological leave-one-out split.

    Parameters
    ----------
    user_ids, item_ids, timestamps : array-like, all the same length
        One row per interaction. `timestamps` only needs to be sortable
        (KuaiRand: use `time_ms`).
    min_len : int
        Users with fewer than this many interactions are dropped — you cannot
        form a held-out test target from a length-1 history. `min_len=2` keeps
        a test target with >=1 training item; `min_len=3` also guarantees a
        distinct validation target.

    Returns
    -------
    dict  user_id -> {"train": np.ndarray[item], "valid": item|None,
                      "test": item}
        Items are time-ordered within "train". "valid" is None when the user
        has exactly `min_len == 2` interactions.
    """
    user_ids = np.asarray(user_ids)
    item_ids = np.asarray(item_ids)
    timestamps = np.asarray(timestamps)

    by_user = defaultdict(list)
    for u, i, t in zip(user_ids, item_ids, timestamps):
        by_user[u].append((t, i))

    split = {}
    for u, events in by_user.items():
        if len(events) < min_len:
            continue
        events.sort(key=lambda ti: ti[0])           # by timestamp, stable
        ordered = [i for _, i in events]
        test = ordered[-1]
        valid = ordered[-2] if len(ordered) >= 3 else None
        train_end = -2 if valid is not None else -1
        split[u] = {
            "train": np.asarray(ordered[:train_end]),
            "valid": valid,
            "test": test,
        }
    return split


def seen_items(split, include_targets=True):
    """user_id -> set of items the user has interacted with.

    With `include_targets=True` (default) the valid/test targets are included
    so they are never drawn as negatives for that user.
    """
    seen = {}
    for u, s in split.items():
        items = set(int(i) for i in s["train"])
        if include_targets:
            if s["valid"] is not None:
                items.add(int(s["valid"]))
            items.add(int(s["test"]))
        seen[u] = items
    return seen


# --------------------------------------------------------------------------- #
#  Negative sampling
# --------------------------------------------------------------------------- #
def sample_negatives(seen_by_user, all_items, n_neg=99, seed=42,
                     strategy="uniform", popularity=None):
    """Sample `n_neg` negatives per user, excluding that user's seen items.

    Deterministic given `seed` and the (sorted) user order, so re-running
    reproduces the identical candidate set — the property that lets every
    model be compared on the same negatives.

    Parameters
    ----------
    seen_by_user : dict  user_id -> set[item_id]
        Typically the output of `seen_items(split)`.
    all_items : array-like
        The full item catalogue (item ids). Need not be contiguous.
    n_neg : int
        Negatives per user (99 for the standard uni100 protocol).
    seed : int
        RNG seed. Fixed -> reproducible -> cacheable.
    strategy : {"uniform", "popularity"}
        "uniform"     : matches published SASRec numbers. Use this as primary.
        "popularity"  : sample proportional to `popularity` (harder negatives);
                        a secondary robustness protocol only.
    popularity : array-like, optional
        Interaction counts aligned to `all_items`. Required for
        strategy="popularity".

    Returns
    -------
    dict  user_id -> np.ndarray[int] of shape (n_neg,)
    """
    all_items = np.asarray(all_items)
    n = all_items.shape[0]
    if n_neg >= n:
        raise ValueError(
            f"n_neg={n_neg} but catalogue has only {n} items; cannot draw "
            f"that many distinct negatives."
        )

    if strategy == "popularity":
        if popularity is None:
            raise ValueError("strategy='popularity' requires `popularity` weights")
        base_w = np.asarray(popularity, dtype=float)
        if base_w.shape[0] != n:
            raise ValueError("`popularity` must be aligned 1:1 with `all_items`")
        if (base_w < 0).any():
            raise ValueError("`popularity` weights must be non-negative")
    elif strategy != "uniform":
        raise ValueError(f"unknown strategy '{strategy}'")

    rng = np.random.default_rng(seed)
    negatives = {}
    for u in sorted(seen_by_user):                    # sorted -> stable order
        seen = seen_by_user[u]
        mask = np.fromiter((int(i) not in seen for i in all_items),
                           dtype=bool, count=n)
        pool = all_items[mask]
        if pool.shape[0] < n_neg:
            raise ValueError(
                f"user {u} has too few unseen items ({pool.shape[0]}) to draw "
                f"{n_neg} negatives."
            )
        if strategy == "uniform":
            chosen = rng.choice(pool, size=n_neg, replace=False)
        else:
            w = base_w[mask]
            total = w.sum()
            if total <= 0:                            # degenerate -> fall back
                chosen = rng.choice(pool, size=n_neg, replace=False)
            else:
                chosen = rng.choice(pool, size=n_neg, replace=False, p=w / total)
        negatives[u] = np.asarray(chosen)
    return negatives


# --------------------------------------------------------------------------- #
#  Candidate matrix (positive + negatives)
# --------------------------------------------------------------------------- #
def build_candidate_matrix(split, negatives, target="test"):
    """Stack [positive, neg_1 .. neg_n] per user into one aligned matrix.

    Returns
    -------
    users : np.ndarray[int]            shape (U,)   sorted user order
    candidates : np.ndarray[int]       shape (U, 1 + n_neg)
        Column 0 (`POSITIVE_COL`) is the held-out positive; the rest are the
        cached negatives. Feed the corresponding model scores to
        `eval.metrics.ranks_from_scores(scores, target_col=0)`.
    """
    users = np.array(sorted(u for u in split if u in negatives))
    rows = []
    for u in users:
        pos = int(split[u][target])
        rows.append(np.concatenate(([pos], negatives[u])))
    candidates = np.vstack(rows).astype(np.int64)
    return users, candidates


# --------------------------------------------------------------------------- #
#  Caching
# --------------------------------------------------------------------------- #
def save_eval_cache(path, users, candidates, seed, strategy="uniform"):
    """Persist the candidate set so every model is scored identically.

    Stored as a compressed .npz alongside the seed/strategy provenance.
    """
    np.savez_compressed(
        path,
        users=np.asarray(users),
        candidates=np.asarray(candidates),
        seed=np.asarray(seed),
        strategy=np.asarray(str(strategy)),
    )
    return path


def load_eval_cache(path):
    """Load a cached candidate set. Returns (users, candidates, meta)."""
    data = np.load(path, allow_pickle=False)
    meta = {"seed": int(data["seed"]), "strategy": str(data["strategy"])}
    return data["users"], data["candidates"], meta
