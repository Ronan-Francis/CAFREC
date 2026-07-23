"""Causal session-context features for the CAFREC gating module.

Given a per-interaction log (one row per click, time-ordered within a user), this
module produces the three "classic" continuous context features:

    session_len      RON-16  position within the current 30-min session, /CAP
    dwell_entropy    RON-17  normalised Shannon entropy of the dwell-ratio
                             histogram over the session so far
    cat_drift        RON-18  fraction of consecutive item pairs (so far) whose
                             primary category changed

Design decisions
----------------
* **Sessions** are cut on a 30-minute inactivity gap within each user, matching
  the T1.1 segmentation decision (PROJECT_LOG).
* **Causality.** The feature attached to interaction i summarises only the rows
  BEFORE i in the same session — i.e. exactly the history the model's short-term
  encoder sees. The first item of every session therefore has all-zero context
  (no history yet). This is what makes the features safe under leave-one-out:
  the held-out last interaction never contributes to its own context (RON-60).
* **dwell_ratio** = clip(play_time_ms / duration_ms, 0, 1). Videos are often
  re-watched (ratio > 1) or have a zero/again missing duration; clipping to
  [0, 1] keeps the entropy histogram well-defined.
* **cat_drift** uses the PRIMARY tag (first token before a comma) so "category
  change" is a single-label comparison. The multi-hot tag encoding is reserved
  for intra-list diversity (see cafrec.eval.category_matrix, risk R2d).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Field names emitted, in x_ctx order. The gating MLP reads these via
# `context_fields` in the model config (a 4th policy feature + the reasoning
# intent label are appended later to reach the planned x_ctx in R^5).
CTX_FIELDS = ["session_len", "dwell_entropy", "cat_drift"]

SESSION_GAP_MS = 30 * 60 * 1000     # 30-minute inactivity gap
SESSION_LEN_CAP = 50                # normalise session position by this (== MAX_ITEM_LIST_LENGTH)
ENTROPY_BINS = 10                   # dwell-ratio histogram resolution (T1.1 used 10)
_LOG2_BINS = np.log2(ENTROPY_BINS)  # normaliser -> entropy in [0, 1]


def dwell_ratio(play_time_ms, duration_ms):
    """Fraction of the video watched, clipped to [0, 1].

    A zero or missing duration yields ratio 0 (undefined watch fraction).
    """
    play = np.asarray(play_time_ms, dtype=float)
    dur = np.asarray(duration_ms, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(dur > 0, play / dur, 0.0)
    return np.clip(ratio, 0.0, 1.0)


def segment_sessions(time_ms, gap_ms=SESSION_GAP_MS):
    """Assign a within-user session id to each row of a time-ordered array.

    A new session starts whenever the gap since the previous interaction exceeds
    `gap_ms`. Returns an int array of the same length, starting at 0.

    Caller must pass a single user's timestamps in ascending order.
    """
    t = np.asarray(time_ms, dtype=np.int64)
    if t.size == 0:
        return np.zeros(0, dtype=np.int64)
    new_session = np.empty(t.size, dtype=bool)
    new_session[0] = False
    new_session[1:] = np.diff(t) > gap_ms
    return np.cumsum(new_session)


def _causal_features_one_session(dwell, category):
    """Causal features for the rows of ONE session (time-ordered).

    For every row i the returned values summarise rows [0, i) only:
        len_i     = i                       (number of prior items)
        entropy_i = H(histogram of dwell[:i])
        drift_i   = changes(category[:i]) / max(i - 1, 1)

    Returns three float arrays of length len(dwell).
    """
    n = len(dwell)
    sess_len = np.zeros(n, dtype=float)
    entropy = np.zeros(n, dtype=float)
    drift = np.zeros(n, dtype=float)

    hist = np.zeros(ENTROPY_BINS, dtype=np.int64)   # running dwell histogram
    seen = 0                                        # rows folded into state
    changes = 0                                     # consecutive category changes so far
    pairs = 0                                       # consecutive pairs so far
    prev_cat = None

    for i in range(n):
        # ---- emit features over the history [0, i) ----
        sess_len[i] = seen
        if seen >= 1:
            p = hist / seen
            nz = p[p > 0]
            entropy[i] = float(-(nz * np.log2(nz)).sum() / _LOG2_BINS) + 0.0  # kill -0.0
        drift[i] = changes / pairs if pairs > 0 else 0.0

        # ---- fold row i into the running state for the NEXT row ----
        b = min(int(dwell[i] * ENTROPY_BINS), ENTROPY_BINS - 1)   # ratio 1.0 -> last bin
        hist[b] += 1
        seen += 1
        cat = category[i]
        if prev_cat is not None:
            pairs += 1
            if cat != prev_cat:
                changes += 1
        prev_cat = cat

    return sess_len, entropy, drift


def compute_context_features(df, cap=SESSION_LEN_CAP):
    """Attach causal session-context features to an interaction frame.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per interaction, with at least:
            user_id     (int)
            time_ms     (int, ms) — used for ordering and session segmentation
            dwell       (float in [0, 1]) — from `dwell_ratio`
            category    (hashable) — primary tag id (or any single-label category)
        The frame need not be pre-sorted; it is sorted by (user_id, time_ms)
        internally and returned in that order.
    cap : int
        `session_len` is min(position, cap) / cap so it lands in [0, 1] on the
        same scale as the other two features.

    Returns
    -------
    pandas.DataFrame
        `df` sorted by (user_id, time_ms) with three added columns
        (`CTX_FIELDS`) plus a `session_id` column, all float except session_id.
    """
    required = {"user_id", "time_ms", "dwell", "category"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"compute_context_features missing columns: {sorted(missing)}")

    out = df.sort_values(["user_id", "time_ms"], kind="mergesort").reset_index(drop=True)

    session_id = np.zeros(len(out), dtype=np.int64)
    sess_len = np.zeros(len(out), dtype=float)
    entropy = np.zeros(len(out), dtype=float)
    drift = np.zeros(len(out), dtype=float)

    dwell_all = out["dwell"].to_numpy(dtype=float)
    cat_all = out["category"].to_numpy()
    time_all = out["time_ms"].to_numpy(dtype=np.int64)

    # Iterate per user over contiguous blocks (rows are grouped after the sort).
    for _, idx in out.groupby("user_id", sort=False).indices.items():
        idx = np.asarray(idx)
        sids = segment_sessions(time_all[idx])
        session_id[idx] = sids
        # within the user, process each session's contiguous rows
        for s in np.unique(sids):
            sel = idx[sids == s]
            sl, en, dr = _causal_features_one_session(dwell_all[sel], cat_all[sel])
            sess_len[sel] = sl
            entropy[sel] = en
            drift[sel] = dr

    out["session_id"] = session_id
    out["session_len"] = np.minimum(sess_len, cap) / float(cap)
    out["dwell_entropy"] = entropy
    out["cat_drift"] = drift
    return out
