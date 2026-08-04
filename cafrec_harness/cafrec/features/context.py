"""Causal session-context features for the CAFREC gating module (x_ctx).

This module is the harness-level, tested reproduction of the T1.1 notebook's
prefix-causal feature pipeline (CAFREC_T1_1_KuaiRand_Preparation.ipynb, cells
27/28/30). It produces the canonical **six-feature** context vector:

    prefix_session_len_log_z   position in the current 30-min session (log, z)
    prefix_dwell_entropy_z     base-2 Shannon entropy of the dwell-ratio hist (z)
    prefix_category_drift_z    fraction of consecutive category changes (z)
    inter_session_gap_log_z    log1p(ms since the previous session) (z)  [recency]
    prefix_policy_flag         did an is_rand transition occur in the prefix? (0/1)
    is_first_session           user's first session? (cold-start flag, 0/1)

Design decisions
----------------
* **Full-log context, organic targets (D1).** Features are computed over the
  FULL behavioural log — organic (is_rand==0) AND random-policy (is_rand==1)
  rows kept as context — so `prefix_policy_flag` (the is_rand -> search_to_rec
  structural replacement, central to RQ3) is non-degenerate. The caller then
  emits only organic-click rows as training/eval targets; random rows never
  become a held-out target. Compute here, filter downstream.
* **Sessions** are cut on a 30-minute inactivity gap within each user (T1.1).
* **Causality.** Every feature attached to interaction i summarises only rows
  STRICTLY BEFORE i in the same session (expanding window, shifted by one). The
  first item of a session therefore has all-zero prefix features. This is what
  makes the features safe under leave-one-out: the held-out last interaction
  never contributes to its own context (RON-60).
* **Standardisation.** The four continuous features are z-scored with
  statistics fit on the TRAIN rows only (`standardize_context`), mirroring the
  notebook so no eval/test information leaks into the scaler. The two binary
  flags pass through unscaled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Continuous features (z-scored) and binary flags (passthrough), in x_ctx order.
CONTINUOUS = ["prefix_session_len_log", "prefix_dwell_entropy",
              "prefix_category_drift", "inter_session_gap_log"]
BINARY = ["prefix_policy_flag", "is_first_session"]

# Final field names the gating MLP reads (via `context_fields` in the model
# config), in x_ctx order. len(CTX_FIELDS) == n_context_features == 6.
CTX_FIELDS = [c + "_z" for c in CONTINUOUS] + BINARY

SESSION_GAP_MS = 30 * 60 * 1000     # 30-minute inactivity gap
ENTROPY_BINS = 10                   # dwell-ratio histogram resolution (T1.1 used 10)


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


def _assign_session_ids(out, user_col="user_id", time_col="time_ms"):
    """Per-user 30-min session id for a frame already sorted by (user, time)."""
    session_id = np.zeros(len(out), dtype=np.int64)
    time_all = out[time_col].to_numpy(dtype=np.int64)
    for _, idx in out.groupby(user_col, sort=False).indices.items():
        idx = np.asarray(idx)
        session_id[idx] = segment_sessions(time_all[idx])
    return session_id


def _prefix_expanding(series, keys):
    """Expanding cumulative sum within `keys`, shifted by one (prefix only).

    Row t receives the sum over rows strictly before t in its group; the first
    row of every group gets 0. This is the leakage-safe prefix aggregator used
    for every causal feature below.
    """
    g = series.groupby(keys, sort=False)
    return g.cumsum().groupby(keys, sort=False).shift(1).fillna(0.0)


def compute_context_features(df):
    """Attach the six raw (pre-standardisation) context features to a frame.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per interaction (organic AND random rows — the full log), with:
            user_id   (int)
            time_ms   (int, ms) — ordering + 30-min session segmentation
            dwell     (float in [0, 1]) — from `dwell_ratio`
            category  (hashable) — primary tag id (single-label category)
            is_rand   (int, 0/1) — standard vs random policy (for policy_flag)
        Not required to be pre-sorted; sorted by (user_id, time_ms) internally.

    Returns
    -------
    pandas.DataFrame
        `df` sorted by (user_id, time_ms) with `session_id` plus the raw feature
        columns `CONTINUOUS + BINARY`. The z-scored `CTX_FIELDS` are produced
        separately by `standardize_context` (train-fit).
    """
    required = {"user_id", "time_ms", "dwell", "category", "is_rand"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"compute_context_features missing columns: {sorted(missing)}")

    out = df.sort_values(["user_id", "time_ms"], kind="mergesort").reset_index(drop=True)
    out["session_id"] = _assign_session_ids(out)
    keys = [out["user_id"], out["session_id"]]
    grp = out.groupby(["user_id", "session_id"], sort=False)

    # (1) prefix_session_len: items seen BEFORE row t in its session (0-indexed).
    out["prefix_session_len"] = grp.cumcount().astype("int64")

    # (2) prefix_dwell_entropy: base-2 Shannon entropy of the dwell-bin
    #     histogram over the prefix. One-hot -> expanding cumsum -> shift(1).
    dwell_bin = pd.cut(out["dwell"].clip(lower=0, upper=1), bins=ENTROPY_BINS,
                       labels=False, include_lowest=True).to_numpy()
    onehot = np.zeros((len(out), ENTROPY_BINS), dtype="float32")
    ok = ~np.isnan(dwell_bin)
    onehot[ok, dwell_bin[ok].astype(int)] = 1.0
    onehot = pd.DataFrame(onehot, index=out.index,
                          columns=[f"b_{i}" for i in range(ENTROPY_BINS)])
    prefix_counts = (onehot.groupby(keys, sort=False).cumsum()
                           .groupby(keys, sort=False).shift(1).fillna(0.0))
    tot = prefix_counts.sum(axis=1).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        p = prefix_counts.to_numpy() / tot[:, None]
        ent = -np.sum(p * np.where(p > 0, np.log2(p), 0.0), axis=1)
    ent[tot == 0] = 0.0
    out["prefix_dwell_entropy"] = np.abs(ent).astype("float32")   # abs() kills -0.0

    # (3) prefix_category_drift: category changes in the prefix / prefix pairs.
    #     notna guards on BOTH current and previous avoid phantom drift on nulls.
    cat_prev = grp["category"].shift(1)
    drift_ev = ((out["category"] != cat_prev)
                & cat_prev.notna() & out["category"].notna()).astype("float32")
    prefix_drift = _prefix_expanding(drift_ev, keys)
    denom = (out["prefix_session_len"] - 1).clip(lower=1)
    out["prefix_category_drift"] = (prefix_drift / denom).astype("float32")

    # (4) prefix_policy_flag: did an is_rand transition occur anywhere in the
    #     prefix? policy_transition = is_rand differs from the previous in-session
    #     row (the standard<->random surface discontinuity, T1.1 record).
    is_rand_prev = grp["is_rand"].shift(1)
    policy_transition = ((out["is_rand"] != is_rand_prev)
                         & is_rand_prev.notna()).astype("float32")
    prefix_pol = _prefix_expanding(policy_transition, keys)
    out["prefix_policy_flag"] = (prefix_pol > 0).astype("int8")

    # (5) inter_session_gap_log + is_first_session: return recency, known at
    #     session start (prefix-safe). First session per user has no predecessor
    #     -> gap imputed with the median log-gap and flagged.
    sb = (out.groupby(["user_id", "session_id"], sort=False)["time_ms"]
             .agg(sess_start="first", sess_end="last").reset_index())
    sb["prev_end"] = sb.groupby("user_id")["sess_end"].shift(1)
    sb["gap_ms"] = sb["sess_start"] - sb["prev_end"]
    gap_log = np.log1p(sb["gap_ms"].clip(lower=0))
    median = gap_log.median()
    if not np.isfinite(median):   # no inter-session gaps at all (all first sessions)
        median = 0.0
    sb["inter_session_gap_log"] = gap_log.fillna(median).astype("float32")
    sb["is_first_session"] = sb["gap_ms"].isna().astype("int8")
    out = out.merge(sb[["user_id", "session_id", "inter_session_gap_log",
                        "is_first_session"]],
                    on=["user_id", "session_id"], how="left")

    # (6) log-count (idempotent; the raw count column is left untouched).
    out["prefix_session_len_log"] = np.log1p(out["prefix_session_len"]).astype("float32")
    return out


def standardize_context(df, train_mask, continuous=CONTINUOUS):
    """Z-score the continuous features using TRAIN-only statistics.

    Adds a `<feat>_z` column for each continuous feature and returns
    `(df, scaler_params)`. `scaler_params` is JSON-serialisable and records the
    per-feature mean/std and the number of train rows the scaler was fit on, so
    the exact transform can be persisted (ctx_scaler_params.json) and reused.

    A zero std (constant feature on the train split) is replaced by 1.0 so the
    z column is well-defined (all zeros) rather than NaN/inf.
    """
    train_mask = np.asarray(train_mask, dtype=bool)
    mu = df.loc[train_mask, continuous].mean()
    sd = df.loc[train_mask, continuous].std(ddof=0).replace(0, 1.0)
    for c in continuous:
        df[c + "_z"] = ((df[c] - mu[c]) / sd[c]).astype("float32")
    params = {
        "features": list(continuous),
        "mean": [float(mu[c]) for c in continuous],
        "std": [float(sd[c]) for c in continuous],
        "fit_rows": int(train_mask.sum()),
    }
    return df, params
