"""Build a feature-augmented RecBole atomic file for CAFREC (RON-16/17/18/19).

The plain `<dataset>.inter` carries only (user_id, item_id, timestamp) — enough
for the baselines. CAFREC's gating module additionally needs the six causal
session-context features (see `cafrec.features.context`), which depend on
`play_time_ms` / `duration_ms`, the video `tag`, and `is_rand` — columns that
live only in the raw KuaiRand logs.

D1 contract — full-log context, organic-click targets
-----------------------------------------------------
The six features are computed over the FULL impression log (organic AND
random-policy rows, clicked or not) so `prefix_policy_flag` (the is_rand ->
search_to_rec structural replacement) is non-degenerate. Only ORGANIC CLICKS
(`is_rand==0 & is_click==1`) are then emitted as interaction rows — exactly the
row-set the baselines' `<dataset>.inter` contains, so the comparison stays fair.
Each emitted row carries its six prefix-context columns.

The four continuous features are z-scored with statistics fit on the TRAIN rows
only (last-two-per-user held out, matching RecBole's leave-one-out split), and
the transform is persisted to `ctx_scaler_params.json` beside the atomic file.

Output goes to a sibling `<dataset>_ctx` directory; baselines keep using the
clean 3-column file and CAFREC points at the `_ctx` variant.

CLI
---
    python -m cafrec.features.build_context_inter --tier light   # kuairand_pure
    python -m cafrec.features.build_context_inter --tier medium  # kuairand_1k

`light` fits in memory; `medium`/`heavy` stream the raw logs in chunks on read.
NOTE: the feature pass itself holds the full per-user impression log in memory
(a pandas groupby); the `heavy` (27K) tier will need an out-of-core pass — out
of scope for local runs (Modal-blocked).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cafrec.features.context import (
    CONTINUOUS,
    CTX_FIELDS,
    compute_context_features,
    dwell_ratio,
    standardize_context,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT.parent / "data"
OUTPUT_ROOT = DATA_DIR / "recbole"

# Mirror the dataset builder's contract exactly (organic clicks, k-core=5).
VERSIONS = {
    "light":  {"variant": "KuaiRand-Pure", "suffix": "pure", "dataset": "kuairand_pure"},
    "medium": {"variant": "KuaiRand-1K",   "suffix": "1k",   "dataset": "kuairand_1k"},
    "heavy":  {"variant": "KuaiRand-27K",  "suffix": "27k",  "dataset": "kuairand_27k"},
}
LOG_BASES = ["log_standard_4_08_to_4_21",
             "log_standard_4_22_to_5_08",
             "log_random_4_22_to_5_08"]
MIN_USER_INTER = 5
CHUNKSIZE = 5_000_000
USE_COLS = ["user_id", "video_id", "time_ms", "is_rand", "is_click",
            "play_time_ms", "duration_ms"]
DTYPES = {"user_id": "int32", "video_id": "int32", "time_ms": "int64",
          "is_rand": "int8", "is_click": "int8",
          "play_time_ms": "int64", "duration_ms": "int64"}

# 3 base columns + the six context features (binary flags as :float so RecBole
# loads them as numeric fields, consistent with how CAFREC reads interaction[f]).
INTER_HEADER = (["user_id:token", "item_id:token", "timestamp:float"]
                + [f"{f}:float" for f in CTX_FIELDS])


def _find_logs(raw: Path, suffix: str):
    """All raw log parts for a tier (exact suffix, else split-part glob)."""
    paths = []
    for base in LOG_BASES:
        exact = raw / f"{base}_{suffix}.csv"
        if exact.exists():
            paths.append(exact)
            continue
        parts = sorted(raw.glob(f"{base}_{suffix}_part*.csv"))
        if parts:
            paths.extend(parts)
        else:
            hits = sorted(raw.glob(f"{base}_*.csv"))
            if hits:
                paths.append(hits[0])
    return paths


def _load_full(paths):
    """Stream the raw logs and keep EVERY impression (organic + random, clicked
    or not) — the full behavioural context. Returns a frame with user_id,
    video_id, time_ms, is_rand, is_click, dwell (in [0,1]).
    """
    kept = []
    for p in paths:
        for chunk in pd.read_csv(p, usecols=USE_COLS, dtype=DTYPES, chunksize=CHUNKSIZE):
            chunk = chunk.copy()
            chunk["dwell"] = dwell_ratio(chunk["play_time_ms"], chunk["duration_ms"])
            kept.append(chunk[["user_id", "video_id", "time_ms",
                               "is_rand", "is_click", "dwell"]])
    cols = ["user_id", "video_id", "time_ms", "is_rand", "is_click", "dwell"]
    if not kept:
        return pd.DataFrame(columns=cols)
    return pd.concat(kept, ignore_index=True)


def _primary_tag_map(raw: Path, suffix: str):
    """video_id -> primary tag id (first token before a comma; -1 if null)."""
    vf = raw / f"video_features_basic_{suffix}.csv"
    parts = sorted(raw.glob(f"video_features_basic_{suffix}_part*.csv")) if not vf.exists() else [vf]
    frames = [pd.read_csv(p, usecols=["video_id", "tag"]) for p in parts]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("video_id")

    def primary(v):
        if pd.isna(v):
            return -1
        tok = str(v).split(",")[0].strip()
        return int(tok) if tok.isdigit() else -1

    df["category"] = df["tag"].map(primary)
    return dict(zip(df["video_id"].to_numpy(), df["category"].to_numpy()))


def _loo_split_mask(frame):
    """Boolean train mask for a per-user time-ordered frame: everything except
    each user's last two interactions (RecBole LS:valid_and_test, order TO).
    Row order within the frame must already be (user_id, timestamp) ascending.
    """
    rank_desc = frame.groupby("user_id").cumcount(ascending=False)  # 0 = last
    return (rank_desc >= 2).to_numpy()


def build(tier: str, write: bool = True):
    meta = VERSIONS[tier]
    raw = DATA_DIR / meta["variant"] / "data"
    t0 = time.perf_counter()

    # 1) full impression log -> categories -> six causal features (over ALL rows)
    full = _load_full(_find_logs(raw, meta["suffix"]))
    tag_map = _primary_tag_map(raw, meta["suffix"])
    full["category"] = full["video_id"].map(tag_map).fillna(-1).astype(np.int64)
    feats = compute_context_features(full)   # sorts by (user_id, time_ms)

    # 2) organic-click targets only (same row-set as the baseline .inter), with
    #    the k-core user floor applied to the TARGET count.
    tgt = feats[(feats["is_rand"] == 0) & (feats["is_click"] == 1)].copy()
    uc = tgt["user_id"].value_counts()
    tgt = tgt[tgt["user_id"].isin(uc[uc >= MIN_USER_INTER].index)]
    tgt = tgt.sort_values(["user_id", "time_ms"], kind="mergesort").reset_index(drop=True)

    # 3) z-score the continuous features on TRAIN rows only; persist the scaler.
    train_mask = _loo_split_mask(tgt)
    tgt, scaler_params = standardize_context(tgt, train_mask)

    out = tgt.rename(columns={"video_id": "item_id", "time_ms": "timestamp"})
    out = out[["user_id", "item_id", "timestamp", *CTX_FIELDS]]

    stats = {
        "tier": tier, "dataset": f"{meta['dataset']}_ctx",
        "users": int(out["user_id"].nunique()),
        "items": int(out["item_id"].nunique()),
        "interactions": int(len(out)),
        "context_rows_total": int(len(feats)),
        "train_rows": int(train_mask.sum()),
        "sessions": int(feats["session_id"].groupby(feats["user_id"]).nunique().sum()),
        "policy_flag_rate": round(float(out["prefix_policy_flag"].mean()), 4),
        "first_session_rate": round(float(out["is_first_session"].mean()), 4),
        "sec": round(time.perf_counter() - t0, 1),
    }

    if write and len(out):
        out_dir = OUTPUT_ROOT / f"{meta['dataset']}_ctx"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{meta['dataset']}_ctx.inter"
        out.to_csv(out_path, sep="\t", index=False, header=INTER_HEADER)
        with open(out_dir / "ctx_scaler_params.json", "w") as f:
            json.dump(scaler_params, f, indent=2)
        stats["path"] = str(out_path)
        stats["size_mb"] = round(out_path.stat().st_size / 1e6, 1)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", choices=list(VERSIONS), default="light")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    s = build(args.tier, write=not args.no_write)
    print("\n--- context-feature build ---")
    for k, v in s.items():
        print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
