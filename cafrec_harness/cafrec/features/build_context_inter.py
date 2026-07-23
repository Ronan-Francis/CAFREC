"""Build a feature-augmented RecBole atomic file for CAFREC (RON-16/17/18).

The plain `<dataset>.inter` carries only (user_id, item_id, timestamp) — enough
for the baselines. CAFREC's gating module additionally needs the causal
session-context features, which depend on `play_time_ms` / `duration_ms` and the
video `tag` — columns that live only in the raw KuaiRand logs.

This module reproduces the EXACT filtering of the dataset builder
(`CAFREC_Dataset_Builder_KuaiRand_inter.ipynb`) so the augmented file has the
same interaction rows, then appends three typed float columns:

    session_len:float  dwell_entropy:float  cat_drift:float

Output goes to a sibling `<dataset>_ctx` dataset directory so the baselines keep
using the clean 3-column file and CAFREC points at the `_ctx` variant. Same
rows -> the comparison stays fair; the only difference is the extra columns
CAFREC reads via `load_col`.

CLI
---
    python -m cafrec.features.build_context_inter --tier light   # kuairand_pure
    python -m cafrec.features.build_context_inter --tier medium  # kuairand_1k

`light` fits in memory; `medium`/`heavy` stream the raw logs in chunks.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cafrec.features.context import CTX_FIELDS, compute_context_features, dwell_ratio

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

INTER_HEADER = ["user_id:token", "item_id:token", "timestamp:float",
                "session_len:float", "dwell_entropy:float", "cat_drift:float"]


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


def _load_filtered(paths):
    """Stream raw logs; keep organic (is_rand==0) clicks (is_click==1).

    Returns a frame with user_id, video_id, time_ms, dwell (in [0,1]).
    """
    kept = []
    for p in paths:
        for chunk in pd.read_csv(p, usecols=USE_COLS, dtype=DTYPES, chunksize=CHUNKSIZE):
            chunk = chunk[(chunk["is_rand"] == 0) & (chunk["is_click"] == 1)]
            if len(chunk) == 0:
                continue
            chunk = chunk.copy()
            chunk["dwell"] = dwell_ratio(chunk["play_time_ms"], chunk["duration_ms"])
            kept.append(chunk[["user_id", "video_id", "time_ms", "dwell"]])
    if not kept:
        return pd.DataFrame(columns=["user_id", "video_id", "time_ms", "dwell"])
    return pd.concat(kept, ignore_index=True)


def _k_core_user(df, min_user):
    """Single-pass user floor (matches the builder for light/medium tiers)."""
    uc = df["user_id"].value_counts()
    return df[df["user_id"].isin(uc[uc >= min_user].index)].reset_index(drop=True)


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


def build(tier: str, write: bool = True):
    meta = VERSIONS[tier]
    raw = DATA_DIR / meta["variant"] / "data"
    t0 = time.perf_counter()

    df = _load_filtered(_find_logs(raw, meta["suffix"]))
    df = _k_core_user(df, MIN_USER_INTER)

    tag_map = _primary_tag_map(raw, meta["suffix"])
    df["category"] = df["video_id"].map(tag_map).fillna(-1).astype(np.int64)

    feats = compute_context_features(df)   # sorts by (user_id, time_ms)

    out = feats.rename(columns={"video_id": "item_id", "time_ms": "timestamp"})
    out = out[["user_id", "item_id", "timestamp", *CTX_FIELDS]]

    stats = {
        "tier": tier, "dataset": f"{meta['dataset']}_ctx",
        "users": int(out["user_id"].nunique()),
        "items": int(out["item_id"].nunique()),
        "interactions": int(len(out)),
        "sessions": int(feats["session_id"].groupby(feats["user_id"]).nunique().sum()),
        "mean_session_len": round(float(feats["session_len"].mean()) * 50, 2),
        "mean_dwell_entropy": round(float(feats["dwell_entropy"].mean()), 4),
        "mean_cat_drift": round(float(feats["cat_drift"].mean()), 4),
        "sec": round(time.perf_counter() - t0, 1),
    }

    if write and len(out):
        out_dir = OUTPUT_ROOT / f"{meta['dataset']}_ctx"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{meta['dataset']}_ctx.inter"
        out.to_csv(out_path, sep="\t", index=False, header=INTER_HEADER)
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
