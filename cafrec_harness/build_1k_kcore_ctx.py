"""Build kuairand_1k_kcore_ctx.inter — context features on the k-core-floored 1K.

Reuses build_context_inter's pipeline (organic-click targets, causal six-feature
context, train-only z-scoring) but adds the iterative item+user k-core floor
(item>=10, user>=5) so the tier is non-degenerate, matching build_1k_kcore.py.
Output feeds CAFREC (logfull) on the 1K generalisation test; SASRec is run on the
same dataset for a matched comparison.
"""
import json
import numpy as np
from cafrec.features import build_context_inter as B

MIN_ITEM, MIN_USER = 10, 5
OUT_DIR = B.OUTPUT_ROOT / "kuairand_1k_kcore_ctx"
OUT = OUT_DIR / "kuairand_1k_kcore_ctx.inter"


def kcore(df, ucol="user_id", icol="video_id"):
    while True:
        n0 = len(df)
        ic = df[icol].value_counts()
        df = df[df[icol].isin(ic[ic >= MIN_ITEM].index)]
        uc = df[ucol].value_counts()
        df = df[df[ucol].isin(uc[uc >= MIN_USER].index)]
        if len(df) == n0:
            return df


def main():
    raw = B.DATA_DIR / B.VERSIONS["medium"]["variant"] / "data"
    full = B._load_full(B._find_logs(raw, "1k"))
    tag_map = B._primary_tag_map(raw, "1k")
    full["category"] = full["video_id"].map(tag_map).fillna(-1).astype(np.int64)
    feats = B.compute_context_features(full)

    tgt = feats[(feats["is_rand"] == 0) & (feats["is_click"] == 1)].copy()
    tgt = kcore(tgt)                                  # <-- the item floor
    tgt = tgt.sort_values(["user_id", "time_ms"], kind="mergesort").reset_index(drop=True)
    print(f"after k-core: rows={len(tgt):,} users={tgt.user_id.nunique()} "
          f"items={tgt.video_id.nunique()}")

    train_mask = B._loo_split_mask(tgt)
    tgt, scaler = B.standardize_context(tgt, train_mask)
    out = tgt.rename(columns={"video_id": "item_id", "time_ms": "timestamp"})
    out = out[["user_id", "item_id", "timestamp", *B.CTX_FIELDS]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, sep="\t", index=False, header=B.INTER_HEADER)
    json.dump(scaler, open(OUT_DIR / "ctx_scaler_params.json", "w"), indent=2)
    print(f"wrote {OUT} ({len(out):,} rows, {out.item_id.nunique()} items)")


if __name__ == "__main__":
    main()
