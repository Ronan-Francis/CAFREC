"""ILD + Coverage for all Pure conditions (offline, no GPU).

test_topk_items are dumped as ORIGINAL KuaiRand video_id tokens (runner.py), so
diversity is a pure offline join to the local video-tag matrix. Reuses the repo's
own cafrec.eval modules. Validates against the reported pure_3seed diversity
(SASRec ILD 0.7930/cov 0.1889; noprof 0.7862/0.2336; bge 0.7920/0.2294) before
extending to the ablation + seq-length conditions. 3-seed means (2 where noted).

Run from cafrec_harness/:  python results/pure_diversity.py
"""
import json, glob, os
import numpy as np
import pandas as pd
from cafrec.eval.diversity import coverage, intra_list_diversity
from cafrec.eval.category_matrix import category_matrix_from_dataframe

RES = "results"
CSV = r"C:/Users/msc/Desktop/G00403092/CAFREC/data/KuaiRand-Pure/data/video_features_basic_pure.csv"
SEEDS = ["2020", "2021", "403092"]

# label -> topk-dump filename glob (per seed). seq20 = reported topk dumps.
CONDS = {
    # reference (reproduce known numbers)
    "SASRec (seq20)":  "SASRec_kuairand_pure_topk_s{s}_*.json",
    "noprof (seq20)":  "CAFREC_kuairand_pure_ctx_noprof_topk_s{s}_*.json",
    "bge (seq20)":     "CAFREC_kuairand_pure_ctx_bge_topk_s{s}_*.json",
    # T3.1 ablations
    "static_gate":     "CAFREC_kuairand_pure_ctx_static_gate_s{s}_*.json",
    "concat":          "CAFREC_kuairand_pure_ctx_concat_s{s}_*.json",
    "standin":         "CAFREC_kuairand_pure_ctx_standin_s{s}_*.json",
    "prof7b":          "CAFREC_kuairand_pure_ctx_prof7b_s{s}_*.json",
    # seq-length sweep
    "SASRec seq10":    "SASRec_kuairand_pure_seq10_s{s}_*.json",
    "bge seq10":       "CAFREC_kuairand_pure_ctx_bge_seq10_s{s}_*.json",
    "noprof seq10":    "CAFREC_kuairand_pure_ctx_noprof_seq10_s{s}_*.json",
    "SASRec seq50":    "SASRec_kuairand_pure_seq50_s{s}_*.json",
    "bge seq50":       "CAFREC_kuairand_pure_ctx_bge_seq50_s{s}_*.json",
    "noprof seq50":    "CAFREC_kuairand_pure_ctx_noprof_seq50_s{s}_*.json",
}


def to_int_rows(topk):
    """topk: list of lists of original video_id tokens (strings). Drop any
    non-numeric token (e.g. a stray [PAD]) so ids can index the category matrix."""
    rows = []
    for r in topk:
        ids = [int(t) for t in r if str(t).lstrip("-").isdigit()]
        rows.append(ids)
    return rows


def main():
    df = pd.read_csv(CSV, usecols=["video_id", "tag"], dtype={"tag": str})
    cat, tag_cols = category_matrix_from_dataframe(df, strategy="multihot")
    print(f"category matrix: {cat.shape[0]} items x {len(tag_cols)} tags\n")

    print(f"{'condition':<16}{'ILD@10':>9}{'Cov@10':>9}{'seeds':>7}")
    for label, pat in CONDS.items():
        ilds, covs, used = [], [], 0
        for s in SEEDS:
            fs = glob.glob(os.path.join(RES, pat.format(s=s)))
            if not fs:
                continue
            d = json.load(open(fs[0]))
            if not d.get("test_topk_items"):
                continue
            rows = to_int_rows(d["test_topk_items"])
            ilds.append(intra_list_diversity(rows, cat))
            covs.append(coverage(rows, d["n_items_catalog"]))
            used += 1
        if used:
            print(f"{label:<16}{np.mean(ilds):>9.4f}{np.mean(covs):>9.4f}{used:>7}")
        if label.startswith("bge (seq20"):
            print("  " + "-" * 30)


if __name__ == "__main__":
    main()
