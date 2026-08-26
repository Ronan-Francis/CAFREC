"""Stage 4 — 10-seed significance for logfull (Pure).

Combines the 3 original seeds (2020/2021/403092) with 7 new (42/77/123/256/512/
1024/2048) for SASRec, noprof, bge, logfull. Reports 10-seed means, an
across-seed paired test on seed-mean NDCG (logfull vs each), and a per-seed
per-user Wilcoxon significance count. Aggregate NDCG from JSON; per-user from ranks.
"""
import json, glob, math, os
import numpy as np
from scipy.stats import wilcoxon, ttest_rel

RES = "results"
OLD = ["2020", "2021", "403092"]
NEW = ["42", "77", "123", "256", "512", "1024", "2048"]
K = 10

# condition -> (old-seed glob, new-seed glob)
PATS = {
    "SASRec":  ("SASRec_kuairand_pure_topk_s{s}_*.json",
                "SASRec_kuairand_pure_ms_s{s}_*.json"),
    "noprof":  ("CAFREC_kuairand_pure_ctx_noprof_topk_s{s}_*.json",
                "CAFREC_kuairand_pure_ctx_noprof_ms_s{s}_*.json"),
    "bge":     ("CAFREC_kuairand_pure_ctx_bge_topk_s{s}_*.json",
                "CAFREC_kuairand_pure_ctx_bge_ms_s{s}_*.json"),
    "logfull": ("CAFREC_kuairand_pure_ctx_hglogfull_s{s}_*.json",
                "CAFREC_kuairand_pure_ctx_logfull_ms_s{s}_*.json"),
}


def resolve(model, s):
    old_g, new_g = PATS[model]
    g = old_g if s in OLD else new_g
    fs = glob.glob(os.path.join(RES, g.format(s=s)))
    assert len(fs) == 1, (model, s, fs)
    return json.load(open(fs[0]))


def per_user_ndcg(d):
    return {str(u): (1 / math.log2(r + 1) if (r and r <= K) else 0.0)
            for u, r in zip(d["test_user_ids"], d["test_ranks"])}


def main():
    seeds = OLD + NEW
    agg = {m: {} for m in PATS}   # model -> seed -> aggregate ndcg
    pu = {m: {} for m in PATS}    # model -> seed -> per-user dict
    for m in PATS:
        for s in seeds:
            d = resolve(m, s)
            agg[m][s] = d["test"]["ndcg@10"]
            pu[m][s] = per_user_ndcg(d)

    print("## 10-seed mean NDCG@10")
    for m in PATS:
        v = np.array([agg[m][s] for s in seeds])
        print(f"  {m:<9} {v.mean():.4f}  (sd {v.std(ddof=1):.4f}, n={len(v)})")

    print("\n## Across-seed paired test on seed-mean NDCG (X - SASRec), n=10 seeds")
    sas = np.array([agg["SASRec"][s] for s in seeds])
    for m in ("noprof", "bge", "logfull"):
        x = np.array([agg[m][s] for s in seeds])
        d = x - sas
        wp = wilcoxon(x, sas).pvalue
        tp = ttest_rel(x, sas).pvalue
        wins = int((d > 0).sum())
        print(f"  {m:<9} meanΔ {d.mean():+.5f}  wins {wins}/10  "
              f"Wilcoxon p={wp:.2e}  t-test p={tp:.2e}")

    print("\n## Per-seed per-user Wilcoxon: seeds where logfull > SASRec (p<0.05)")
    sig = 0
    for s in seeds:
        us = sorted(pu["logfull"][s])
        a = np.array([pu["logfull"][s][u] for u in us])
        b = np.array([pu["SASRec"][s][u] for u in us])
        p = wilcoxon(a, b).pvalue if np.any(a - b) else 1.0
        hit = p < 0.05 and (a - b).mean() > 0
        sig += hit
        print(f"  seed {s:<7} meanΔ {(a-b).mean():+.5f}  p={p:.2e}  {'SIG' if hit else ''}")
    print(f"\n  logfull > SASRec significant on {sig}/10 seeds (per-user).")


if __name__ == "__main__":
    main()
