"""Sequence-length sensitivity — 3-seed means + paired significance (Pure).

SASRec vs CAFREC (bge / noprof) at short-term window seq_len in {10, 20, 50}.
seq_len 20 reuses the reported topk dumps. Per-user NDCG@10/HIT@10 from dumped
ranks (== RecBole aggregate); paired per-user Wilcoxon of (CAFREC - SASRec) per
seed. Question: does CAFREC's edge over the strong short-term baseline depend on
how much short-term history the encoder is given?
"""
import json, glob, math, os
import numpy as np
from scipy.stats import wilcoxon

RES = "results"
SEEDS = ["2020", "2021", "403092"]
K = 10

# per seq_len: model -> filename glob. seq20 = the reported topk dumps.
PATS = {
    10: {"SASRec": "SASRec_kuairand_pure_seq10_s{s}_*.json",
         "bge":    "CAFREC_kuairand_pure_ctx_bge_seq10_s{s}_*.json",
         "noprof": "CAFREC_kuairand_pure_ctx_noprof_seq10_s{s}_*.json"},
    20: {"SASRec": "SASRec_kuairand_pure_topk_s{s}_*.json",
         "bge":    "CAFREC_kuairand_pure_ctx_bge_topk_s{s}_*.json",
         "noprof": "CAFREC_kuairand_pure_ctx_noprof_topk_s{s}_*.json"},
    50: {"SASRec": "SASRec_kuairand_pure_seq50_s{s}_*.json",
         "bge":    "CAFREC_kuairand_pure_ctx_bge_seq50_s{s}_*.json",
         "noprof": "CAFREC_kuairand_pure_ctx_noprof_seq50_s{s}_*.json"},
}


def per_user(d):
    nd, ht = {}, {}
    for u, r in zip(d["test_user_ids"], d["test_ranks"]):
        u = str(u)
        if r is not None and r <= K:
            nd[u] = 1.0 / math.log2(r + 1); ht[u] = 1.0
        else:
            nd[u] = ht[u] = 0.0
    return nd, ht


def load(pat, s):
    fs = glob.glob(os.path.join(RES, pat.format(s=s)))
    assert len(fs) == 1, (pat, s, fs)
    return per_user(json.load(open(fs[0])))


def main():
    print(f"{'seq_len':>7}{'model':>9}{'NDCG':>9}{'HIT':>9}"
          f"{'dNDCG_vs_SAS':>14}{'wilcoxon_p(per seed)':>34}")
    for L in (10, 20, 50):
        sas = {s: load(PATS[L]["SASRec"], s) for s in SEEDS}
        for m in ("SASRec", "bge", "noprof"):
            cond = {s: load(PATS[L][m], s) for s in SEEDS}
            nd = np.mean([np.mean(list(cond[s][0].values())) for s in SEEDS])
            ht = np.mean([np.mean(list(cond[s][1].values())) for s in SEEDS])
            if m == "SASRec":
                print(f"{L:>7}{m:>9}{nd:>9.4f}{ht:>9.4f}{'—':>14}{'':>34}")
                continue
            deltas, ps = [], []
            for s in SEEDS:
                us = sorted(cond[s][0])
                a = np.array([cond[s][0][u] for u in us])
                b = np.array([sas[s][0][u] for u in us])
                deltas.append((a - b).mean())
                try:
                    ps.append(wilcoxon(a, b).pvalue)
                except ValueError:
                    ps.append(1.0)
            pstr = " ".join(f"{p:.1e}" for p in ps)
            print(f"{L:>7}{m:>9}{nd:>9.4f}{ht:>9.4f}{np.mean(deltas):>+14.5f}   {pstr:>28}")
        print()


if __name__ == "__main__":
    main()
