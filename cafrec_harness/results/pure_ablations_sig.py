"""Pure T3.1 ablations + capacity ladder — 3-seed means and paired significance.

Compares each condition against the reported CAFREC bge-profiler (ablation none,
frozen bge-large profile) by reconstructing per-user NDCG@10 / HIT@10 from the
dumped held-out ranks and running a paired per-user Wilcoxon signed-rank test
plus a 95% bootstrap CI of the mean per-user difference (bge - condition).

Conditions:
  static_gate  context-independent constant gate  -> isolates RQ3 gating
  concat       z := W[z_long ; z_short]            -> gated-vs-concat fusion
  standin      learnable z_long (no frozen profile)-> RQ2 profiler value
  prof7b       7B frozen profile                   -> capacity ladder top

Reads local result JSONs; uses whatever seeds have rank dumps.
"""
import json, glob, math
import numpy as np
from scipy.stats import wilcoxon

RES = "results"
SEEDS = ["2020", "2021", "403092"]
K = 10

REF = "CAFREC_kuairand_pure_ctx_bge_topk_s{seed}_*.json"
CONDS = {
    "static_gate": "CAFREC_kuairand_pure_ctx_static_gate_s{seed}_*.json",
    "concat":      "CAFREC_kuairand_pure_ctx_concat_s{seed}_*.json",
    "standin":     "CAFREC_kuairand_pure_ctx_standin_s{seed}_*.json",
    "prof7b":      "CAFREC_kuairand_pure_ctx_prof7b_s{seed}_*.json",
}


def load(pat, seed):
    hits = glob.glob(f"{RES}/{pat.format(seed=seed)}")
    if not hits:
        return None
    d = json.load(open(hits[0]))
    if not d.get("test_ranks"):
        return None
    return d


def per_user(d):
    """map original uid -> (ndcg@K, hit@K) from 1-indexed target ranks."""
    out = {}
    for u, r in zip(d["test_user_ids"], d["test_ranks"]):
        if r is not None and r <= K:
            out[str(u)] = (1.0 / math.log2(r + 1), 1.0)
        else:
            out[str(u)] = (0.0, 0.0)
    return out


def paired(ref, cond, idx):
    """idx 0=ndcg 1=hit. returns (meanRef, meanCond, meanDelta, ci_lo, ci_hi, p, n)."""
    us = sorted(set(ref) & set(cond))
    a = np.array([ref[u][idx] for u in us])   # reference (bge)
    b = np.array([cond[u][idx] for u in us])  # condition
    delta = a - b                              # bge - condition
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(delta, delta.size, replace=True).mean()
                     for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    try:
        p = wilcoxon(a, b, zero_method="wilcox").pvalue
    except ValueError:      # all-zero differences
        p = 1.0
    return a.mean(), b.mean(), delta.mean(), lo, hi, p, len(us)


def main():
    # 3-seed means (of per-user means) for the table
    print("## 3-seed means (NDCG@10 / HIT@10)\n")
    ref_means = {}
    refs = {s: load(REF, s) for s in SEEDS}
    for label, files in [("bge (ref)", None)] + [(c, CONDS[c]) for c in CONDS]:
        ns, hs = [], []
        for s in SEEDS:
            d = refs[s] if files is None else load(files, s)
            if d is None:
                continue
            pu = per_user(d)
            ns.append(np.mean([v[0] for v in pu.values()]))
            hs.append(np.mean([v[1] for v in pu.values()]))
        seeds_used = len(ns)
        print(f"  {label:<14} NDCG {np.mean(ns):.4f}  HIT {np.mean(hs):.4f}"
              f"  ({seeds_used} seed{'s' if seeds_used != 1 else ''})")

    print("\n## Paired per-user Wilcoxon + 95% bootstrap CI of mean d (bge - condition)\n")
    print(f"{'condition':<12}{'seed':>7}{'metric':>7}"
          f"{'bge':>9}{'cond':>9}{'mean_d':>10}{'95% CI':>22}{'p':>10}{'n':>8}")
    for cond, files in CONDS.items():
        for s in SEEDS:
            ref, cd = refs[s], load(files, s)
            if ref is None or cd is None:
                continue
            pr, pc = per_user(ref), per_user(cd)
            for idx, name in ((0, "NDCG"), (1, "HIT")):
                mr, mc, md, lo, hi, p, n = paired(pr, pc, idx)
                print(f"{cond:<12}{s:>7}{name:>7}{mr:>9.4f}{mc:>9.4f}"
                      f"{md:>+10.5f}{f'[{lo:+.5f},{hi:+.5f}]':>22}{p:>10.2e}{n:>8}")
        print()


if __name__ == "__main__":
    main()
