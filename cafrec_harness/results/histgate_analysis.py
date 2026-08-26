"""History-gated profiler (RON-45) — does the H2 fix work? (Pure, 3-seed)

Compares CAFREC-bge+history_gate against SASRec, plain bge, and noprof:
  (A) 3-seed overall means,
  (B) paired per-user Wilcoxon (histgate - other) overall,
  (C) sparse/dense cohort means (does history-gating recover the sparse loss the
      plain profiler caused, while keeping the dense gain?).
Cohort = history length <20 (sparse) / >=20 (dense). Per-user metrics from dumped
ranks (== RecBole aggregate). Script mirrors h2_sparse_strata / pure_ablations_sig.
"""
import json, glob, math, os
from collections import Counter
import numpy as np
from scipy.stats import wilcoxon

RES = "results"
INTER = r"C:/Users/msc/Desktop/G00403092/CAFREC/data/recbole/kuairand_pure_ctx/kuairand_pure_ctx.inter"
SEEDS = ["2020", "2021", "403092"]
K = 10

PATS = {
    "SASRec":   "SASRec_kuairand_pure_topk_s{s}_*.json",
    "noprof":   "CAFREC_kuairand_pure_ctx_noprof_topk_s{s}_*.json",
    "bge":      "CAFREC_kuairand_pure_ctx_bge_topk_s{s}_*.json",
    "histgate": "CAFREC_kuairand_pure_ctx_histgate_s{s}_*.json",
}


def per_user(d):
    nd, ht = {}, {}
    for u, r in zip(d["test_user_ids"], d["test_ranks"]):
        u = str(u)
        good = r is not None and r <= K
        nd[u] = (1.0 / math.log2(r + 1)) if good else 0.0
        ht[u] = 1.0 if good else 0.0
    return nd, ht


def load(model, s):
    return per_user(json.load(open(glob.glob(os.path.join(RES, PATS[model].format(s=s)))[0])))


def history_counts():
    c = Counter()
    with open(INTER) as f:
        iu = {h.split(":")[0]: i for i, h in enumerate(f.readline().rstrip("\n").split("\t"))}["user_id"]
        for line in f:
            c[line.split("\t", iu + 1)[iu]] += 1
    return c


def main():
    cohort = {u: ("Sparse" if n < 20 else "Dense") for u, n in history_counts().items()}
    print("cohort:", dict(Counter(cohort.values())), "\n")

    M = {m: {s: load(m, s) for s in SEEDS} for m in PATS}

    # (A) overall means
    print("## Overall 3-seed means")
    for m in PATS:
        nd = np.mean([np.mean(list(M[m][s][0].values())) for s in SEEDS])
        ht = np.mean([np.mean(list(M[m][s][1].values())) for s in SEEDS])
        print(f"  {m:<9} NDCG {nd:.4f}  HR {ht:.4f}")

    # (B) paired histgate vs others, overall
    print("\n## Paired per-user Wilcoxon (histgate - X), overall NDCG, per seed")
    for other in ("SASRec", "bge", "noprof"):
        cells = []
        for s in SEEDS:
            us = sorted(M["histgate"][s][0])
            a = np.array([M["histgate"][s][0][u] for u in us])
            b = np.array([M[other][s][0][u] for u in us])
            p = wilcoxon(a, b).pvalue if np.any(a - b) else 1.0
            cells.append(f"d{(a-b).mean():+.5f}(p={p:.1e})")
        print(f"  vs {other:<7}: " + "  ".join(cells))

    # (C) cohort means + sparse recovery test
    print("\n## Sparse/Dense NDCG (3-seed mean)")
    print(f"  {'model':<9}{'Sparse':>9}{'Dense':>9}")
    for m in PATS:
        def strat(coh):
            vals = []
            for s in SEEDS:
                nd = M[m][s][0]
                vals.append(np.mean([nd[u] for u in nd if cohort.get(u) == coh]))
            return np.mean(vals)
        print(f"  {m:<9}{strat('Sparse'):>9.4f}{strat('Dense'):>9.4f}")

    print("\n## H2 recovery: (histgate - X) within cohort, NDCG, per seed")
    for coh in ("Sparse", "Dense"):
        for other in ("bge", "noprof", "SASRec"):
            cells = []
            for s in SEEDS:
                us = [u for u in M["histgate"][s][0] if cohort.get(u) == coh]
                a = np.array([M["histgate"][s][0][u] for u in us])
                b = np.array([M[other][s][0][u] for u in us])
                p = wilcoxon(a, b).pvalue if np.any(a - b) else 1.0
                cells.append(f"{(a-b).mean():+.5f}(p{p:.0e})")
            print(f"  {coh:<7} vs {other:<7}: " + " ".join(cells))
        print()


if __name__ == "__main__":
    main()
