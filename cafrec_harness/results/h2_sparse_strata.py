"""H2 test — does the LLM profiler help SPARSE-history users specifically? (Pure)

H2 (thesis): offline LLM temporal profiling most benefits users with short
interaction histories, whose short-term encoder has little to work with. The
cleanest on/off contrast is CAFREC_bge (frozen profile) vs CAFREC_noprof
(z_long forced to 0, same architecture). If H2 holds, the per-user profiler
gain (bge - noprof) should be LARGER for the sparse cohort than the dense one.

Cohort = total interaction count per user in the atomic file (history length),
split sparse (<20) vs dense (>=20), matching the T1.1 cohort threshold. Per-user
NDCG@10 / HIT@10 reconstructed from dumped ranks (asserted == RecBole aggregate),
3-seed means; paired per-user Wilcoxon of (bge - noprof) within each cohort.
"""
import json, glob, math, os
from collections import defaultdict, Counter
import numpy as np
from scipy.stats import wilcoxon

RES = r"C:/Users/msc/Desktop/G00403092/CAFREC/cafrec_harness/results"
INTER = r"C:/Users/msc/Desktop/G00403092/CAFREC/data/recbole/kuairand_pure_ctx/kuairand_pure_ctx.inter"
SEEDS = ["2020", "2021", "403092"]
K = 10
THRESH = 20

MODELS = {
    "SASRec":        "SASRec_kuairand_pure_topk_s{seed}_*.json",
    "CAFREC_noprof": "CAFREC_kuairand_pure_ctx_noprof_topk_s{seed}_*.json",
    "CAFREC_bge":    "CAFREC_kuairand_pure_ctx_bge_topk_s{seed}_*.json",
}


def per_user(ranks, uids):
    nd, ht = {}, {}
    for u, r in zip(uids, ranks):
        u = str(u)
        if r is not None and r <= K:
            nd[u] = 1.0 / math.log2(r + 1); ht[u] = 1.0
        else:
            nd[u] = 0.0; ht[u] = 0.0
    return nd, ht


def history_counts():
    """original user_id -> number of rows (interactions) in the atomic file."""
    c = Counter()
    with open(INTER) as f:
        header = f.readline().rstrip("\n").split("\t")
        iu = {h.split(":")[0]: i for i, h in enumerate(header)}["user_id"]
        for line in f:
            c[line.split("\t", iu + 1)[iu]] += 1
    return c


def main():
    hist = history_counts()
    vals = np.array(list(hist.values()))
    print(f"history length: n_users={len(vals)} min={vals.min()} "
          f"q25={np.percentile(vals,25):.0f} median={np.median(vals):.0f} "
          f"q75={np.percentile(vals,75):.0f} max={vals.max()}")
    cohort = {u: ("Sparse" if n < THRESH else "Dense") for u, n in hist.items()}
    sizes = Counter(cohort.values())
    print(f"cohort split at <{THRESH}: {dict(sizes)}\n")

    # 3-seed mean per-user metrics
    def load_all(pat):
        nd_s = defaultdict(list); ht_s = defaultdict(list)
        for seed in SEEDS:
            fs = glob.glob(os.path.join(RES, pat.format(seed=seed)))
            assert len(fs) == 1, (pat, seed, fs)
            d = json.load(open(fs[0]))
            nd, ht = per_user(d["test_ranks"], d["test_user_ids"])
            for u in nd:
                nd_s[u].append(nd[u]); ht_s[u].append(ht[u])
        return ({u: np.mean(v) for u, v in nd_s.items()},
                {u: np.mean(v) for u, v in ht_s.items()})

    M = {m: load_all(p) for m, p in MODELS.items()}

    print(f"{'Model':<15}{'Overall NDCG':>13}{'Sparse NDCG':>13}{'Dense NDCG':>12}"
          f"{'Sparse HR':>11}{'Dense HR':>10}")
    for m in MODELS:
        nd, ht = M[m]
        def strat(d, coh):
            return np.mean([d[u] for u in d if cohort.get(u) == coh])
        print(f"{m:<15}{np.mean(list(nd.values())):>13.4f}"
              f"{strat(nd,'Sparse'):>13.4f}{strat(nd,'Dense'):>12.4f}"
              f"{strat(ht,'Sparse'):>11.4f}{strat(ht,'Dense'):>10.4f}")

    # H2 core: paired (bge - noprof) per-user within each cohort, per seed
    print("\n## Profiler on/off (bge - noprof), paired per-user Wilcoxon by cohort\n")
    print(f"{'cohort':<8}{'seed':>7}{'metric':>7}{'mean_d':>10}{'p':>11}{'n':>8}")
    for coh in ("Sparse", "Dense"):
        for seed in SEEDS:
            fb = glob.glob(os.path.join(RES, MODELS['CAFREC_bge'].format(seed=seed)))[0]
            fn = glob.glob(os.path.join(RES, MODELS['CAFREC_noprof'].format(seed=seed)))[0]
            bnd, bht = per_user(*[json.load(open(fb))[k] for k in ("test_ranks","test_user_ids")])
            nnd, nht = per_user(*[json.load(open(fn))[k] for k in ("test_ranks","test_user_ids")])
            us = [u for u in bnd if cohort.get(u) == coh]
            for name, bd, nd in (("NDCG", bnd, nnd), ("HIT", bht, nht)):
                a = np.array([bd[u] for u in us]); b = np.array([nd[u] for u in us])
                try:
                    p = wilcoxon(a, b, zero_method="wilcox").pvalue
                except ValueError:
                    p = 1.0
                print(f"{coh:<8}{seed:>7}{name:>7}{(a-b).mean():>+10.5f}{p:>11.2e}{len(us):>8}")
        print()


if __name__ == "__main__":
    main()
