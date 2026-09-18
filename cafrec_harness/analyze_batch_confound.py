"""Batch-size confound check for the H1 headline (Cycle 10, 2026-09-16).

SASRec ran on dataset "kuairand_pure" and CAFREC-NP on "kuairand_pure_ctx";
modal_run.py's `dataset != "kuairand_pure"` branch therefore trained SASRec at
train_batch_size 2048 and CAFREC-NP at 512. This script crosses the two models
over both batch sizes (2 x 2, seven headline seeds) and reports, for each pair
of conditions, the paper's per-user statistics (Section IV-F): each user's
metric averaged over the shared seeds, a Wilcoxon signed-rank test on the
per-user differences, a 95% paired bootstrap CI over user indices, and
per-seed counts of seeds reaching p < 0.05 and seeds sharing the pooled sign.

Conditions (results/modal/):
    SASRec    @2048   SASRec_kuairand_pure_ms_*                  (existing)
    SASRec    @512    SASRec_kuairand_pure_bs512_sasrec_*        (bsweep)
    CAFREC-NP @2048   CAFREC_kuairand_pure_ctx_bs2048_noprof_*   (bsweep)
    CAFREC-NP @512    CAFREC_kuairand_pure_ctx_noprof_ms_*       (existing)

Usage:  python analyze_batch_confound.py [results_dir]
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import sys

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results")

SEEDS = [42, 77, 123, 256, 512, 1024, 2048]
N_BOOT = 2000
BOOT_SEED = 20260916

PATS = {
    "SASRec @2048":    "modal/SASRec_kuairand_pure_ms_s{s}_seed{s}_*.json",
    "SASRec @512":     "modal/SASRec_kuairand_pure_bs512_sasrec_seed{s}_*.json",
    "CAFREC-NP @2048": "modal/CAFREC_kuairand_pure_ctx_bs2048_noprof_seed{s}_*.json",
    "CAFREC-NP @512":  "modal/CAFREC_kuairand_pure_ctx_noprof_ms_s{s}_seed{s}_*.json",
}
# (variant, reference, what it tests)
PAIRS = [
    ("CAFREC-NP @2048", "SASRec @2048",    "gate effect at MATCHED batch 2048"),
    ("CAFREC-NP @512",  "SASRec @512",     "gate effect at MATCHED batch 512"),
    ("CAFREC-NP @512",  "SASRec @2048",    "the PAPER'S comparison (confounded)"),
    ("SASRec @512",     "SASRec @2048",    "batch-size effect on SASRec alone"),
]
METRICS = ("hit@10", "ndcg@10", "mrr@10")


def load(pattern, seed):
    hits = sorted(glob.glob(os.path.join(RES, pattern.format(s=seed))))
    if not hits:
        return None
    with open(hits[-1]) as fh:
        return json.load(fh)


def per_user(d):
    """{user_id: {metric: value}} from the held-out rank dump."""
    out = {}
    for u, r in zip(d["test_user_ids"], d["test_ranks"]):
        if r is None:
            continue
        hit = 1.0 if r <= 10 else 0.0
        out[str(u)] = {
            "hit@10": hit,
            "ndcg@10": (1.0 / math.log2(r + 1)) if r <= 10 else 0.0,
            "mrr@10": (1.0 / r) if r <= 10 else 0.0,
        }
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def wilcoxon(diffs):
    """Two-sided Wilcoxon signed-rank, normal approximation with tie-average ranks."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n < 20:
        return float("nan")
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(ranks[i] for i in range(n) if nz[i] > 0)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    return math.erfc(abs((w_plus - mu) / sigma) / math.sqrt(2))


def boot_ci(diffs, n_boot=N_BOOT, seed=BOOT_SEED):
    """Paired bootstrap over user indices; returns (lo, hi) of the mean difference."""
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def main():
    loaded = {k: {s_: load(pat, s_) for s_ in SEEDS} for k, pat in PATS.items()}
    print(f"results dir : {RES}")
    print(f"seeds       : {SEEDS}")
    print(f"bootstrap   : {N_BOOT} resamples, seed {BOOT_SEED}")
    print()
    print("=" * 78)
    print("Aggregate test metrics, mean +- sd over 7 seeds")
    print("=" * 78)
    print(f"{'Condition':<18} {'HR@10':>17} {'NDCG@10':>17} {'MRR@10':>17}")
    for k in PATS:
        ds = loaded[k]
        if any(v is None for v in ds.values()):
            miss = [s_ for s_ in SEEDS if ds[s_] is None]
            print(f"{k:<18} MISSING seeds {miss}")
            continue
        cells = []
        for m in METRICS:
            vals = [ds[s_]["test"][m] for s_ in SEEDS]
            cells.append(f"{mean(vals):.4f} +- {sd(vals):.4f}")
        print(f"{k:<18} {cells[0]:>17} {cells[1]:>17} {cells[2]:>17}")

    print()
    print("=" * 78)
    print("Paired per-user comparisons")
    print("=" * 78)
    for var, ref, what in PAIRS:
        dv, dr = loaded[var], loaded[ref]
        if any(v is None for v in dv.values()) or any(v is None for v in dr.values()):
            print(f"\n{var} vs {ref}: MISSING")
            continue
        pv = {s_: per_user(dv[s_]) for s_ in SEEDS}
        pr = {s_: per_user(dr[s_]) for s_ in SEEDS}
        common = sorted(set.intersection(*[set(pv[s_]) for s_ in SEEDS],
                                         *[set(pr[s_]) for s_ in SEEDS]))
        print(f"\n{var}  vs  {ref}    [{what}]   n = {len(common):,}")
        for m in METRICS:
            diffs = [mean([pv[s_][u][m] for s_ in SEEDS]) - mean([pr[s_][u][m] for s_ in SEEDS])
                     for u in common]
            d = mean(diffs); pval = wilcoxon(diffs); lo, hi = boot_ci(diffs)
            sig = sign = 0
            for s_ in SEEDS:
                ds_ = [pv[s_][u][m] - pr[s_][u][m] for u in common]
                if wilcoxon(ds_) < 0.05: sig += 1
                if (mean(ds_) < 0) == (d < 0): sign += 1
            pct = 100.0 * d / mean([mean([pr[s_][u][m] for s_ in SEEDS]) for u in common])
            print(f"  {m:<9} delta={d:+.5f} ({pct:+.1f}%)  95% CI [{lo:+.5f}, {hi:+.5f}]  "
                  f"p={pval:.3g}  sig {sig}/7  sign {sign}/7")


if __name__ == "__main__":
    main()
