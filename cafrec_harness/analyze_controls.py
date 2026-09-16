"""Publication-form statistics for the CAFREC-NP gate controls.

Reproduces the paper's convention (Section IV-F): average each user's metric
over the seeds both conditions share, test per-user differences with the
Wilcoxon signed-rank test, and compute a 95% CI for the mean difference with a
paired bootstrap over user indices. Also reports per-seed counts of seeds
reaching p < 0.05 and seeds sharing the sign of the pooled difference.

Conditions, all CAFREC on kuairand_pure_ctx with z_long := 0 (CAFREC-NP):
    reference        no_profiler        (tag noprof_ms)
    vector gate      np_vector_gate     g := sigma(theta), theta in R^H
    shuffled context np_shuffled_ctx    x_ctx permuted across sessions

Usage:  python analyze_controls.py [results_dir]
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

SEEDS = [42, 77, 123]
N_BOOT = 2000
BOOT_SEED = 20260916

REFERENCE = ("CAFREC-NP (reference)",
             "modal/CAFREC_kuairand_pure_ctx_noprof_ms_s{s}_seed{s}_*.json")
VARIANTS = [
    ("Context-free vector gate", "modal/CAFREC_kuairand_pure_ctx_gpu_vector_gate_seed{s}_*.json"),
    ("Shuffled context",         "modal/CAFREC_kuairand_pure_ctx_gpu_shuffled_ctx_seed{s}_*.json"),
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
    ref_label, ref_pat = REFERENCE
    ref = {s: load(ref_pat, s) for s in SEEDS}
    missing = [s for s in SEEDS if ref[s] is None]
    if missing:
        print(f"reference missing seeds {missing} -- cannot proceed")
        return
    ref_pu = {s: per_user(ref[s]) for s in SEEDS}

    print(f"results dir : {RES}")
    print(f"seeds       : {SEEDS}   (headline set; the paper's ablation set is "
          f"2020/2021/403092)")
    print(f"bootstrap   : {N_BOOT} resamples, seed {BOOT_SEED}\n")

    print("=" * 78)
    print("Aggregate test metrics, mean +- sd over seeds")
    print("=" * 78)
    rows = [(ref_label, ref)]
    for label, pat in VARIANTS:
        rows.append((label, {s: load(pat, s) for s in SEEDS}))
    print(f"{'Variant':<26} {'HR@10':>17} {'NDCG@10':>17} {'MRR@10':>17}")
    for label, ds in rows:
        if any(v is None for v in ds.values()):
            print(f"{label:<26} MISSING")
            continue
        cells = []
        for m in METRICS:
            vals = [ds[s]["test"][m] for s in SEEDS]
            cells.append(f"{mean(vals):.4f} +- {sd(vals):.4f}")
        print(f"{label:<26} {cells[0]:>17} {cells[1]:>17} {cells[2]:>17}")

    print("\n" + "=" * 78)
    print(f"Paired per-user comparisons against {ref_label}")
    print("=" * 78)
    for label, pat in VARIANTS:
        ds = {s: load(pat, s) for s in SEEDS}
        if any(v is None for v in ds.values()):
            print(f"\n{label}: MISSING seeds")
            continue
        pu = {s: per_user(ds[s]) for s in SEEDS}
        common = sorted(set.intersection(*[set(ref_pu[s]) for s in SEEDS],
                                         *[set(pu[s]) for s in SEEDS]))
        print(f"\n{label}   (n = {len(common):,} users)")
        for m in METRICS:
            # user-level metric averaged over the shared seeds, then paired
            diffs = [mean([pu[s][u][m] for s in SEEDS]) - mean([ref_pu[s][u][m] for s in SEEDS])
                     for u in common]
            d = mean(diffs)
            p = wilcoxon(diffs)
            lo, hi = boot_ci(diffs)
            sig = sign = 0
            for s in SEEDS:
                ds_ = [pu[s][u][m] - ref_pu[s][u][m] for u in common]
                if wilcoxon(ds_) < 0.05:
                    sig += 1
                if (mean(ds_) < 0) == (d < 0):
                    sign += 1
            print(f"  {m:<9} delta={d:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  "
                  f"p={p:.3g}  sig {sig}/{len(SEEDS)}  sign {sign}/{len(SEEDS)}")


if __name__ == "__main__":
    main()
