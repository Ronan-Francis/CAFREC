"""Compare the 2026-09-16 GPU re-run of the gate controls against the existing
Modal CAFREC-NP baseline, pairing per-user held-out ranks by original user id.

Conditions (all CAFREC, kuairand_pure_ctx, seeds 42/77/123):
    control : no_profiler          -- gpu_noprof (new) and noprof_ms (existing)
    vg      : np_vector_gate       -- context-free per-dimension gate
    sc      : np_shuffled_ctx      -- x_ctx rows permuted (context destroyed)

Usage:  python compare_gpu_rerun.py [results_dir]
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict

RES = sys.argv[1] if len(sys.argv) > 1 else (
    r"C:\Users\Ronan\Desktop\THESIS\CAFREC\cafrec_harness\results")

SEEDS = [42, 77, 123]
# label -> (subdir, filename glob). noprof_ms is the pre-existing Modal control.
CONDITIONS = {
    "control (gpu_noprof)":   ("modal", "CAFREC_kuairand_pure_ctx_gpu_noprof_seed{s}_*.json"),
    "control (noprof_ms)":    ("modal", "CAFREC_kuairand_pure_ctx_noprof_ms_s{s}_seed{s}_*.json"),
    "np_vector_gate":         ("modal", "CAFREC_kuairand_pure_ctx_gpu_vector_gate_seed{s}_*.json"),
    "np_shuffled_ctx":        ("modal", "CAFREC_kuairand_pure_ctx_gpu_shuffled_ctx_seed{s}_*.json"),
}


def load(subdir, pattern, seed):
    hits = sorted(glob.glob(os.path.join(RES, subdir, pattern.format(s=seed))))
    if not hits:
        return None
    with open(hits[-1]) as fh:                     # newest if several
        return json.load(fh)


def per_user(d):
    """{user_id: (hit@10, ndcg@10)} from the rank dump."""
    out = {}
    for u, r in zip(d["test_user_ids"], d["test_ranks"]):
        if r is None:
            continue
        hit = 1.0 if r <= 10 else 0.0
        ndcg = (1.0 / math.log2(r + 1)) if r <= 10 else 0.0
        out[str(u)] = (hit, ndcg)
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def wilcoxon(diffs):
    """Two-sided Wilcoxon signed-rank via normal approximation (n is large here)."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n < 20:
        return float("nan"), n
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
    z = (w_plus - mu) / sigma
    p = math.erfc(abs(z) / math.sqrt(2))
    return p, n


def main():
    loaded = defaultdict(dict)
    print(f"results dir: {RES}\n")
    for label, (sub, pat) in CONDITIONS.items():
        for s in SEEDS:
            d = load(sub, pat, s)
            if d is not None:
                loaded[label][s] = d
        have = sorted(loaded[label])
        print(f"  {label:<24} seeds found: {have if have else 'NONE'}")

    print("\n" + "=" * 74)
    print("Test metrics, mean +- sd over seeds")
    print("=" * 74)
    print(f"{'condition':<24} {'n':>2}  {'HR@10':>16} {'NDCG@10':>16} {'MRR@10':>16}")
    for label in CONDITIONS:
        ds = loaded[label]
        if not ds:
            continue
        cols = []
        for k in ("hit@10", "ndcg@10", "mrr@10"):
            vals = [d["test"][k] for d in ds.values()]
            cols.append(f"{mean(vals):.4f} +- {sd(vals):.4f}" if len(vals) > 1
                        else f"{mean(vals):.4f}          ")
        print(f"{label:<24} {len(ds):>2}  {cols[0]:>16} {cols[1]:>16} {cols[2]:>16}")

    # paired per-user tests against whichever control is present
    # noprof_ms, not gpu_noprof: same condition, same code path, but it has all
    # three seeds. gpu_noprof seed 42 exists only as a GPU-reproduces-GPU check.
    ctrl_label = "control (noprof_ms)"
    a, b = loaded["control (gpu_noprof)"].get(42), loaded["control (noprof_ms)"].get(42)
    if a and b:
        pa, pb = per_user(a), per_user(b)
        common = sorted(set(pa) & set(pb))
        dn = [pa[u][1] - pb[u][1] for u in common]
        pv, n = wilcoxon(dn)
        print(f"\nreproducibility check, gpu_noprof vs noprof_ms @ seed 42: "
              f"dNDCG={mean(dn):+.5f}  p={pv:.3g}  (n={len(common)})")
    print("\n" + "=" * 74)
    print(f"Paired per-user tests vs {ctrl_label}  (Wilcoxon, two-sided)")
    print("=" * 74)
    for label in ("np_vector_gate", "np_shuffled_ctx"):
        if not loaded[label]:
            continue
        print(f"\n{label}")
        per_seed_sign = []
        for s in SEEDS:
            c, t = loaded[ctrl_label].get(s), loaded[label].get(s)
            if c is None or t is None:
                print(f"  seed {s:<4} skipped (missing {'control' if c is None else label})")
                continue
            pc, pt = per_user(c), per_user(t)
            common = sorted(set(pc) & set(pt))
            dn = [pt[u][1] - pc[u][1] for u in common]
            dh = [pt[u][0] - pc[u][0] for u in common]
            p, n = wilcoxon(dn)
            per_seed_sign.append(mean(dn))
            print(f"  seed {s:<4} users={len(common):<6} "
                  f"dNDCG={mean(dn):+.5f}  dHR={mean(dh):+.5f}  "
                  f"nonzero={n:<6} p={p:.3g}")
        if per_seed_sign:
            neg = sum(1 for d in per_seed_sign if d < 0)
            print(f"  -> NDCG lower than control in {neg}/{len(per_seed_sign)} seeds")


if __name__ == "__main__":
    main()
