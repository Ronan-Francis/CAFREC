"""Re-derive every SASRec-relative number in the paper at matched batch size 512.

The paper's SASRec reference (SASRec_kuairand_pure_ms_*) trained at batch 2048
while every CAFREC variant trained at 512 (PROJECT_LOG Cycle 10 cont.). This
script swaps in SASRec @512 (bs512_sasrec, 7 headline seeds) and recomputes:

  1. tab:res:tests    CAFREC-NP / CAFREC / CAFREC-H vs SASRec, 3 metrics, plus
                      Holm-Bonferroni over those nine tests
  2. cohort tables    per-cohort means (SASRec @512 row) and paired per-cohort
                      CAFREC-NP / CAFREC-H vs SASRec (HR, NDCG)
  3. tab:res:coverage SASRec @512 and the batch-512 HGN / HGRU4Rec (BPR and CE)
                      rows, plus a check that the existing CAFREC rows reproduce

Statistics follow Section IV-F via analyze_batch_confound.py. Cohorts: users
with < 20 rows in kuairand_pure.inter are sparse (reproduces 10,345 / 12,567).
Coverage: distinct items across a seed's top-10 lists / n_items_catalog, mean
and sd over seeds; top-10 share per seed, averaged.

Usage:  python analyze_paper_512.py [--skip-tests]
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import os
import statistics as st
import sys

from analyze_batch_confound import boot_ci, mean, per_user, sd, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "modal")
INTER = os.path.join(HERE, "..", "data", "recbole", "kuairand_pure", "kuairand_pure.inter")

HEADLINE = [42, 77, 123, 256, 512, 1024, 2048]
ABLATION = [2020, 2021, 403092]
METRICS = ("hit@10", "ndcg@10", "mrr@10")

PATS = {
    "SASRec @512":  "SASRec_kuairand_pure_bs512_sasrec_seed{s}_*.json",
    "SASRec @2048": "SASRec_kuairand_pure_ms_s{s}_seed{s}_*.json",
    "CAFREC-NP":    "CAFREC_kuairand_pure_ctx_noprof_ms_s{s}_seed{s}_*.json",
    "CAFREC":       "CAFREC_kuairand_pure_ctx_bge_ms_s{s}_seed{s}_*.json",
    "CAFREC-H":     "CAFREC_kuairand_pure_ctx_logfull_ms_s{s}_seed{s}_*.json",
}
COVERAGE = [  # (label, pattern, seeds)
    ("SASRec @512",          "SASRec_kuairand_pure_bs512_sasrec_seed{s}_*.json", HEADLINE),
    ("SASRec @2048 (paper)", "SASRec_kuairand_pure_topk_s{s}_seed{s}_*.json", ABLATION),
    ("CAFREC-NP",            "CAFREC_kuairand_pure_ctx_noprof_topk_s{s}_seed{s}_*.json", ABLATION),
    ("CAFREC",               "CAFREC_kuairand_pure_ctx_bge_topk_s{s}_seed{s}_*.json", ABLATION),
    ("Static scalar gate",   "CAFREC_kuairand_pure_ctx_static_gate_s{s}_seed{s}_*.json", ABLATION),
    ("Concatenation",        "CAFREC_kuairand_pure_ctx_concat_s{s}_seed{s}_*.json", ABLATION),
    ("CAFREC-H",             "CAFREC_kuairand_pure_ctx_hglogfull_s{s}_seed{s}_*.json", ABLATION),
    ("HGN (BPR) @512",       "HGN_kuairand_pure_bs512_hgn_seed{s}_*.json", HEADLINE),
    ("HGN (CE) @512",        "HGN_kuairand_pure_bs512_ce_hgn_seed{s}_*.json", HEADLINE),
    ("HGRU4Rec (BPR) @512",  "HGRU4Rec_kuairand_pure_bs512_hgru4rec_seed{s}_*.json", HEADLINE),
    ("HGRU4Rec (CE) @512",   "HGRU4Rec_kuairand_pure_bs512_ce_hgru4rec_seed{s}_*.json", HEADLINE),
]


def load(pattern, seed):
    hits = sorted(glob.glob(os.path.join(RES, pattern.format(s=seed))))
    if not hits:
        return None
    with open(hits[-1]) as fh:
        return json.load(fh)


def cohorts():
    cnt = collections.Counter()
    with open(INTER) as fh:
        r = csv.reader(fh, delimiter="\t")
        next(r)
        for row in r:
            cnt[row[0]] += 1
    return cnt


def paired(pv, pr, users, seeds, m):
    diffs = [mean([pv[s][u][m] for s in seeds]) - mean([pr[s][u][m] for s in seeds])
             for u in users]
    d = mean(diffs)
    sig = sign = 0
    for s in seeds:
        ds_ = [pv[s][u][m] - pr[s][u][m] for u in users]
        if wilcoxon(ds_) < 0.05:
            sig += 1
        if (mean(ds_) < 0) == (d < 0):
            sign += 1
    base = mean([mean([pr[s][u][m] for s in seeds]) for u in users])
    lo, hi = boot_ci(diffs)
    return d, lo, hi, wilcoxon(diffs), sig, sign, 100.0 * d / base


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    skip_tests = "--skip-tests" in sys.argv
    cnt = cohorts()
    data = {k: {s: load(p, s) for s in HEADLINE} for k, p in PATS.items()}
    for k, v in data.items():
        miss = [s for s in HEADLINE if v[s] is None]
        assert not miss, f"{k} missing seeds {miss}"
    pu = {k: {s: per_user(v[s]) for s in HEADLINE} for k, v in data.items()}
    users = sorted(set.intersection(*[set(pu[k][s]) for k in pu for s in HEADLINE]))
    sparse = [u for u in users if cnt[u] < 20]
    dense = [u for u in users if cnt[u] >= 20]
    print(f"users {len(users):,}  sparse {len(sparse):,}  dense {len(dense):,}\n")

    print("== Aggregate (7 seeds, mean +- sd) ==")
    for k in PATS:
        cells = [f"{mean([data[k][s]['test'][m] for s in HEADLINE]):.4f} +- "
                 f"{sd([data[k][s]['test'][m] for s in HEADLINE]):.4f}" for m in METRICS]
        print(f"  {k:<13} " + "   ".join(cells))

    print("\n== Cohort means (7 seeds): sparse HR NDCG | dense HR NDCG ==")
    for k in PATS:
        row = []
        for grp in (sparse, dense):
            for m in ("hit@10", "ndcg@10"):
                row.append(mean([mean([pu[k][s][u][m] for s in HEADLINE]) for u in grp]))
        print(f"  {k:<13} " + "  ".join(f"{x:.4f}" for x in row))

    print("\n== Coverage@10 ==")
    for label, pat, seeds in COVERAGE:
        covs, items, shares, hrs = [], [], [], []
        for s in seeds:
            d = load(pat, s)
            if d is None or "test_topk_items" not in d:
                continue
            c = collections.Counter(i for row in d["test_topk_items"] for i in row)
            items.append(len(c))
            covs.append(len(c) / d["n_items_catalog"])
            shares.append(sum(v for _, v in c.most_common(10)) / sum(c.values()))
            hrs.append(d["test"]["hit@10"])
        if not covs:
            print(f"  {label:<22} no top-k dumps")
            continue
        print(f"  {label:<22} seeds={len(covs)}  cov {mean(covs):.3f} +- {sd(covs):.3f}  "
              f"items {mean(items):,.0f}  top10share {mean(shares):.3f}  HR {mean(hrs):.4f}")

    if skip_tests:
        return

    print("\n== Paired vs SASRec @512, all users, 7 seeds ==")
    pvals, labels = [], []
    for k in ("CAFREC-NP", "CAFREC", "CAFREC-H"):
        for m in METRICS:
            d, lo, hi, p, sig, sign, pct = paired(pu[k], pu["SASRec @512"], users, HEADLINE, m)
            pvals.append(p); labels.append(f"{k} {m}")
            print(f"  {k:<10} {m:<8} delta={d:+.5f} ({pct:+.1f}%)  CI [{lo:+.5f}, {hi:+.5f}]  "
                  f"p={p:.3g}  sig {sig}/7  sign {sign}/7", flush=True)
    print("\n  Holm-Bonferroni over these nine tests:")
    for lab, p, a in zip(labels, pvals, holm(pvals)):
        print(f"    {lab:<22} p={p:.3g}  p_holm={a:.3g}")

    print("\n== Paired vs SASRec @512 within cohorts (HR, NDCG) ==")
    for k in ("CAFREC-NP", "CAFREC", "CAFREC-H"):
        for name, grp in (("sparse", sparse), ("dense", dense)):
            for m in ("hit@10", "ndcg@10"):
                d, lo, hi, p, sig, sign, pct = paired(pu[k], pu["SASRec @512"], grp, HEADLINE, m)
                print(f"  {k:<10} {name:<6} {m:<8} delta={d:+.5f} ({pct:+.1f}%)  "
                      f"CI [{lo:+.5f}, {hi:+.5f}]  p={p:.3g}  sign {sign}/7", flush=True)


if __name__ == "__main__":
    main()
