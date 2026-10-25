"""Paired permutation test of coverage@10 margins for H4 (context gate vs static fusion).

For each comparison (context gate vs static scalar gate; context gate vs concatenation)
and each user set (all test users; the 3,521-user high-drift and low-drift samples of
analyze_plan_hypotheses.py), and for each ablation seed, both models' top-10 lists are
taken for every user in the set. Each of 2,000 permutations swaps the two models' lists
independently for each user with probability 0.5, recomputes each model's coverage (in
items) over the whole set, and averages the difference over the three seeds.

    two-sided p = (1 + #{|permuted diff| >= |observed diff|}) / (N_PERM + 1)

Unlike a user bootstrap, every user appears once in every replicate, so coverage is
not shrunk by repeated users. The stratum samples are drawn exactly as in
analyze_plan_hypotheses.coverage (default_rng(7), one draw per seed in seed order,
the same users for all three models), so the observed margins reproduce Table XIII.

Usage:  .venv/Scripts/python analyze_coverage_permutation.py
Writes results/coverage_permutation_<date>.txt and .json.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy import sparse

from analyze_plan_hypotheses import ABLATION, HEADLINE, Runs, _load_json, cohorts

HERE = os.path.dirname(os.path.abspath(__file__))
N_PERM, PERM_SEED, CHUNK = 2000, 20261003, 250
GATE, ALTS = "CAFREC [abl]", ("Static gate", "Concatenation")


def stratum_rows(runs, mask, n_sample):
    """Row indices into the top-k lists, per seed, as analyze_plan_hypotheses draws them."""
    rng = np.random.default_rng(7)
    allowed = set(np.array(runs.users)[mask])
    picks = []
    for s in ABLATION:
        uids = runs.topk[(GATE, s)][0]
        rows = np.array([k for k, u in enumerate(uids) if str(u) in allowed])
        picks.append(rows[rng.choice(len(rows), size=min(n_sample, len(rows)), replace=False)])
    return picks


def incidence(items, rows, vocab):
    """Binary users x items matrix of the top-k lists in `rows`."""
    cols = np.array([vocab[i] for r in rows for i in items[r]])
    ptr = np.repeat(np.arange(len(rows)), [len(items[r]) for r in rows])
    m = sparse.csr_matrix((np.ones(len(cols)), (ptr, cols)), shape=(len(rows), len(vocab)))
    m.data[:] = 1.0  # duplicates within a list count once
    return m


def margins(runs, alt, picks, rng):
    """Observed per-seed margins (items) and N_PERM permuted margins averaged over seeds."""
    obs, perm = [], np.zeros(N_PERM)
    for s, rows in zip(ABLATION, picks):
        ua, ia, _ = runs.topk[(GATE, s)]
        ub, ib, _ = runs.topk[(alt, s)]
        assert ua == ub, f"user order differs: {alt} seed {s}"
        vocab = {i: k for k, i in enumerate(sorted({i for r in rows for i in ia[r] + ib[r]}))}
        A, B = incidence(ia, rows, vocab), incidence(ib, rows, vocab)
        obs.append(int((A.sum(0) > 0).sum() - (B.sum(0) > 0).sum()))
        AT, BT = A.T.tocsr(), B.T.tocsr()
        for start in range(0, N_PERM, CHUNK):
            w = (rng.random((len(rows), min(CHUNK, N_PERM - start))) < 0.5).astype(np.float32)
            a_cnt = AT @ (1.0 - w) + BT @ w   # items x permutations, list of model "A" after swaps
            b_cnt = AT @ w + BT @ (1.0 - w)
            perm[start:start + w.shape[1]] += ((a_cnt > 0).sum(0) - (b_cnt > 0).sum(0)) / len(ABLATION)
    return obs, perm


def main():
    t0 = time.time()
    rng = np.random.default_rng(PERM_SEED)
    probe = _load_json("SASRec", HEADLINE[0])
    users = sorted(str(u) for u in probe["test_user_ids"])
    runs = Runs(users)
    masks, _ = cohorts(users)
    for name in (GATE,) + ALTS:
        runs.get(name, ABLATION)
    n_cat = runs.topk[(GATE, ABLATION[0])][2]
    n_eq = int(min(masks["high drift"].sum(), masks["low drift"].sum()))

    lines = [f"Paired permutation test of coverage@10 margins, {N_PERM:,} permutations "
             f"(numpy seed {PERM_SEED}); seeds {ABLATION}; catalogue {n_cat:,} items; "
             f"stratum samples n={n_eq:,} (default_rng(7), as analyze_plan_hypotheses.py)"]
    out = []
    for coh in ("all", "high drift", "low drift"):
        n = int(masks[coh].sum()) if coh == "all" else n_eq
        picks = stratum_rows(runs, masks[coh], n)
        lines.append(f"\n== {coh} (n={len(picks[0]):,} users per seed) ==")
        for alt in ALTS:
            obs, perm = margins(runs, alt, picks, rng)
            d = float(np.mean(obs))
            k = int((np.abs(perm) >= abs(d)).sum())
            p = (1 + k) / (N_PERM + 1)
            lines.append(f"  {GATE} - {alt}: observed {d:+.1f} items ({d / n_cat:+.4f}); "
                         f"per seed {', '.join(f'{o:+d}' for o in obs)}; "
                         f"permuted |diff| max {np.abs(perm).max():.1f}, 95th pct {np.quantile(np.abs(perm), .95):.1f}; "
                         f"count >= observed {k}; two-sided p = {p:.5f}")
            out.append({"cohort": coh, "a": GATE, "b": alt, "n_users": len(picks[0]),
                        "observed_items": d, "observed_frac": d / n_cat, "per_seed_items": obs,
                        "perm_abs_max": float(np.abs(perm).max()),
                        "perm_abs_q95": float(np.quantile(np.abs(perm), .95)),
                        "count_ge_observed": k, "p_two_sided": p})
    lines.append(f"\nwall {time.time() - t0:.0f}s")
    stamp = time.strftime("%Y%m%d")
    meta = {"n_perm": N_PERM, "perm_seed": PERM_SEED, "seeds": ABLATION, "n_catalog": n_cat}
    with open(os.path.join(HERE, "results", f"coverage_permutation_{stamp}.json"), "w") as fh:
        json.dump({"meta": meta, "results": out}, fh, indent=1)
    txt = "\n".join(lines)
    with open(os.path.join(HERE, "results", f"coverage_permutation_{stamp}.txt"), "w") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
