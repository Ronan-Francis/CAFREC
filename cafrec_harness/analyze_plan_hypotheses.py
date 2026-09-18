"""Tests of the hypotheses AS REGISTERED in Project_Plan_2, plus the statistics fixes
requested in the 2026-09-17 supervisor review. CPU only; uses existing rank/top-k dumps.

Registered hypotheses (Project_Plan_2, Sec. II-C) and how each is operationalised:

  H1  Dual-network architectures (CAFREC, with profile) outperform SASRec, HGN and
      HGRU4Rec on HR@10 and NDCG@10, including on the short-session cohort.
      -> CAFREC vs each baseline (HGN/HGRU4Rec under native BPR and matched CE),
         all users and the short-session cohort, 7 headline seeds, batch 512.
  H2  Profiles achieve higher NDCG@10 than interaction-derived embeddings for users
      in the lowest quartile of interaction counts.
      -> frozen bge profile vs trainable stand-in (the interaction-derived per-user
         embedding), lowest-quartile users, 3 ablation seeds; CAFREC vs CAFREC-NP
         (no user embedding at all) on the same users, 7 seeds, as a second reading.
  H3  Context-adaptive gating outperforms static-weighted and concatenation fusion
      across all activity strata, with the largest margin on high-category-drift sessions.
      -> CAFREC vs static gate / concat within each activity quartile and within
         high- vs low-drift sessions (NDCG@10, HR@10), plus a test that the margin is
         larger for high-drift sessions.
  H4  Context-adaptive gating yields higher ILD and coverage than static fusion, with
      the largest gains in high-drift sessions.
      -> coverage only (ILD needs video_features_basic_pure.csv), overall and within
         drift strata on equal-size user samples. Descriptive (3 seeds).

Cohort definitions (fixed here, before looking at any stratified result):
  activity quartiles  rows per user in kuairand_pure.inter (organic clicks)
  short-session       held-out interaction has NO earlier interaction in its session
                      (prefix session length 0; 68.5% of test users)
  high-drift          held-out interaction's session prefix contains at least one tag
                      change (raw prefix category drift > 0.1; ~25% of test users)

Statistics: per-user metric averaged over shared seeds; two-sided Wilcoxon signed-rank
(normal approximation, as analyze_batch_confound.py); paired bootstrap 95% CI
(2,000 user resamples, numpy); per-seed significance/sign counts; and, for 7-seed
comparisons, SEED-LEVEL paired tests on the seven per-seed means (exact Wilcoxon and
paired t). Holm-Bonferroni within each hypothesis family and globally across all
registered-hypothesis tests. Post-hoc variants (CAFREC-H, controls) are excluded
from the families.

Usage:  .venv/Scripts/python analyze_plan_hypotheses.py
Writes results/plan_hypotheses_<date>.txt and .json.
"""
from __future__ import annotations

import collections
import glob
import json
import math
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata, ttest_rel, wilcoxon as sp_wilcoxon, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "modal")
DATA = os.path.join(HERE, "..", "data", "recbole")
SCALER = os.path.join(HERE, "..", "data", "ctx_scaler_params.json")

HEADLINE = [42, 77, 123, 256, 512, 1024, 2048]
ABLATION = [2020, 2021, 403092]
N_BOOT, BOOT_SEED = 2000, 20260917

PATS = {
    "SASRec":        ["SASRec_kuairand_pure_bs512_sasrec_seed{s}_*.json"],
    "CAFREC-NP":     ["CAFREC_kuairand_pure_ctx_noprof_ms_s{s}_seed{s}_*.json"],
    "CAFREC":        ["CAFREC_kuairand_pure_ctx_bge_ms_s{s}_seed{s}_*.json"],
    "HGN (BPR)":     ["HGN_kuairand_pure_bs512_hgn_seed{s}_*.json"],
    "HGN (CE)":      ["HGN_kuairand_pure_bs512_ce_hgn_seed{s}_*.json"],
    "HGRU4Rec (BPR)": ["HGRU4Rec_kuairand_pure_bs512_hgru4rec_seed{s}_*.json"],
    "HGRU4Rec (CE)": ["HGRU4Rec_kuairand_pure_bs512_ce_hgru4rec_seed{s}_*.json"],
    # ablation seeds
    "CAFREC [abl]":    ["CAFREC_kuairand_pure_ctx_bge_topk_s{s}_seed{s}_*.json"],
    "CAFREC-NP [abl]": ["CAFREC_kuairand_pure_ctx_noprof_topk_s{s}_seed{s}_*.json"],
    "Static gate":     ["CAFREC_kuairand_pure_ctx_static_gate_s{s}_seed{s}_*.json"],
    "Concatenation":   ["CAFREC_kuairand_pure_ctx_concat_s{s}_seed{s}_*.json"],
    "Stand-in":        ["CAFREC_kuairand_pure_ctx_standin_s{s}_seed{s}_*.json",
                        "CAFREC_kuairand_pure_ctx_standin_seed{s}_*.json"],
    "Vector gate":     ["CAFREC_kuairand_pure_ctx_vector_gate_s{s}_seed{s}_*.json"],
    # replication of the session-position pattern at batch 2048 (7 headline seeds)
    "SASRec @2048":    ["SASRec_kuairand_pure_ms_s{s}_seed{s}_*.json"],
    "CAFREC-NP @2048": ["CAFREC_kuairand_pure_ctx_bs2048_noprof_seed{s}_*.json"],
    "Shuffled ctx":    ["CAFREC_kuairand_pure_ctx_shuffled_ctx_s{s}_seed{s}_*.json"],
}


# ---------------------------------------------------------------- loading
def _load_json(name, seed):
    for pat in PATS[name]:
        hits = sorted(glob.glob(os.path.join(RES, pat.format(s=seed))))
        if hits:
            with open(hits[-1]) as fh:
                return json.load(fh)
    raise FileNotFoundError(f"{name} seed {seed}")


class Runs:
    """Per-user metric matrices [n_seeds, n_users] aligned to one user order."""

    def __init__(self, users):
        self.users = users
        self.index = {u: i for i, u in enumerate(users)}
        self.cache, self.topk = {}, {}

    def get(self, name, seeds):
        key = (name, tuple(seeds))
        if key in self.cache:
            return self.cache[key]
        ranks = np.full((len(seeds), len(self.users)), np.nan)
        for j, s in enumerate(seeds):
            d = _load_json(name, s)
            for u, r in zip(d["test_user_ids"], d["test_ranks"]):
                i = self.index.get(str(u))
                if i is not None and r is not None:
                    ranks[j, i] = r
            if "test_topk_items" in d:
                self.topk[(name, s)] = (d["test_topk_user_ids"], d["test_topk_items"],
                                        d["n_items_catalog"])
        ink = ranks <= 10
        m = {"hit@10": ink.astype(float),
             "ndcg@10": np.where(ink, 1.0 / np.log2(ranks + 1.0), 0.0),
             "mrr@10": np.where(ink, 1.0 / ranks, 0.0)}
        assert not np.isnan(ranks).any(), f"{name}: users missing from dump"
        self.cache[key] = m
        return m


def cohorts(users):
    inter = pd.read_csv(os.path.join(DATA, "kuairand_pure", "kuairand_pure.inter"),
                        sep="\t", usecols=[0], dtype=str)
    counts = inter.iloc[:, 0].value_counts()
    n = np.array([counts.get(u, 0) for u in users])
    q1, q2, q3 = np.quantile(n, [0.25, 0.5, 0.75])
    quart = np.select([n <= q1, n <= q2, n <= q3], [1, 2, 3], 4)

    ctx = pd.read_csv(os.path.join(DATA, "kuairand_pure_ctx", "kuairand_pure_ctx.inter"),
                      sep="\t", dtype={"user_id:token": str})
    ctx.columns = [c.split(":")[0] for c in ctx.columns]
    last = ctx.sort_values(["user_id", "timestamp"], kind="stable").groupby("user_id").tail(1)
    last = last.set_index("user_id").loc[users]
    sc = json.load(open(SCALER))
    mu, sd = dict(zip(sc["features"], sc["mean"])), dict(zip(sc["features"], sc["std"]))
    plen = np.expm1(last.prefix_session_len_log_z * sd["prefix_session_len_log"]
                    + mu["prefix_session_len_log"]).round().to_numpy()
    drift = (last.prefix_category_drift_z * sd["prefix_category_drift"]
             + mu["prefix_category_drift"]).to_numpy()
    return {
        "all": np.ones(len(users), bool),
        "sparse (<20)": n < 20, "dense (>=20)": n >= 20,
        "Q1 activity": quart == 1, "Q2 activity": quart == 2,
        "Q3 activity": quart == 3, "Q4 activity": quart == 4,
        "short session": plen == 0, "longer session": plen > 0,
        "high drift": drift > 0.1, "low drift": drift <= 0.1,
    }, {"quartile_cuts": [float(q1), float(q2), float(q3)]}


# ---------------------------------------------------------------- statistics
def wilcoxon_normal(d):
    nz = d[d != 0]
    n = len(nz)
    if n < 20:
        return float("nan")
    r = rankdata(np.abs(nz))
    w_plus = r[nz > 0].sum()
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    return math.erfc(abs((w_plus - mu) / sigma) / math.sqrt(2))


def boot_ci(d, rng):
    n = len(d)
    means = np.empty(N_BOOT)
    for start in range(0, N_BOOT, 100):
        stop = min(start + 100, N_BOOT)
        idx = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = d[idx].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def compare(runs, a, b, seeds, metric, mask, rng):
    A, B = runs.get(a, seeds)[metric][:, mask], runs.get(b, seeds)[metric][:, mask]
    d = A.mean(0) - B.mean(0)
    delta = float(d.mean())
    base = float(B.mean())
    lo, hi = boot_ci(d, rng)
    sig = sign = 0
    for j in range(len(seeds)):
        ds = A[j] - B[j]
        sig += wilcoxon_normal(ds) < 0.05
        sign += (ds.mean() < 0) == (delta < 0)
    out = {"a": a, "b": b, "metric": metric, "n_users": int(mask.sum()),
           "seeds": len(seeds), "delta": delta, "rel_pct": 100 * delta / base if base else float("nan"),
           "ci": [lo, hi], "p": wilcoxon_normal(d), "sig": int(sig), "sign": int(sign)}
    if len(seeds) >= 5:
        sa, sb = A.mean(1), B.mean(1)
        out["seed_p_wilcoxon_exact"] = float(sp_wilcoxon(sa, sb, method="exact").pvalue)
        out["seed_p_ttest"] = float(ttest_rel(sa, sb).pvalue)
    out["_diffs"] = d
    return out


def holm(ps):
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def coverage(runs, name, seeds, users_mask, n_sample, rng):
    vals = []
    allowed_all = set(np.array(runs.users)[users_mask])
    for s in seeds:
        uids, items, n_cat = runs.topk[(name, s)]
        rows = [row for u, row in zip(uids, items) if str(u) in allowed_all]
        pick = rng.choice(len(rows), size=min(n_sample, len(rows)), replace=False)
        c = collections.Counter(i for k in pick for i in rows[k])
        vals.append(len(c) / n_cat)
    return float(np.mean(vals)), float(np.std(vals, ddof=1)), vals


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    rng = np.random.default_rng(BOOT_SEED)
    probe = _load_json("SASRec", HEADLINE[0])
    users = sorted(str(u) for u in probe["test_user_ids"])
    runs = Runs(users)
    masks, meta = cohorts(users)
    lines = [f"Users {len(users):,}; activity quartile cuts {meta['quartile_cuts']}"]
    lines += [f"  {k:<15} n={int(v.sum()):,}" for k, v in masks.items()]

    fam = collections.defaultdict(list)

    # H1 ------------------------------------------------------------------
    for base in ("SASRec", "HGN (BPR)", "HGN (CE)", "HGRU4Rec (BPR)", "HGRU4Rec (CE)"):
        for coh in ("all", "short session"):
            for m in ("hit@10", "ndcg@10"):
                fam["H1"].append(compare(runs, "CAFREC", base, HEADLINE, m, masks[coh], rng)
                                 | {"cohort": coh})
    for coh in ("all", "short session"):
        for m in ("hit@10", "ndcg@10"):
            fam["H1"].append(compare(runs, "CAFREC-NP", "SASRec", HEADLINE, m, masks[coh], rng)
                             | {"cohort": coh})

    # Heterogeneity by session position (NOT in the Holm families: the plan named only
    # the short-session cohort; the complement and the interaction are follow-ups).
    hetero, controls = [], []
    for a, ref in (("CAFREC-NP", "SASRec"), ("CAFREC", "SASRec"), ("CAFREC-NP @2048", "SASRec @2048")):
        for m in ("hit@10", "ndcg@10"):
            sh = compare(runs, a, ref, HEADLINE, m, masks["short session"], rng)
            lo = compare(runs, a, ref, HEADLINE, m, masks["longer session"], rng)
            dh, dl = sh["_diffs"], lo["_diffs"]
            bh = np.array([dh[rng.integers(0, len(dh), len(dh))].mean()
                           - dl[rng.integers(0, len(dl), len(dl))].mean() for _ in range(N_BOOT)])
            # seed-level interaction: per-seed (short effect - longer effect), n = 7
            A, B = runs.get(a, HEADLINE)[m], runs.get(ref, HEADLINE)[m]
            inter = ((A - B)[:, masks["short session"]].mean(1)
                     - (A - B)[:, masks["longer session"]].mean(1))
            hetero.append({"a": a, "metric": m, "short": sh, "longer": lo,
                           "interaction": float(dh.mean() - dl.mean()),
                           "ci": [float(np.quantile(bh, .025)), float(np.quantile(bh, .975))],
                           "p_mannwhitney": float(mannwhitneyu(dh, dl).pvalue),
                           "seed_interaction_p_W": float(sp_wilcoxon(inter, method="exact").pvalue),
                           "seed_interaction_sign": int((np.sign(inter) == np.sign(inter.mean())).sum())})
    # Do the gate controls lose accuracy where the gate appears to help? (ablation seeds)
    for ctl in ("Shuffled ctx", "Vector gate"):
        for coh in ("short session", "longer session"):
            for m in ("hit@10", "ndcg@10"):
                controls.append(compare(runs, ctl, "CAFREC-NP [abl]", ABLATION, m, masks[coh], rng)
                                | {"cohort": coh})
    for coh in ("short session", "longer session"):
        for m in ("hit@10", "ndcg@10"):
            controls.append(compare(runs, "CAFREC-NP [abl]", "CAFREC [abl]", ABLATION, m, masks[coh], rng)
                            | {"cohort": coh})

    # H2 ------------------------------------------------------------------
    for coh in ("Q1 activity", "all"):
        fam["H2"].append(compare(runs, "CAFREC [abl]", "Stand-in", ABLATION, "ndcg@10",
                                 masks[coh], rng) | {"cohort": coh})
        fam["H2"].append(compare(runs, "CAFREC", "CAFREC-NP", HEADLINE, "ndcg@10",
                                 masks[coh], rng) | {"cohort": coh})

    # H3 ------------------------------------------------------------------
    strata = ("all", "Q1 activity", "Q2 activity", "Q3 activity", "Q4 activity",
              "high drift", "low drift")
    for alt in ("Static gate", "Concatenation"):
        for coh in strata:
            for m in ("ndcg@10", "hit@10"):
                fam["H3"].append(compare(runs, "CAFREC [abl]", alt, ABLATION, m, masks[coh], rng)
                                 | {"cohort": coh})
    margin = []
    for alt in ("Static gate", "Concatenation"):
        for m in ("ndcg@10", "hit@10"):
            hi = next(r for r in fam["H3"] if r["b"] == alt and r["cohort"] == "high drift" and r["metric"] == m)
            lo = next(r for r in fam["H3"] if r["b"] == alt and r["cohort"] == "low drift" and r["metric"] == m)
            dh, dl = hi["_diffs"], lo["_diffs"]
            bh = np.array([dh[rng.integers(0, len(dh), len(dh))].mean() - dl[rng.integers(0, len(dl), len(dl))].mean()
                           for _ in range(N_BOOT)])
            margin.append({"alt": alt, "metric": m, "margin_high_minus_low": float(dh.mean() - dl.mean()),
                           "ci": [float(np.quantile(bh, .025)), float(np.quantile(bh, .975))],
                           "p_mannwhitney": float(mannwhitneyu(dh, dl).pvalue)})

    # Holm -----------------------------------------------------------------
    allrows = [r for f in ("H1", "H2", "H3") for r in fam[f]]
    g_adj = holm(np.array([r["p"] for r in allrows]))
    for r, a in zip(allrows, g_adj):
        r["p_holm_global"] = float(a)
    for f in ("H1", "H2", "H3"):
        for r, a in zip(fam[f], holm(np.array([r["p"] for r in fam[f]]))):
            r["p_holm_family"] = float(a)

    def fmt(r):
        s = (f"  {r['a']:<13} vs {r['b']:<15} {r['cohort']:<14} {r['metric']:<8} "
             f"d={r['delta']:+.5f} ({r['rel_pct']:+.1f}%) CI[{r['ci'][0]:+.5f},{r['ci'][1]:+.5f}] "
             f"p={r['p']:.3g} holm_fam={r['p_holm_family']:.3g} holm_all={r['p_holm_global']:.3g} "
             f"sig {r['sig']}/{r['seeds']} sign {r['sign']}/{r['seeds']}")
        if "seed_p_ttest" in r:
            s += f" | seed-level p_t={r['seed_p_ttest']:.3g} p_W={r['seed_p_wilcoxon_exact']:.3g}"
        return s

    for f, title in (("H1", "H1 (plan): CAFREC vs baselines, all users and short-session cohort"),
                     ("H2", "H2 (plan): profile vs interaction-derived embedding, lowest activity quartile"),
                     ("H3", "H3 (plan): context gate vs static / concat by activity quartile and drift")):
        lines.append(f"\n== {title} ==")
        lines += [fmt(r) for r in fam[f]]
    lines.append("\n== H3 margin: (high drift) minus (low drift) ==")
    lines += [f"  CAFREC vs {m['alt']:<14} {m['metric']:<8} margin={m['margin_high_minus_low']:+.5f} "
              f"CI[{m['ci'][0]:+.5f},{m['ci'][1]:+.5f}] Mann-Whitney p={m['p_mannwhitney']:.3g}" for m in margin]

    lines.append("\n== Follow-up: session-position heterogeneity vs SASRec (7 seeds; not Holm-corrected) ==")
    for h in hetero:
        for part in ("short", "longer"):
            r = h[part]
            lines.append(f"  {h['a']:<10} {part:<7} {h['metric']:<8} d={r['delta']:+.5f} ({r['rel_pct']:+.1f}%) "
                         f"CI[{r['ci'][0]:+.5f},{r['ci'][1]:+.5f}] p={r['p']:.3g} sign {r['sign']}/7 "
                         f"| seed-level p_t={r['seed_p_ttest']:.3g}")
        lines.append(f"  {h['a']:<10} short-minus-longer {h['metric']:<8} {h['interaction']:+.5f} "
                     f"CI[{h['ci'][0]:+.5f},{h['ci'][1]:+.5f}] MW p={h['p_mannwhitney']:.3g} "
                     f"seed-level exact W p={h['seed_interaction_p_W']:.3g} (same sign {h['seed_interaction_sign']}/7)")
    lines.append("\n== Follow-up: controls by session position (ablation seeds, vs CAFREC-NP; not corrected) ==")
    for r in controls:
        r2 = dict(r); r2.setdefault("p_holm_family", float("nan")); r2.setdefault("p_holm_global", float("nan"))
        lines.append(fmt(r2))

    # H4 coverage ----------------------------------------------------------
    lines.append("\n== H4 (plan): coverage@10, equal-size user samples (3 ablation seeds) ==")
    n_eq = int(min(masks["high drift"].sum(), masks["low drift"].sum()))
    cov = []
    for name in ("CAFREC [abl]", "Static gate", "Concatenation"):
        runs.get(name, ABLATION)
        row = {"model": name}
        for coh in ("all", "high drift", "low drift"):
            n = int(masks[coh].sum()) if coh == "all" else n_eq
            mu_, sd_, _ = coverage(runs, name, ABLATION, masks[coh], n, np.random.default_rng(7))
            row[coh] = [mu_, sd_]
        cov.append(row)
        lines.append(f"  {name:<14} all {row['all'][0]:.3f}+-{row['all'][1]:.3f}  "
                     f"high drift {row['high drift'][0]:.3f}+-{row['high drift'][1]:.3f}  "
                     f"low drift {row['low drift'][0]:.3f}+-{row['low drift'][1]:.3f}  (n={n_eq:,} per stratum)")

    stamp = time.strftime("%Y%m%d")
    for f in fam.values():
        for r in f:
            r.pop("_diffs", None)
    for h in hetero:
        h["short"].pop("_diffs", None); h["longer"].pop("_diffs", None)
    for r in controls:
        r.pop("_diffs", None)
    with open(os.path.join(HERE, "results", f"plan_hypotheses_{stamp}.json"), "w") as fh:
        json.dump({"meta": meta, "families": fam, "h3_margin": margin, "h4_coverage": cov,
                   "session_heterogeneity": hetero, "controls_by_session": controls}, fh, indent=1)
    lines.append(f"\nwall {time.time() - t0:.0f}s")
    txt = "\n".join(lines)
    with open(os.path.join(HERE, "results", f"plan_hypotheses_{stamp}.txt"), "w") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
