"""Intra-list diversity (ILD@10) for runs that saved top-10 lists (2026-09-17).

ILD@10 = mean over users of the mean pairwise (1 - cosine) between the category
vectors of the ten recommended items (cafrec.eval.diversity.intra_list_diversity).

Category sources (alignment audit: results/content_data_audit_2026-09-17.txt)
  first_level   kuairand_video_categories.csv first-level category, one-hot (PRIMARY)
  second_level  same file, second-level category, one-hot (sensitivity)
  kuairand_tag  video_features_basic_pure.csv `tag`, multi-hot (sensitivity)
A video with no category at a level gets a zero row, which intra_list_diversity
treats as maximally dissimilar to everything; the share of recommended slots
holding such a video is reported next to each ILD.

Model groups and seed sets are the paper's coverage table (analyze_paper_512.COVERAGE,
minus its SASRec @2048 row) plus MostPop (results/local).

H4 as registered: context gate (CAFREC) vs static gate and vs concatenation, overall
and within drift strata (analyze_plan_hypotheses.cohorts: high drift = raw prefix
category drift > 0.1 on the held-out row). Strata use one equal-size user sample
(n = size of the smaller stratum) drawn once and shared by every model and seed, so
user-level differences pair exactly. Descriptive mean +- sd over the 3 ablation
seeds; per-user paired Wilcoxon (normal approximation) and bootstrap CI on ILD
averaged over seeds; the high-minus-low drift margin with a bootstrap CI and
Mann-Whitney p. Full-stratum (unsampled) paired tests are reported as a check.

Also writes ILD for every KuaiRand-Pure result JSON with top-k lists to
results/ild_all_runs_<date>.csv.

Usage (from cafrec_harness/):  <env python> analyze_ild.py [--skip-all-runs]
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from analyze_paper_512 import COVERAGE
from analyze_plan_hypotheses import ABLATION, BOOT_SEED, N_BOOT, boot_ci, cohorts, wilcoxon_normal
from cafrec.eval.category_matrix import build_category_matrix, category_matrix_from_dataframe
from cafrec.eval.diversity import intra_list_diversity

HERE = os.path.dirname(os.path.abspath(__file__))
RES_MODAL = os.path.join(HERE, "results", "modal")
RES_LOCAL = os.path.join(HERE, "results", "local")
DATA = os.path.join(HERE, "..", "data")
CAT_CSV = os.path.join(DATA, "content_pure", "categories_pure.csv")
BASIC = os.path.join(DATA, "KuaiRand-Pure", "data", "video_features_basic_pure.csv")
UNKNOWN_ID = -124.0
DATE = time.strftime("%Y-%m-%d")
LEVELS = ("first_level", "second_level", "kuairand_tag")

GROUPS = [(label, RES_MODAL, pat, seeds) for label, pat, seeds in COVERAGE
          if label != "SASRec @2048 (paper)"]
GROUPS.append(("MostPop", RES_LOCAL, "Pop_kuairand_pure_pop_seed{s}.json", [2020]))
H4_MODELS = ("CAFREC", "Static scalar gate", "Concatenation")


# ---------------------------------------------------------------- categories
def category_matrices():
    cat = pd.read_csv(CAT_CSV)
    basic = pd.read_csv(BASIC)
    n_items = int(max(cat["final_video_id"].max(), basic["video_id"].max())) + 1

    def level(lvl):
        v = cat[f"{lvl}_level_category_id"]
        tokens = [None if (pd.isna(x) or x == UNKNOWN_ID) else str(int(x)) for x in v]
        m, vocab = build_category_matrix(cat["final_video_id"].to_numpy(), tokens,
                                         strategy="primary", n_items=n_items)
        return m, len(vocab)

    first, k1 = level("first")
    second, k2 = level("second")
    tag, tag_vocab = category_matrix_from_dataframe(basic, item_col="video_id", tag_col="tag",
                                                    strategy="multihot", n_items=n_items)
    mats = {"first_level": first, "second_level": second, "kuairand_tag": tag}
    dims = {"first_level": k1, "second_level": k2, "kuairand_tag": len(tag_vocab)}
    return mats, dims


def unit_rows(m):
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.clip(norms, 1e-12, None)).astype(np.float32), (norms[:, 0] == 0)


def per_user_ild(topk, unit):
    v = unit[topk]                                       # [n, k, d]
    sim = np.einsum("nkd,njd->nkj", v, v)
    iu = np.triu_indices(topk.shape[1], k=1)
    return (1.0 - sim[:, iu[0], iu[1]]).mean(axis=1)     # [n]


# ---------------------------------------------------------------- run loading
def load_json(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    if b'"test_topk_items"' not in raw:
        return None
    return json.loads(raw)


def find(directory, pattern, seed):
    hits = sorted(glob.glob(os.path.join(directory, pattern.format(s=seed))))
    return hits[-1] if hits else None


def run_diversity(d, units, zero_rows):
    topk = np.asarray(d["test_topk_items"], dtype=np.int64)
    out = {"users": [str(u) for u in d["test_topk_user_ids"]],
           "coverage": len(np.unique(topk)) / d["n_items_catalog"],
           "hr10": d["test"]["hit@10"]}
    for lvl in LEVELS:
        out[f"ild_{lvl}_per_user"] = per_user_ild(topk, units[lvl])
        out[f"ild_{lvl}"] = float(out[f"ild_{lvl}_per_user"].mean())
        out[f"nocat_{lvl}"] = float(zero_rows[lvl][topk].mean())
    return out, topk


def msd(xs):
    xs = np.asarray(xs, float)
    return float(xs.mean()), (float(xs.std(ddof=1)) if len(xs) > 1 else float("nan"))


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    lines, out = [], {"meta": {}, "groups": {}, "h4": {}}

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    mats, dims = category_matrices()
    units, zero_rows = {}, {}
    for lvl in LEVELS:
        units[lvl], zero_rows[lvl] = unit_rows(mats[lvl])
    say(f"ILD@10  {time.strftime('%Y-%m-%d %H:%M:%S')}  (analyze_ild.py)")
    say("category vocabularies on Pure: " + ", ".join(f"{k} {v}" for k, v in dims.items()))
    out["meta"]["category_dims"] = dims

    # ------------------------------------------------ groups (coverage-table rows)
    say("\n== ILD@10 by model (mean +- sd over seeds). nocat = share of top-10 slots whose item "
        "has no category at that level ==")
    say(f"  {'model':<22} {'seeds':>5}  {'ILD first-level':>16} {'nocat':>6}  {'ILD second-level':>16} "
        f"{'nocat':>6}  {'ILD tag':>16} {'nocat':>6}  {'Coverage@10':>14}  {'HR@10':>7}")
    per_user_store = {}
    checked = False
    for label, directory, pat, seeds in GROUPS:
        rows = []
        for s in seeds:
            path = find(directory, pat, s)
            d = load_json(path) if path else None
            if d is None:
                say(f"  {label:<22} seed {s}: no top-k dump found ({pat})")
                continue
            r, topk = run_diversity(d, units, zero_rows)
            if not checked:   # vectorised ILD == reference implementation
                ref = intra_list_diversity(topk, mats["first_level"])
                assert abs(ref - r["ild_first_level"]) < 1e-5, (ref, r["ild_first_level"])
                say(f"  (check: vectorised ILD {r['ild_first_level']:.6f} == diversity.intra_list_diversity "
                    f"{ref:.6f} on {os.path.basename(path)})")
                checked = True
            r["file"] = os.path.basename(path)
            r["seed"] = s
            rows.append(r)
            if label in H4_MODELS or label == "CAFREC-NP":
                per_user_store[(label, s)] = r
        if not rows:
            continue
        g = {"seeds": [r["seed"] for r in rows], "files": [r["file"] for r in rows]}
        for key in ([f"ild_{l}" for l in LEVELS] + [f"nocat_{l}" for l in LEVELS] + ["coverage", "hr10"]):
            g[key] = msd([r[key] for r in rows])
        out["groups"][label] = g
        cell = lambda k: f"{g[k][0]:.4f}+-{g[k][1]:.4f}" if len(rows) > 1 else f"{g[k][0]:.4f}        "
        say(f"  {label:<22} {len(rows):>5}  {cell('ild_first_level'):>16} {g['nocat_first_level'][0]:>6.3f}  "
            f"{cell('ild_second_level'):>16} {g['nocat_second_level'][0]:>6.3f}  {cell('ild_kuairand_tag'):>16} "
            f"{g['nocat_kuairand_tag'][0]:>6.3f}  {cell('coverage'):>14}  {g['hr10'][0]:.4f}")

    # ------------------------------------------------ H4
    users = sorted(per_user_store[("CAFREC", ABLATION[0])]["users"])
    masks, meta = cohorts(users)
    index = {u: i for i, u in enumerate(users)}
    n_hi, n_lo = int(masks["high drift"].sum()), int(masks["low drift"].sum())
    n_eq = min(n_hi, n_lo)
    rng = np.random.default_rng(BOOT_SEED)
    samples = {"all": np.flatnonzero(masks["all"])}
    for coh in ("high drift", "low drift"):
        samples[coh] = np.sort(rng.choice(np.flatnonzero(masks[coh]), size=n_eq, replace=False))
    full = {coh: np.flatnonzero(masks[coh]) for coh in ("high drift", "low drift")}
    say(f"\n== H4 (registered): ILD@10, context gate vs static fusion, 3 ablation seeds {ABLATION} ==")
    say(f"users {len(users):,}; high drift {n_hi:,}, low drift {n_lo:,}; equal-size sample {n_eq:,} per "
        f"drift stratum (one sample, shared by all models and seeds); 'all' uses every user")
    out["h4"]["strata_sizes"] = {"users": len(users), "high_drift": n_hi, "low_drift": n_lo, "sample": n_eq}

    def matrix(label, lvl):
        m = np.full((len(ABLATION), len(users)), np.nan)
        for j, s in enumerate(ABLATION):
            r = per_user_store[(label, s)]
            idx = np.array([index[u] for u in r["users"]])
            m[j, idx] = r[f"ild_{lvl}_per_user"]
        assert not np.isnan(m).any(), f"{label} {lvl}: users missing"
        return m

    for lvl in LEVELS:
        mats_h4 = {lab: matrix(lab, lvl) for lab in H4_MODELS + ("CAFREC-NP",)}
        tag = "PRIMARY" if lvl == "first_level" else "sensitivity"
        say(f"\n-- {lvl} ({tag}) --")
        say("  descriptive ILD@10 (mean +- sd over seeds of the per-seed sample mean):")
        desc = {}
        for lab in H4_MODELS + ("CAFREC-NP",):
            row = {}
            for coh, idx in samples.items():
                row[coh] = msd(mats_h4[lab][:, idx].mean(axis=1))
            desc[lab] = row
            say(f"    {lab:<19} all {row['all'][0]:.4f}+-{row['all'][1]:.4f}   high drift "
                f"{row['high drift'][0]:.4f}+-{row['high drift'][1]:.4f}   low drift "
                f"{row['low drift'][0]:.4f}+-{row['low drift'][1]:.4f}")
        tests = []
        for alt in ("Static scalar gate", "Concatenation"):
            for sampling, idx_map in (("sample", samples), ("full stratum", full)):
                diffs = {}
                for coh, idx in idx_map.items():
                    A, B = mats_h4["CAFREC"][:, idx], mats_h4[alt][:, idx]
                    d = A.mean(0) - B.mean(0)
                    diffs[coh] = d
                    lo, hi = boot_ci(d, rng)
                    sign = int(sum((A[j] - B[j]).mean() > 0 for j in range(len(ABLATION))))
                    t = {"alt": alt, "level": lvl, "sampling": sampling, "cohort": coh, "n_users": len(idx),
                         "delta": float(d.mean()), "rel_pct": float(100 * d.mean() / B.mean()),
                         "ci": [lo, hi], "p": wilcoxon_normal(d), "seeds_cafrec_higher": sign}
                    tests.append(t)
                    say(f"  CAFREC vs {alt:<19} {sampling:<12} {coh:<10} n={len(idx):>6,}  d={t['delta']:+.5f} "
                        f"({t['rel_pct']:+.2f}%) CI[{lo:+.5f},{hi:+.5f}] p={t['p']:.3g}  "
                        f"CAFREC higher in {sign}/{len(ABLATION)} seeds")
                dh, dl = diffs["high drift"], diffs["low drift"]
                bh = np.array([dh[rng.integers(0, len(dh), len(dh))].mean()
                               - dl[rng.integers(0, len(dl), len(dl))].mean() for _ in range(N_BOOT)])
                mg = {"alt": alt, "level": lvl, "sampling": sampling, "cohort": "high minus low drift",
                      "margin": float(dh.mean() - dl.mean()),
                      "ci": [float(np.quantile(bh, .025)), float(np.quantile(bh, .975))],
                      "p_mannwhitney": float(mannwhitneyu(dh, dl).pvalue)}
                tests.append(mg)
                say(f"  CAFREC vs {alt:<19} {sampling:<12} margin (high - low drift) {mg['margin']:+.5f} "
                    f"CI[{mg['ci'][0]:+.5f},{mg['ci'][1]:+.5f}] Mann-Whitney p={mg['p_mannwhitney']:.3g}")
        out["h4"][lvl] = {"descriptive": desc, "tests": tests}

    # ------------------------------------------------ every run with top-k lists
    if "--skip-all-runs" not in sys.argv:
        recs = []
        paths = sorted(glob.glob(os.path.join(RES_MODAL, "*.json")) + glob.glob(os.path.join(RES_LOCAL, "*.json")))
        for p in paths:
            d = load_json(p)
            # the category matrices cover KuaiRand-Pure video ids only
            if d is None or not str(d.get("dataset", "")).startswith("kuairand_pure"):
                continue
            r, _ = run_diversity(d, units, zero_rows)
            recs.append({"dir": os.path.basename(os.path.dirname(p)), "file": os.path.basename(p),
                         "model": d.get("model"), "seed": d.get("seed"),
                         "train_batch_size": (d.get("config") or {}).get("train_batch_size"),
                         "n_users": len(r["users"]), "hr10": r["hr10"], "coverage": r["coverage"],
                         **{f"ild_{l}": r[f"ild_{l}"] for l in LEVELS},
                         **{f"nocat_{l}": r[f"nocat_{l}"] for l in LEVELS}})
        csv_path = os.path.join(HERE, "results", f"ild_all_runs_{DATE}.csv")
        pd.DataFrame(recs).to_csv(csv_path, index=False)
        say(f"\nILD for all {len(recs)} result JSONs with top-k lists -> {os.path.basename(csv_path)}")

    say(f"\nwall {time.time() - t0:.0f}s")
    with open(os.path.join(HERE, "results", f"ild_{DATE}.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "results", f"ild_{DATE}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
