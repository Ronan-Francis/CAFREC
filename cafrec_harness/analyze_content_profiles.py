"""Content-bearing long-term profiles vs CAFREC-NP, same device (2026-09-17).

Conditions (local_run.py queue `content_profiles`, results/local/, batch 512):
  CAFREC-NP (local)  ablation=no_profiler                       tag np512
  cat_beyond         first-level category histogram, beyond-window  tag catbeyond512
  cap_beyond         mean caption embedding, beyond-window          tag capbeyond512
  cap_all            mean caption embedding, all training rows      tag capall512
(build_content_profiles.py; sidecar JSONs next to each cache in data/profiles/.)

Statistics are analyze_plan_hypotheses.compare: per-user metric averaged over seeds,
two-sided Wilcoxon signed-rank (normal approximation), paired bootstrap 95% CI
(2,000 user resamples), per-seed significance / sign counts, and seed-level exact
Wilcoxon + paired t when there are >= 5 seeds. Cohorts are analyze_plan_hypotheses.cohorts.

Cuts, in order:
  1. dense cohort (>= 20 rows)  PRIMARY, pre-stated in the Discussion; Holm over
     3 profiles x {HR@10, NDCG@10}
  2. lowest activity quartile, NDCG@10: cap_all vs CAFREC-NP (local), and cap_all vs the
     trainable stand-in, which exists only as Modal GPU runs -> cross-device, indicative
  3. all users
  4. session opener (prefix length 0) vs within-session
  5. Coverage@10 and ILD@10 (first-level primary, second-level and tag sensitivity)
Also: profile coverage (share of users with a non-zero vector) by cohort, and each
profile vs CAFREC-NP in every activity quartile.

Usage (from cafrec_harness/):  <env python> analyze_content_profiles.py [--seeds 2020,2021,403092]
Writes results/content_profiles_<date>.txt and .json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from analyze_ild import LEVELS, category_matrices, per_user_ild, unit_rows
from analyze_plan_hypotheses import BOOT_SEED, compare, cohorts, holm

HERE = os.path.dirname(os.path.abspath(__file__))
RES_LOCAL = os.path.join(HERE, "results", "local")
RES_MODAL = os.path.join(HERE, "results", "modal")
PROFILE_ROWS = os.path.join(HERE, "..", "data", "content_pure", "profile_user_rows.csv")
DATE = time.strftime("%Y-%m-%d")

REF = "CAFREC-NP (local)"
PROFILES = ("cat_beyond", "cap_beyond", "cap_all")
PATS = {
    REF:          [(RES_LOCAL, "CAFREC_kuairand_pure_ctx_np512_seed{s}.json")],
    "cat_beyond": [(RES_LOCAL, "CAFREC_kuairand_pure_ctx_catbeyond512_seed{s}.json")],
    "cap_beyond": [(RES_LOCAL, "CAFREC_kuairand_pure_ctx_capbeyond512_seed{s}.json")],
    "cap_all":    [(RES_LOCAL, "CAFREC_kuairand_pure_ctx_capall512_seed{s}.json")],
    # Modal GPU (different device): the H2 reference and the template-profile reading
    "Stand-in (Modal)": [(RES_MODAL, "CAFREC_kuairand_pure_ctx_standin_s{s}_seed{s}_*.json"),
                         (RES_MODAL, "CAFREC_kuairand_pure_ctx_standin_seed{s}_*.json")],
    "CAFREC template (Modal)":  [(RES_MODAL, "CAFREC_kuairand_pure_ctx_bge_topk_s{s}_seed{s}_*.json")],
    "CAFREC-NP (Modal)":        [(RES_MODAL, "CAFREC_kuairand_pure_ctx_noprof_topk_s{s}_seed{s}_*.json")],
    "Hashing (Modal)":          [(RES_MODAL, "CAFREC_kuairand_pure_ctx_hash_s{s}_seed{s}_*.json")],
}


def _path(name, seed):
    for directory, pat in PATS[name]:
        hits = sorted(glob.glob(os.path.join(directory, pat.format(s=seed))))
        if hits:
            return hits[-1]
    return None


class Runs:
    """Same interface as analyze_plan_hypotheses.Runs, with per-condition directories."""

    def __init__(self, users):
        self.users = users
        self.index = {u: i for i, u in enumerate(users)}
        self.cache, self.raw = {}, {}

    def load(self, name, seed):
        if (name, seed) not in self.raw:
            path = _path(name, seed)
            if path is None:
                raise FileNotFoundError(f"{name} seed {seed}")
            with open(path) as fh:
                d = json.load(fh)
            d["_file"] = os.path.basename(path)
            self.raw[(name, seed)] = d
        return self.raw[(name, seed)]

    def get(self, name, seeds):
        key = (name, tuple(seeds))
        if key in self.cache:
            return self.cache[key]
        ranks = np.full((len(seeds), len(self.users)), np.nan)
        for j, s in enumerate(seeds):
            d = self.load(name, s)
            for u, r in zip(d["test_user_ids"], d["test_ranks"]):
                i = self.index.get(str(u))
                if i is not None and r is not None:
                    ranks[j, i] = r
        assert not np.isnan(ranks).any(), f"{name}: users missing from dump"
        ink = ranks <= 10
        m = {"hit@10": ink.astype(float),
             "ndcg@10": np.where(ink, 1.0 / np.log2(ranks + 1.0), 0.0),
             "mrr@10": np.where(ink, 1.0 / ranks, 0.0)}
        self.cache[key] = m
        return m


def fmt(r):
    s = (f"  {r['a']:<11} vs {r['b']:<24} {r['cohort']:<15} {r['metric']:<8} n={r['n_users']:>6,} "
         f"d={r['delta']:+.5f} ({r['rel_pct']:+.1f}%) CI[{r['ci'][0]:+.5f},{r['ci'][1]:+.5f}] p={r['p']:.3g}")
    if "p_holm" in r:
        s += f" holm={r['p_holm']:.3g}"
    s += f" sig {r['sig']}/{r['seeds']} sign {r['sign']}/{r['seeds']}"
    if "seed_p_ttest" in r:
        s += f" | seed-level p_t={r['seed_p_ttest']:.3g} p_W={r['seed_p_wilcoxon_exact']:.3g}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="2020,2021,403092")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    rng = np.random.default_rng(BOOT_SEED)
    lines, out = [], {"seeds": seeds}

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    probe = Runs([]).load(REF, seeds[0])
    users = sorted(str(u) for u in probe["test_user_ids"])
    runs = Runs(users)
    masks, meta = cohorts(users)
    say(f"CONTENT PROFILES vs CAFREC-NP  {time.strftime('%Y-%m-%d %H:%M:%S')}  seeds {seeds}")
    say(f"users {len(users):,}; activity quartile cuts {meta['quartile_cuts']}")
    say("  " + "  ".join(f"{k} {int(v.sum()):,}" for k, v in masks.items()))
    say("files:")
    for name in (REF,) + PROFILES:
        for s in seeds:
            d = runs.load(name, s)
            cfg = d.get("config", {})
            say(f"  {name:<18} seed {s:<6} {d['_file']:<58} device={d.get('device')} "
                f"batch={cfg.get('train_batch_size')} ablation={cfg.get('ablation')} "
                f"profile_dim={cfg.get('profile_dim')} profile={os.path.basename(str(cfg.get('llm_profile_path')))} "
                f"HR@10={d['test']['hit@10']:.4f} wall={d.get('wall_seconds')}s")
            assert cfg.get("train_batch_size") == 512, f"{name} seed {s} not at batch 512"

    # ------------------------------------------------ profile coverage by cohort
    pr = pd.read_csv(PROFILE_ROWS, dtype={"user_id": str}).set_index("user_id").loc[users]
    say("\n== Profile coverage: share of users with a non-zero profile vector ==")
    cov_rows = {}
    for p in PROFILES:
        nz = pr[f"nz_{p}"].to_numpy().astype(bool)
        cov_rows[p] = {k: float(nz[masks[k]].mean()) for k in ("all", "sparse (<20)", "dense (>=20)",
                                                             "Q1 activity", "Q2 activity", "Q3 activity",
                                                             "Q4 activity", "short session", "longer session")}
        say(f"  {p:<11} " + "  ".join(f"{k} {100 * v:.1f}%" for k, v in cov_rows[p].items()))
    out["profile_coverage"] = cov_rows

    # ------------------------------------------------ accuracy cuts
    sections = []

    def block(title, rows, holm_rows=None):
        if holm_rows:
            for r, a in zip(holm_rows, holm(np.array([r["p"] for r in holm_rows]))):
                r["p_holm"] = float(a)
        say(f"\n== {title} ==")
        for r in rows:
            say(fmt(r))
        sections.append({"title": title, "rows": rows})

    rows = [compare(runs, p, REF, seeds, m, masks["dense (>=20)"], rng) | {"cohort": "dense (>=20)"}
            for p in PROFILES for m in ("hit@10", "ndcg@10", "mrr@10")]
    block("1. PRIMARY: dense cohort (>= 20 rows), profile vs CAFREC-NP (local); Holm over 3 profiles x HR/NDCG",
          rows, [r for r in rows if r["metric"] in ("hit@10", "ndcg@10")])

    rows = [compare(runs, "cap_all", REF, seeds, "ndcg@10", masks["Q1 activity"], rng) | {"cohort": "Q1 activity"}]
    rows += [compare(runs, p, REF, seeds, "ndcg@10", masks["Q1 activity"], rng) | {"cohort": "Q1 activity"}
             for p in ("cat_beyond", "cap_beyond")]
    block("2a. Lowest activity quartile (<= 12 rows), NDCG@10, vs CAFREC-NP (local, same device)", rows)
    rows = []
    for coh in ("Q1 activity", "all"):
        try:
            rows.append(compare(runs, "cap_all", "Stand-in (Modal)", seeds, "ndcg@10", masks[coh], rng)
                        | {"cohort": coh})
            for a in ("CAFREC-NP (Modal)", "CAFREC template (Modal)", "Hashing (Modal)"):
                rows.append(compare(runs, a, "Stand-in (Modal)", seeds, "ndcg@10", masks[coh], rng)
                            | {"cohort": coh})
            # does the template profile help against no profile at all, on one device?
            rows.append(compare(runs, "CAFREC template (Modal)", "CAFREC-NP (Modal)", seeds, "ndcg@10",
                                masks[coh], rng) | {"cohort": coh})
        except FileNotFoundError as e:
            say(f"  (stand-in comparison skipped: {e})")
            break
    block("2b. H2 INDICATIVE, cross-device: cap_all (local CPU) vs trainable stand-in (Modal GPU); "
          "the Modal-vs-Modal rows are same-device references for the frozen profiles", rows)

    rows = [compare(runs, p, REF, seeds, m, masks["all"], rng) | {"cohort": "all"}
            for p in PROFILES for m in ("hit@10", "ndcg@10", "mrr@10")]
    block("3. All users", rows)

    rows = [compare(runs, p, REF, seeds, m, masks[coh], rng) | {"cohort": coh}
            for p in PROFILES for coh in ("short session", "longer session") for m in ("hit@10", "ndcg@10")]
    block("4. Session opener (short session, prefix length 0) vs within-session (longer session)", rows)

    rows = [compare(runs, p, REF, seeds, "ndcg@10", masks[coh], rng) | {"cohort": coh}
            for p in PROFILES for coh in ("sparse (<20)", "Q1 activity", "Q2 activity", "Q3 activity", "Q4 activity")]
    block("Supplementary: NDCG@10 by sparse cohort and activity quartile, vs CAFREC-NP (local)", rows)

    # ------------------------------------------------ coverage and ILD
    mats, _ = category_matrices()
    units, zero_rows = {}, {}
    for lvl in LEVELS:
        units[lvl], zero_rows[lvl] = unit_rows(mats[lvl])
    say("\n== 5. Coverage@10 and ILD@10 (mean +- sd over seeds) ==")
    div = {}
    for name in (REF,) + PROFILES:
        vals = {k: [] for k in ["coverage"] + [f"ild_{l}" for l in LEVELS]}
        for s in seeds:
            d = runs.load(name, s)
            topk = np.asarray(d["test_topk_items"], dtype=np.int64)
            vals["coverage"].append(len(np.unique(topk)) / d["n_items_catalog"])
            for lvl in LEVELS:
                vals[f"ild_{lvl}"].append(float(per_user_ild(topk, units[lvl]).mean()))
        div[name] = {k: [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")]
                     for k, v in vals.items()}
        say(f"  {name:<18} coverage {div[name]['coverage'][0]:.4f}+-{div[name]['coverage'][1]:.4f}  "
            + "  ".join(f"ILD {l} {div[name][f'ild_{l}'][0]:.4f}+-{div[name][f'ild_{l}'][1]:.4f}" for l in LEVELS))
    out["diversity"] = div

    for sec in sections:
        for r in sec["rows"]:
            r.pop("_diffs", None)
    out["sections"] = sections
    say(f"\nwall {time.time() - t0:.0f}s")
    with open(os.path.join(HERE, "results", f"content_profiles_{DATE}.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "results", f"content_profiles_{DATE}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
