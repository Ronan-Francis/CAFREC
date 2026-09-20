"""Tuned SASRec vs tuned CAFREC-NP, and the policy-flag ablation (local GPU runs).

Reads results/local_gpu/, written by `local_run.py --queue gpu_grid` then
`--queue gpu_tuned` (and optionally `--queue gpu_converge`). Every run here used
the same device (RTX 2060 SUPER) and batch 512, so no arm pairs with Modal or CPU.

  1. Grid: 16 points per model, seed 403092, validation NDCG@10 (selection) and
     test NDCG@10 (reported only, never used to choose).
  2. Tuned CAFREC-NP vs tuned SASRec, 7 headline seeds: all users, session openers
     (prefix length 0), within-session users, and the opener-minus-within interaction.
  3. Policy-flag ablation: tuned CAFREC-NP with the five-feature x_ctx (no
     prefix_policy_flag) vs tuned CAFREC-NP, same seeds and cohorts.
  4. Coverage@10 over all test users for the tuned pair (descriptive).

Holm-Bonferroni is applied over all six tests of family 2 (HR/NDCG x all/openers/within)
and within family 3 (HR/NDCG, all users). Cohort definitions and statistics are those of
analyze_plan_hypotheses.py.

Usage:  .venv/Scripts/python analyze_tuned_gpu.py
Writes results/tuned_gpu_<date>.txt and .json.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy.stats import wilcoxon as sp_wilcoxon

import analyze_plan_hypotheses as aph
from analyze_plan_hypotheses import HEADLINE, Runs, cohorts, compare, holm
from local_run import GRID16, GRID_SEED

GPU = os.path.join(aph.HERE, "results", "local_gpu")
aph.RES = GPU                                   # _load_json reads from here
aph.PATS.update({
    "SASRec (tuned)":     ["SASRec_kuairand_pure_tuned16_sasrec_gs*_seed{s}.json"],
    "CAFREC-NP (tuned)":  ["CAFREC_kuairand_pure_ctx_tuned16_np_gs*_seed{s}.json"],
    "NP no policy flag":  ["CAFREC_kuairand_pure_ctx_tuned16_np_nopolicy_gs*_seed{s}.json"],
})
OPEN, WITHIN = "short session", "longer session"
METRICS = ("hit@10", "ndcg@10")


def grid_table():
    rows = []
    for arm, prefix in (("sasrec", "SASRec_kuairand_pure"), ("np", "CAFREC_kuairand_pure_ctx")):
        for i, cfg in enumerate(GRID16):
            p = os.path.join(GPU, f"{prefix}_gs16_{arm}_{i:02d}_seed{GRID_SEED}.json")
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            rows.append({"arm": arm, "point": i, **cfg, "valid_ndcg": d["valid_best"]["ndcg@10"],
                         "test_ndcg": d["test"]["ndcg@10"], "wall_s": d.get("wall_seconds")})
    return rows


def converge_rows():
    out = []
    for f in sorted(os.listdir(GPU)):
        if "_converge30_" in f:
            d = json.load(open(os.path.join(GPU, f)))
            out.append({"file": f, "valid_ndcg": d["valid_best"]["ndcg@10"],
                        "test_ndcg": d["test"]["ndcg@10"], "wall_s": d.get("wall_seconds")})
    return out


def fmt(r, label):
    seed = (f" | seed-level W p={r['seed_p_wilcoxon_exact']:.3g} t p={r['seed_p_ttest']:.3g}"
            if "seed_p_ttest" in r else "")
    holm_txt = f" holm={r['p_holm']:.3g}" if "p_holm" in r else ""
    return (f"  {label:<34} {r['metric']:<8} d={r['delta']:+.5f} ({r['rel_pct']:+.1f}%) "
            f"CI[{r['ci'][0]:+.5f},{r['ci'][1]:+.5f}] p={r['p']:.3g}{holm_txt} "
            f"sign {r['sign']}/{r['seeds']} sig {r['sig']}/{r['seeds']}{seed}  (n={r['n_users']:,})")


def main():
    t0 = time.time()
    rng = np.random.default_rng(aph.BOOT_SEED)
    lines = [f"Local GPU runs in {GPU}", ""]

    grid = grid_table()
    lines.append("== 1. Grid (seed 403092, batch 512; selection on validation NDCG@10) ==")
    lines.append("  arm    pt  dropout  L/H  lr      wd      valid NDCG  test NDCG  wall s")
    best = {}
    for r in grid:
        if r["arm"] not in best or r["valid_ndcg"] > best[r["arm"]]["valid_ndcg"]:
            best[r["arm"]] = r
    for r in grid:
        star = " *" if best.get(r["arm"]) is r else ""
        lines.append(f"  {r['arm']:<6} {r['point']:02d}  {r['hidden_dropout_prob']:<7}  "
                     f"{r['n_layers']}/{r['n_heads']}  {r['learning_rate']:<7} {r['weight_decay']:<7} "
                     f"{r['valid_ndcg']:.4f}      {r['test_ndcg']:.4f}     {r['wall_s']}{star}")
    for c in converge_rows():
        lines.append(f"  convergence check (30 epochs, patience 5): {c['file']}  "
                     f"valid {c['valid_ndcg']:.4f} test {c['test_ndcg']:.4f} wall {c['wall_s']}s")

    probe = aph._load_json("SASRec (tuned)", HEADLINE[0])
    users = sorted(str(u) for u in probe["test_user_ids"])
    runs = Runs(users)
    masks, _ = cohorts(users)

    # Holm over all six tests of the tuned comparison (3 cohorts x 2 metrics), chosen
    # 2026-09-20: the widest family, so the correction cannot be said to favour a result.
    fam2 = [compare(runs, "CAFREC-NP (tuned)", "SASRec (tuned)", HEADLINE, m, masks[c], rng) | {"cohort": c}
            for c in ("all", OPEN, WITHIN) for m in METRICS]
    for r, a in zip(fam2, holm(np.array([r["p"] for r in fam2]))):
        r["p_holm"] = float(a)
    within = []
    inter = []
    for m in METRICS:
        A, B = runs.get("CAFREC-NP (tuned)", HEADLINE)[m], runs.get("SASRec (tuned)", HEADLINE)[m]
        per_seed = (A - B)[:, masks[OPEN]].mean(1) - (A - B)[:, masks[WITHIN]].mean(1)
        inter.append({"metric": m, "interaction": float(per_seed.mean()),
                      "seed_p_W": float(sp_wilcoxon(per_seed, method="exact").pvalue),
                      "same_sign": int((np.sign(per_seed) == np.sign(per_seed.mean())).sum())})

    lines += ["", "== 2. Tuned CAFREC-NP vs tuned SASRec (7 headline seeds; Holm over all six tests) =="]
    for r in fam2 + within:
        lines.append(fmt(r, r["cohort"].replace("short session", "openers")
                            .replace("longer session", "within session")))
    for r in inter:
        lines.append(f"  opener-minus-within {r['metric']:<8} {r['interaction']:+.5f} "
                     f"seed-level exact W p={r['seed_p_W']:.3g} (same sign {r['same_sign']}/7)")

    fam3 = [compare(runs, "NP no policy flag", "CAFREC-NP (tuned)", HEADLINE, m, masks["all"], rng)
            | {"cohort": "all"} for m in METRICS]
    for r, a in zip(fam3, holm(np.array([r["p"] for r in fam3]))):
        r["p_holm"] = float(a)
    sub3 = [compare(runs, "NP no policy flag", "CAFREC-NP (tuned)", HEADLINE, m, masks[c], rng)
            | {"cohort": c} for c in (OPEN, WITHIN) for m in METRICS]
    lines += ["", "== 3. Policy-flag ablation: five-feature x_ctx vs six (tuned CAFREC-NP, 7 seeds) =="]
    for r in fam3 + sub3:
        lines.append(fmt(r, r["cohort"].replace("short session", "openers")
                            .replace("longer session", "within session")))

    lines += ["", "== 4. Coverage@10, all test users (descriptive) =="]
    cov = {}
    for name in ("SASRec (tuned)", "CAFREC-NP (tuned)", "NP no policy flag"):
        mu, sd, _ = aph.coverage(runs, name, HEADLINE, masks["all"], len(users), rng)
        cov[name] = [mu, sd]
        lines.append(f"  {name:<18} {mu:.3f} +- {sd:.3f}")

    lines.append(f"\nwall {time.time() - t0:.0f}s")
    text = "\n".join(lines)
    print(text)
    out = os.path.join(aph.HERE, "results", f"tuned_gpu_{time.strftime('%Y%m%d')}")
    with open(out + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    strip = lambda rs: [{k: v for k, v in r.items() if k != "_diffs"} for r in rs]
    with open(out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"grid": grid, "converge": converge_rows(), "tuned": strip(fam2 + within),
                   "interaction": inter, "policy": strip(fam3 + sub3), "coverage": cov}, fh, indent=2)
    print(f"-> {out}.txt / .json")


if __name__ == "__main__":
    main()
