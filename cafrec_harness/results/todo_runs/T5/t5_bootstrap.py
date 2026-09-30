"""T5 (TODO.md): bootstrap intervals for the policy-flag and four-feature ablations.

Fills D3a (05_results.tex, RES-7, three markers) and D3b (RES-15). CPU only, saved ranks.

Protocol (Sec. IV-F), via analyze_plan_hypotheses.compare, unchanged: each user's metric
is averaged over the seeds shared by both conditions, users are paired by the ORIGINAL
user id, and the 95% CI is a paired bootstrap with 2,000 user resamples (resampled jointly
across both models), numpy default_rng(20260917).

D3a  tuned CAFREC-NP without the policy flag (five-feature x_ctx) vs tuned CAFREC-NP,
     7 headline seeds, all users, HR@10 and NDCG@10 (results/local_gpu, RTX 2060 SUPER).
     The call sequence of analyze_tuned_gpu.py is repeated (six tuned-comparison calls,
     then the two ablation calls) so the generator state, and therefore the interval,
     matches the logged results/tuned_gpu_20260920.txt.
     Also: the larger absolute bound of each interval as a percent of tuned CAFREC-NP.
D3b  CAFREC with the four core features (bge_r4) vs CAFREC (bge_topk), 3 ablation seeds,
     all users, NDCG@10 (results/modal, A10G). Fresh generator with the same seed.

Gates: policy flag HR@10 -0.8% and NDCG@10 -0.0%; four-feature NDCG@10 delta -0.00005.

Usage (from cafrec_harness/):  <recbole env python> results/todo_runs/T5/t5_bootstrap.py
Writes results/todo_runs/T5/t5_results.json and t5_summary.txt.
"""
from __future__ import annotations

import glob
import json
import os
import platform
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HARNESS)
os.chdir(HARNESS)

import analyze_plan_hypotheses as aph  # noqa: E402
from analyze_plan_hypotheses import ABLATION, HEADLINE, Runs, cohorts, compare  # noqa: E402

GPU = os.path.join(HARNESS, "results", "local_gpu")
MODAL = os.path.join(HARNESS, "results", "modal")
LOGGED = os.path.join(HARNESS, "results", "tuned_gpu_20260920.txt")
METRICS = ("hit@10", "ndcg@10")


def git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HARNESS, text=True).strip()


def strip(r):
    return {k: v for k, v in r.items() if k != "_diffs"}


def logged_policy_cis():
    """CIs of the two all-user policy-flag rows in the logged analysis output."""
    txt = open(LOGGED, encoding="utf-8").read().split("== 3.")[1]
    out = {}
    for m in METRICS:
        row = re.search(rf"^\s+all\s+{re.escape(m)}\s+d=(\S+) \(\S+\) CI\[(\S+),(\S+)\]", txt, re.M)
        out[m] = [float(row.group(2)), float(row.group(3))]
    return out


def d3a():
    aph.RES = GPU
    aph.PATS.update({
        "SASRec (tuned)":    ["SASRec_kuairand_pure_tuned16_sasrec_gs*_seed{s}.json"],
        "CAFREC-NP (tuned)": ["CAFREC_kuairand_pure_ctx_tuned16_np_gs*_seed{s}.json"],
        "NP no policy flag": ["CAFREC_kuairand_pure_ctx_tuned16_np_nopolicy_gs*_seed{s}.json"],
    })
    rng = np.random.default_rng(aph.BOOT_SEED)
    probe = aph._load_json("SASRec (tuned)", HEADLINE[0])
    users = sorted(str(u) for u in probe["test_user_ids"])
    runs = Runs(users)
    masks, _ = cohorts(users)
    # analyze_tuned_gpu.py family 2 (same order), only to advance the generator identically
    for c in ("all", "short session", "longer session"):
        for m in METRICS:
            compare(runs, "CAFREC-NP (tuned)", "SASRec (tuned)", HEADLINE, m, masks[c], rng)
    rows = [compare(runs, "NP no policy flag", "CAFREC-NP (tuned)", HEADLINE, m, masks["all"], rng)
            for m in METRICS]
    out = {}
    for r in rows:
        ref = float(runs.get("CAFREC-NP (tuned)", HEADLINE)[r["metric"]].mean())
        bound = max(abs(r["ci"][0]), abs(r["ci"][1]))
        out[r["metric"]] = strip(r) | {"reference_mean": ref, "larger_abs_bound": bound,
                                       "larger_abs_bound_pct_of_ref": 100 * bound / ref}
    logged = logged_policy_cis()
    gate = {
        "hr_rel_pct": round(out["hit@10"]["rel_pct"], 1), "ndcg_rel_pct": round(out["ndcg@10"]["rel_pct"], 1),
        "expected": {"hr_rel_pct": -0.8, "ndcg_rel_pct": -0.0},
        "logged_cis": logged,
        "ci_matches_logged": {m: bool(np.allclose(out[m]["ci"], logged[m], atol=5e-6)) for m in METRICS},
    }
    gate["passed"] = (gate["hr_rel_pct"] == -0.8 and gate["ndcg_rel_pct"] in (-0.0, 0.0)
                      and out["ndcg@10"]["rel_pct"] <= 0)
    worst = max(METRICS, key=lambda m: out[m]["larger_abs_bound_pct_of_ref"])
    return {"rows": out, "gate": gate, "larger_bound_overall_pct": out[worst]["larger_abs_bound_pct_of_ref"],
            "larger_bound_metric": worst, "n_users": len(users), "seeds": HEADLINE,
            "inputs": sorted(os.path.basename(sorted(glob.glob(os.path.join(GPU, aph.PATS[n][0].format(s=s))))[-1])
                             for s in HEADLINE for n in ("CAFREC-NP (tuned)", "NP no policy flag")),
            "device": "local-gpu (NVIDIA GeForce RTX 2060 SUPER), per the result files"}


def d3b():
    aph.RES = MODAL
    aph.PATS.update({"CAFREC R4 [abl]": ["CAFREC_kuairand_pure_ctx_bge_r4_s{s}_seed{s}_*.json"]})
    rng = np.random.default_rng(aph.BOOT_SEED)
    probe = aph._load_json("CAFREC [abl]", ABLATION[0])
    users = sorted(str(u) for u in probe["test_user_ids"])
    runs = Runs(users)
    mask = np.ones(len(users), bool)
    r = compare(runs, "CAFREC R4 [abl]", "CAFREC [abl]", ABLATION, "ndcg@10", mask, rng)
    gate = {"delta": r["delta"], "expected_delta": -0.00005,
            "passed": bool(round(r["delta"], 5) == -0.00005)}
    return {"row": strip(r), "gate": gate, "n_users": len(users), "seeds": ABLATION,
            "device": "Modal A10G (results/modal)"}


def main():
    t0 = time.time()
    res = {"task": "T5", "markers": ["D3a", "D3b"], "D3a": d3a(), "D3b": d3b(),
           "protocol": {"n_boot": aph.N_BOOT, "boot_seed": aph.BOOT_SEED,
                        "pairing": "original user id; per-user mean over shared seeds",
                        "p": "two-sided Wilcoxon signed-rank, normal approximation"},
           "provenance": {"commit": git_head(), "device": f"{platform.node()} CPU",
                          "modal_run_id": None, "wall_s": round(time.time() - t0, 1),
                          "date": time.strftime("%Y-%m-%d %H:%M:%S")}}
    with open(os.path.join(HERE, "t5_results.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    a, b = res["D3a"], res["D3b"]
    hr, nd = a["rows"]["hit@10"], a["rows"]["ndcg@10"]
    summary = (f"T5: D3a HR@10 {hr['rel_pct']:+.1f}% CI[{hr['ci'][0]:+.5f},{hr['ci'][1]:+.5f}], "
               f"NDCG@10 {nd['rel_pct']:+.1f}% CI[{nd['ci'][0]:+.5f},{nd['ci'][1]:+.5f}], larger bound "
               f"{a['larger_bound_overall_pct']:.2f}% of ref ({a['larger_bound_metric']}); gate "
               f"{'PASS' if a['gate']['passed'] else 'FAIL'}, CIs match log {a['gate']['ci_matches_logged']}. "
               f"D3b NDCG@10 d={b['row']['delta']:+.6f} CI[{b['row']['ci'][0]:+.5f},{b['row']['ci'][1]:+.5f}] "
               f"p={b['row']['p']:.3g}; gate {'PASS' if b['gate']['passed'] else 'FAIL'}")
    with open(os.path.join(HERE, "t5_summary.txt"), "w") as fh:
        fh.write(summary + "\n")
    print(json.dumps(res, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
