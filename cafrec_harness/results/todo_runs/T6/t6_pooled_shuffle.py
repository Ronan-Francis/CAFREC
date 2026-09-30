"""T6 (TODO.md): ten-seed pooled shuffled-context estimate. Fills D4 (05_results.tex, RES-14).

Per-seed delta = mean over the 22,912 test users of (shuffled - CAFREC-NP), NDCG@10 and HR@10:
  three ablation seeds {2020, 2021, 403092}: shuffled_ctx_s* vs noprof_topk_s* (3-seed reference);
  seven headline seeds {42, ..., 2048}:      gpu_shuffled_ctx_seed* vs noprof_ms_s* (Table IV).
Pooled over the ten per-seed deltas: mean delta as a percent of the mean CAFREC-NP value of the
ten reference runs; seed-level t 95% CI (9 degrees of freedom), in absolute units and as a percent;
exact two-sided Wilcoxon signed-rank p over the ten deltas; number of seeds below zero.

Gates (IV-F protocol via analyze_plan_hypotheses.compare, rng seed 20260917): the seven-seed batch
reproduces NDCG@10 -0.0009 and HR@10 -0.0014, the three-seed batch -0.0001 and -0.0002.

Run check (TODO step 2): device and batch size of all twenty runs, read from each file's
resolved `config` block where it exists; where it does not, the value follows from the launch
path documented in PROJECT_LOG (Cycle 10: Modal runs of *_ctx datasets before 2026-09-16 took
the `dataset != "kuairand_pure"` branch of modal_run.py, train_batch_size 512).

Usage (from cafrec_harness/):  <recbole env python> results/todo_runs/T6/t6_pooled_shuffle.py
Writes results/todo_runs/T6/t6_results.json, t6_runs.csv and t6_summary.txt.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPO = os.path.dirname(HARNESS)
sys.path.insert(0, HARNESS)
os.chdir(HARNESS)

import analyze_plan_hypotheses as aph  # noqa: E402
from analyze_plan_hypotheses import ABLATION, HEADLINE, Runs, compare  # noqa: E402

MODAL = os.path.join(HARNESS, "results", "modal")
aph.RES = MODAL
aph.PATS.update({"Shuffled ctx [head]": ["CAFREC_kuairand_pure_ctx_gpu_shuffled_ctx_seed{s}_*.json"]})
ARMS = {"head": ("Shuffled ctx [head]", "CAFREC-NP", HEADLINE),
        "abl": ("Shuffled ctx", "CAFREC-NP [abl]", ABLATION)}
METRICS = ("ndcg@10", "hit@10")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def run_rows():
    rows = []
    for batch, (shuf, ref, seeds) in ARMS.items():
        for name in (shuf, ref):
            for s in seeds:
                path = sorted(glob.glob(os.path.join(MODAL, aph.PATS[name][0].format(s=s))))[-1]
                d = json.load(open(path))
                cfg = d.get("config") or {}
                rel = os.path.relpath(path, REPO).replace("\\", "/")
                bs = cfg.get("train_batch_size")
                rows.append({
                    "result_file": rel, "arm": name, "batch": batch, "seed": s,
                    "commit": git("log", "--diff-filter=A", "--format=%h", "-1", "--", rel) or "untracked",
                    "batch_size": bs if bs is not None else 512,
                    "batch_size_source": "config block" if bs is not None
                    else "inferred: pre-fix Modal *_ctx branch (PROJECT_LOG Cycle 10)",
                    "device": "Modal A10G" + (" (config: cuda)" if d.get("device") == "cuda" else ""),
                    "date": path.rsplit("_", 1)[-1].split(".")[0],
                    "test_hit@10": d["test"]["hit@10"], "test_ndcg@10": d["test"]["ndcg@10"],
                })
    return rows


def main():
    t0 = time.time()
    probe = aph._load_json("CAFREC-NP", HEADLINE[0])
    users = sorted(str(u) for u in probe["test_user_ids"])
    runs = Runs(users)
    mask = np.ones(len(users), bool)

    # gates: per-batch pooled per-user comparisons (IV-F protocol)
    rng = np.random.default_rng(aph.BOOT_SEED)
    batch_stats = {}
    for batch, (shuf, ref, seeds) in ARMS.items():
        batch_stats[batch] = {m: {k: v for k, v in compare(runs, shuf, ref, seeds, m, mask, rng).items()
                                  if k != "_diffs"} for m in METRICS}
    gate = {
        "head_ndcg": round(batch_stats["head"]["ndcg@10"]["delta"], 4),
        "head_hr": round(batch_stats["head"]["hit@10"]["delta"], 4),
        "abl_ndcg": round(batch_stats["abl"]["ndcg@10"]["delta"], 4),
        "abl_hr": round(batch_stats["abl"]["hit@10"]["delta"], 4),
        "expected": {"head_ndcg": -0.0009, "head_hr": -0.0014, "abl_ndcg": -0.0001, "abl_hr": -0.0002},
    }
    gate["passed"] = all(gate[k] == v for k, v in gate["expected"].items())

    # per-seed deltas and the ten-seed pool
    per_seed, pooled = [], {}
    for m in METRICS:
        deltas, refs = [], []
        for batch, (shuf, ref, seeds) in ARMS.items():
            A, B = runs.get(shuf, seeds)[m], runs.get(ref, seeds)[m]
            for j, s in enumerate(seeds):
                d = float(A[j].mean() - B[j].mean())
                deltas.append(d)
                refs.append(float(B[j].mean()))
                per_seed.append({"metric": m, "batch": batch, "seed": s, "shuffled": float(A[j].mean()),
                                 "cafrec_np": float(B[j].mean()), "delta": d})
        d, refs = np.array(deltas), np.array(refs)
        n = len(d)
        mean, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
        tcrit = stats.t.ppf(0.975, n - 1)
        lo, hi = mean - tcrit * se, mean + tcrit * se
        base = refs.mean()
        pooled[m] = {
            "n_seeds": n, "mean_delta": float(mean), "ref_mean": float(base),
            "mean_delta_pct": float(100 * mean / base),
            "t_ci95": [float(lo), float(hi)], "t_ci95_pct": [float(100 * lo / base), float(100 * hi / base)],
            "t_df": n - 1, "t_p": float(stats.ttest_1samp(d, 0.0).pvalue),
            "wilcoxon_exact_p": float(stats.wilcoxon(d, method="exact").pvalue),
            "seeds_below_zero": int((d < 0).sum()),
        }

    rows = run_rows()
    with open(os.path.join(HERE, "t6_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    check = {
        "runs": len(rows),
        "batch_sizes": sorted({r["batch_size"] for r in rows}),
        "batch_size_recorded": sum(r["batch_size_source"] == "config block" for r in rows),
        "devices": sorted({r["device"].split(" (")[0] for r in rows}),
        "same_batch_and_device": len({r["batch_size"] for r in rows}) == 1
        and len({r["device"].split(" (")[0] for r in rows}) == 1,
    }
    res = {"task": "T6", "markers": ["D4", "D4"], "gate": gate, "batch_stats": batch_stats,
           "per_seed": per_seed, "pooled": pooled, "run_check": check,
           "provenance": {"commit": git("rev-parse", "HEAD"), "device": f"{platform.node()} CPU",
                          "modal_run_id": None, "wall_s": round(time.time() - t0, 1),
                          "date": time.strftime("%Y-%m-%d %H:%M:%S")}}
    with open(os.path.join(HERE, "t6_results.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    p = pooled
    summary = "T6: " + "; ".join(
        f"{m} {p[m]['mean_delta_pct']:+.2f}% (d={p[m]['mean_delta']:+.5f}, t95 CI "
        f"[{p[m]['t_ci95'][0]:+.5f},{p[m]['t_ci95'][1]:+.5f}] = [{p[m]['t_ci95_pct'][0]:+.2f}%,"
        f"{p[m]['t_ci95_pct'][1]:+.2f}%], exact W p={p[m]['wilcoxon_exact_p']:.3g}, "
        f"{p[m]['seeds_below_zero']}/10 below zero)" for m in METRICS) + \
        f"; gate {'PASS' if gate['passed'] else 'FAIL'}; runs {check}"
    with open(os.path.join(HERE, "t6_summary.txt"), "w") as fh:
        fh.write(summary + "\n")
    print(json.dumps(res, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
