"""T9 helper: identify the result files behind Table VIII (tab:res:profile) rows.

The 7B-vs-bge row has two candidate bge-large references on the volume (the 08-15 bge_topk
set and the 08-14 prof_bge seed-2020 run). Each candidate pairing is run through the IV-F
protocol (analyze_plan_hypotheses.compare) and compared with the printed values; the
pairing that reproduces them is the one the manifest cites. No value in the paper changes.

Printed (05_results.tex, Table VIII):
  CAFREC vs CAFREC-NP (7 seeds)       HR -0.0006 [-0.0019, +0.0007] p 0.54; NDCG -0.0001 [-0.0008, +0.0005] p 0.78
  e5-mistral-7B vs bge-large (3)      HR -0.0005 [-0.0018, +0.0008] p 0.21; NDCG -0.0005 [-0.0011, +0.0002] p 0.21
  Hashing vs bge-large (3)            HR +0.0004 [-0.0009, +0.0017] p 0.60; NDCG +0.0003 [-0.0004, +0.0009] p 0.46
  Frozen bge vs trainable stand-in (3) HR +0.0077 [+0.0053, +0.0098] p 6.5e-15; NDCG +0.0054 [+0.0041, +0.0067] p 6.5e-11
Writes results/todo_runs/T9/t9_check_profile_rows.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HARNESS)
os.chdir(HARNESS)

import analyze_plan_hypotheses as aph  # noqa: E402
from analyze_plan_hypotheses import ABLATION, HEADLINE, Runs, compare  # noqa: E402

aph.PATS.update({
    "7B": ["CAFREC_kuairand_pure_ctx_prof7b_s{s}_seed{s}_*.json",
           "CAFREC_kuairand_pure_ctx_prof_7b_seed{s}_*.json"],
    "bge (prof_bge s2020 + bge_topk)": ["CAFREC_kuairand_pure_ctx_prof_bge_seed{s}_*.json",
                                        "CAFREC_kuairand_pure_ctx_bge_topk_s{s}_seed{s}_*.json"],
    "Hashing": ["CAFREC_kuairand_pure_ctx_hash_s{s}_seed{s}_*.json"],
})
PAIRS = [
    ("CAFREC", "CAFREC-NP", HEADLINE),
    ("7B", "CAFREC [abl]", ABLATION),
    ("7B", "bge (prof_bge s2020 + bge_topk)", ABLATION),
    ("Hashing", "CAFREC [abl]", ABLATION),
    ("CAFREC [abl]", "Stand-in", ABLATION),
]


def files(name, seeds):
    import glob
    out = []
    for s in seeds:
        for pat in aph.PATS[name]:
            hits = sorted(glob.glob(os.path.join(aph.RES, pat.format(s=s))))
            if hits:
                out.append(os.path.basename(hits[-1]))
                break
    return out


def main():
    probe = aph._load_json("CAFREC-NP", HEADLINE[0])
    users = sorted(str(u) for u in probe["test_user_ids"])
    runs = Runs(users)
    mask = np.ones(len(users), bool)
    out = []
    for a, b, seeds in PAIRS:
        rng = np.random.default_rng(aph.BOOT_SEED)
        row = {"a": a, "b": b, "files_a": files(a, seeds), "files_b": files(b, seeds)}
        for m in ("hit@10", "ndcg@10"):
            r = compare(runs, a, b, seeds, m, mask, rng)
            row[m] = {"delta": round(r["delta"], 5), "ci": [round(x, 5) for x in r["ci"]], "p": r["p"]}
        out.append(row)
        print(json.dumps(row))
    with open(os.path.join(HERE, "t9_check_profile_rows.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
