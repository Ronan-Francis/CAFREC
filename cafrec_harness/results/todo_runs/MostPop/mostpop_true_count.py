"""True-count MostPop row for Table XIII (tab:res:coverage). CPU query; no model is trained.

Popularity = number of training rows per item (all users), under the harness's leave-one-out
split rebuilt as in T1 (stable time sort, last click = test, second-last = validation; verified
against the SASRec top-10 dumps in todo_runs/T1/t1_split_check.json). Each user's top-10 list is
the ten most popular items after masking that user's training and validation items, the test item
left in (RecBole's full-sort masking for a general recommender: used items minus the positive).
Ties in count are broken by the smaller original item id; the script reports whether any tie
straddles a user's tenth slot.

Metrics use the definitions that produced every other row of Table XIII, imported unchanged:
  Coverage@10, Items, Top-10 share   analyze_paper_512.py lines 140-143
  ILD@10 (first level; second level and tag as sensitivity), no-category share
                                     analyze_ild.category_matrices / unit_rows / per_user_ild
  HR@10                              share of users whose test item is in the list

Gate: the same metric code applied to the logged RecBole Pop run
(results/local/Pop_kuairand_pure_pop_seed2020.json) must reproduce the current row
(coverage 0.004, 28 items, ILD 0.958, top-10 share 0.879, HR@10 0.0357), and the true-count
HR@10 must equal T1's 0.0432 (todo_runs/T1/t1_results.json).

Usage (from cafrec_harness/):  <recbole env python> results/todo_runs/MostPop/mostpop_true_count.py
Writes mostpop_results.json, mostpop_topk.npz and mostpop_summary.txt in this folder.
"""
from __future__ import annotations

import collections
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HARNESS)
os.chdir(HARNESS)

from analyze_ild import LEVELS, category_matrices, per_user_ild, unit_rows  # noqa: E402

INTER = os.path.join(HARNESS, "..", "data", "recbole", "kuairand_pure", "kuairand_pure.inter")
POP_RUN = os.path.join(HARNESS, "results", "local", "Pop_kuairand_pure_pop_seed2020.json")
T1 = os.path.join(HARNESS, "results", "todo_runs", "T1", "t1_results.json")
N_CATALOG = 7210
K = 10


def row_metrics(topk, test_items, units, zero_rows):
    """Table XIII columns for one set of top-10 lists (original item ids)."""
    c = collections.Counter(int(i) for row in topk for i in row)          # analyze_paper_512 L140
    out = {"items": len(c), "coverage": len(c) / N_CATALOG,             # L141-142
           "top10_share": sum(v for _, v in c.most_common(10)) / sum(c.values())}  # L143
    if test_items is not None:
        out["hr10"] = float(np.mean([t in set(row) for row, t in zip(topk.tolist(), test_items)]))
    for lvl in LEVELS:
        out[f"ild_{lvl}"] = float(per_user_ild(topk, units[lvl]).mean())
        out[f"nocat_{lvl}"] = float(zero_rows[lvl][topk].mean())
    out["top10_items"] = [int(i) for i, _ in c.most_common(10)]
    return out


def main():
    t0 = time.time()
    mats, _ = category_matrices()
    units, zero_rows = {}, {}
    for lvl in LEVELS:
        units[lvl], zero_rows[lvl] = unit_rows(mats[lvl])

    # ---- gate part 1: the logged RecBole Pop run through the same metric code
    pop = json.load(open(POP_RUN))
    logged_topk = np.asarray(pop["test_topk_items"], dtype=np.int64)
    logged = row_metrics(logged_topk, None, units, zero_rows)
    logged["hr10"] = pop["test"]["hit@10"]

    # ---- split and true-count popularity (as T1)
    df = pd.read_csv(INTER, sep="\t", dtype=str)
    df.columns = ["user", "item", "ts"]
    df["ts"] = df["ts"].astype(float).astype(np.int64)
    df["item"] = df["item"].astype(np.int64)
    df["row"] = np.arange(len(df))
    df = df.sort_values(["ts", "row"], kind="stable")
    df["pos"] = df.groupby("user").cumcount()
    df["n_u"] = df.groupby("user")["user"].transform("size")
    test = df[df.pos == df.n_u - 1].set_index("user")["item"]
    valid = df[df.pos == df.n_u - 2].set_index("user")["item"]
    train = df[df.pos < df.n_u - 2]
    users = sorted(test.index, key=int)

    counts = train["item"].value_counts()
    items = np.array(sorted(df["item"].unique()))
    cnt = counts.reindex(items, fill_value=0).to_numpy()
    order = np.lexsort((items, -cnt))            # count descending, then smaller original id
    ranked_items, ranked_cnt = items[order], cnt[order]

    hist = {u: set(g.tolist()) for u, g in train.groupby("user")["item"]}
    topk = np.empty((len(users), K), dtype=np.int64)
    tie_at_boundary = 0
    for n, u in enumerate(users):
        masked = (hist.get(u, set()) | {int(valid[u])}) - {int(test[u])}
        picked, last_cnt, k = [], None, 0
        while len(picked) < K:
            it = int(ranked_items[k])
            if it not in masked:
                picked.append(it)
                last_cnt = ranked_cnt[k]
            k += 1
        # does an unmasked item outside the list share the tenth item's count?
        while k < len(ranked_items) and ranked_cnt[k] == last_cnt:
            if int(ranked_items[k]) not in masked:
                tie_at_boundary += 1
                break
            k += 1
        topk[n] = picked
    test_items = [int(test[u]) for u in users]
    true = row_metrics(topk, test_items, units, zero_rows)
    true["tie_at_tenth_slot_users"] = tie_at_boundary
    true["top_counts"] = [int(x) for x in ranked_cnt[:15]]

    t1 = json.load(open(T1))["B2b"]["hr10_full_strict"]
    gate = {
        "logged_reproduced": {
            "coverage_3dp": round(logged["coverage"], 3), "items": logged["items"],
            "ild_first_3dp": round(logged["ild_first_level"], 3),
            "top10_share_3dp": round(logged["top10_share"], 3), "hr10": logged["hr10"],
            "expected": {"coverage_3dp": 0.004, "items": 28, "ild_first_3dp": 0.958,
                         "top10_share_3dp": 0.879, "hr10": 0.0357}},
        "true_hr_equals_t1": bool(abs(true["hr10"] - t1) < 1e-12), "t1_hr10": t1,
    }
    e = gate["logged_reproduced"]["expected"]
    gate["passed"] = (all(gate["logged_reproduced"][k] == v for k, v in e.items())
                      and gate["true_hr_equals_t1"])

    res = {"task": "MostPop true-count row (Table XIII)", "gate": gate,
           "logged_recbole_pop": logged, "true_count": true,
           "definition": {"popularity": "training-row count per item over all users",
                          "masking": "user's training and validation items; test item kept",
                          "ties": "smaller original item id first", "users": len(users),
                          "train_rows": int(len(train))},
           "provenance": {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HARNESS,
                                                            text=True).strip(),
                          "device": f"{platform.node()} CPU", "modal_run_id": None,
                          "wall_s": round(time.time() - t0, 1), "date": time.strftime("%Y-%m-%d %H:%M:%S")}}
    with open(os.path.join(HERE, "mostpop_results.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    np.savez_compressed(os.path.join(HERE, "mostpop_topk.npz"), users=np.array(users), topk=topk,
                        test_items=np.array(test_items))
    s = (f"MostPop true count: coverage {true['coverage']:.3f}, items {true['items']}, ILD@10 "
         f"{true['ild_first_level']:.3f}, top-10 share {true['top10_share']:.3f}, HR@10 {true['hr10']:.4f}; "
         f"logged Pop through the same code: {logged['coverage']:.3f}, {logged['items']}, "
         f"{logged['ild_first_level']:.3f}, {logged['top10_share']:.3f}, {logged['hr10']}; "
         f"gate {'PASS' if gate['passed'] else 'FAIL'}; ties at tenth slot: {tie_at_boundary} users")
    with open(os.path.join(HERE, "mostpop_summary.txt"), "w") as fh:
        fh.write(s + "\n")
    print(json.dumps(res, indent=2))
    print(s)


if __name__ == "__main__":
    main()
