"""T1 split check, independent of the MostPop gate.

For every user whose held-out item a logged SASRec run placed in its top 10, the item at
position `rank` of that run's dumped top-10 list must be the test item of the split that
t1_timeline_leak.py rebuilds. Uses the seven batch-512 SASRec runs of tab:res:seeds.

Also records how RecBole's Pop model counts popularity, the source of the gate mismatch:
`item_cnt[item, :] = item_cnt[item, :] + 1` adds one per distinct item per batch. The
demonstration below shows that indexed assignment on a toy tensor, and the saturation
count uses the true per-item training counts.

Writes results/todo_runs/T1/t1_split_check.json.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
INTER = os.path.join(HARNESS, "..", "data", "recbole", "kuairand_pure", "kuairand_pure.inter")
POP_RUN = os.path.join(HARNESS, "results", "local", "Pop_kuairand_pure_pop_seed2020.json")


def main():
    df = pd.read_csv(INTER, sep="\t", dtype=str)
    df.columns = ["user", "item", "ts"]
    df["ts"] = df["ts"].astype(float).astype(np.int64)
    df["row"] = np.arange(len(df))
    df = df.sort_values(["ts", "row"], kind="stable")
    last = df.groupby("user").tail(1).set_index("user")["item"]
    train = df.groupby("user", group_keys=False).apply(lambda g: g.iloc[:-2])

    checks = []
    for f in sorted(glob.glob(os.path.join(HARNESS, "results", "modal",
                                           "SASRec_kuairand_pure_bs512_sasrec_seed*_*.json"))):
        d = json.load(open(f))
        top = dict(zip(map(str, d["test_topk_user_ids"]), d["test_topk_items"]))
        n = agree = 0
        for u, r in zip(map(str, d["test_user_ids"]), d["test_ranks"]):
            if r is not None and r <= 10:
                n += 1
                agree += str(top[u][int(r) - 1]) == str(last[u])
        checks.append({"file": os.path.basename(f), "users_in_top10": n, "test_item_agrees": agree})

    # RecBole Pop counting on a toy tensor: duplicate indices within one batch add once
    cnt = torch.zeros(3, 1, dtype=torch.long)
    idx = torch.tensor([0, 0, 0, 1])
    cnt[idx, :] = cnt[idx, :] + 1
    toy = cnt.squeeze(-1).tolist()

    counts = train["item"].value_counts()
    n_batches = int(np.ceil(len(train) / 2048))
    pop_run = json.load(open(POP_RUN))
    out = {
        "split_check": checks,
        "all_agree": all(c["users_in_top10"] == c["test_item_agrees"] for c in checks),
        "train_rows": int(len(train)),
        "recbole_pop_toy": {"indices": idx.tolist(), "resulting_counts": toy,
                            "note": "three occurrences of item 0 in one batch add 1, not 3"},
        "logged_pop_run": {"file": os.path.relpath(POP_RUN, HARNESS),
                           "train_batch_size": pop_run["config"]["train_batch_size"],
                           "epochs": pop_run["config"]["epochs"],
                           "batches_per_epoch": n_batches,
                           "reported_test_hit@10": pop_run["test"]["hit@10"],
                           "ranks_check_from_dump": pop_run["ranks_check"]},
        "items_with_more_train_rows_than_batches": int((counts > n_batches).sum()),
    }
    with open(os.path.join(HERE, "t1_split_check.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
