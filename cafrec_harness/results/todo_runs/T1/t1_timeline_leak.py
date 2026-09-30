"""T1 (TODO.md): timeline-leak share and MostPop HR@10 with full vs causal popularity.

Fills B2a and B2b in 06_discussion.tex. CPU only, no model is trained.

Split: the harness's leave-one-out split, reproduced here. Rows of the filtered organic-click
table (data/recbole/kuairand_pure/kuairand_pure.inter) are stably sorted by timestamp, with
file order breaking ties (RecBole's TO ordering); per user the last row is the test click,
the second-last the validation click, and the rest are training rows.

B2a  For each test user u with test time t_u: training rows of OTHER users with time > t_u,
     divided by all training rows of other users. Mean and median over test users.
B2b  MostPop HR@10, full-catalogue ranking, with the user's training and validation items
     masked and the test item left in (RecBole's full-sort history = used items minus the
     positive item):
       (a) popularity counted over all training rows (gate: HR@10 0.0357, tab:res:coverage);
       (b) popularity counted per user over training rows with time < t_u only.
     Ties in popularity are reported both ways (strict: rank = 1 + #items scoring higher;
     pessimistic: 1 + #items scoring at least as high, excluding the test item).
Optional  mean share of each test item's training interactions that post-date its test click.

Gate check: the per-user ranks of (a) are compared with the logged MostPop run
(results/local/Pop_kuairand_pure_pop_seed2020.json).

Usage (from cafrec_harness/):  <recbole env python> results/todo_runs/T1/t1_timeline_leak.py
Writes results/todo_runs/T1/t1_results.json, t1_per_user.npz and t1_summary.txt.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
INTER = os.path.join(HARNESS, "..", "data", "recbole", "kuairand_pure", "kuairand_pure.inter")
POP_RUN = os.path.join(HARNESS, "results", "local", "Pop_kuairand_pure_pop_seed2020.json")


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HARNESS, text=True).strip()
    except Exception as e:  # pragma: no cover
        return f"unavailable ({e})"


def ranks_for(pop, test_item, masked):
    """Strict and pessimistic rank of test_item under scores `pop`, masked items removed."""
    s = pop[test_item]
    keep = np.ones(len(pop), bool)
    keep[masked] = False
    keep[0] = False                      # padding id, never ranked
    keep[test_item] = False              # compare against the other items
    others = pop[keep]
    strict = 1 + int((others > s).sum())
    pess = 1 + int((others >= s).sum())
    return strict, pess


def main():
    t0 = time.time()
    df = pd.read_csv(INTER, sep="\t", dtype=str)
    df.columns = ["user", "item", "ts"]
    df["ts"] = df["ts"].astype(float).astype(np.int64)
    df["row"] = np.arange(len(df))
    n_rows = len(df)

    # RecBole TO ordering: stable sort by time over the whole table, then group by user.
    df = df.sort_values(["ts", "row"], kind="stable")
    df["pos"] = df.groupby("user").cumcount()
    df["n_u"] = df.groupby("user")["user"].transform("size")
    role = np.where(df["pos"] == df["n_u"] - 1, "test",
                    np.where(df["pos"] == df["n_u"] - 2, "valid", "train"))
    df["role"] = role

    items = sorted(df["item"].unique(), key=int)
    item_idx = {it: k + 1 for k, it in enumerate(items)}        # 0 = padding
    df["iid"] = df["item"].map(item_idx).astype(np.int64)
    n_items = len(items)

    train = df[df.role == "train"]
    test = df[df.role == "test"].set_index("user")
    valid = df[df.role == "valid"].set_index("user")
    users = sorted(test.index, key=int)
    n_train = len(train)

    # ---- B2a: share of other users' training rows that post-date u's test click
    tr_ts_sorted = np.sort(train["ts"].to_numpy())
    tr_by_user = train.groupby("user")
    n_tr_u = tr_by_user.size()
    t_u = test.loc[users, "ts"].to_numpy()
    after_all = n_train - np.searchsorted(tr_ts_sorted, t_u, side="right")
    own_after = np.array([int((tr_by_user.get_group(u)["ts"].to_numpy() > t).sum())
                          for u, t in zip(users, t_u)])
    denom = n_train - n_tr_u.loc[users].to_numpy()
    share = (after_all - own_after) / denom

    # ---- B2b: MostPop, full vs causal popularity
    tr_iid = train["iid"].to_numpy()
    pop_full = np.bincount(tr_iid, minlength=n_items + 1).astype(float)
    hist = {u: g["iid"].to_numpy() for u, g in tr_by_user}
    test_iid = test.loc[users, "iid"].to_numpy()
    valid_iid = valid.loc[users, "iid"].to_numpy()

    rk_full = np.empty((len(users), 2), int)
    for k, u in enumerate(users):
        masked = np.setdiff1d(np.append(hist[u], valid_iid[k]), [test_iid[k]])
        rk_full[k] = ranks_for(pop_full, test_iid[k], masked)

    # causal: users in order of test time; training rows added while ts < t_u
    order_tr = np.argsort(train["ts"].to_numpy(), kind="stable")
    tr_ts_o = train["ts"].to_numpy()[order_tr]
    tr_iid_o = tr_iid[order_tr]
    pop_c = np.zeros(n_items + 1)
    ptr = 0
    rk_causal = np.empty((len(users), 2), int)
    n_rows_used = np.empty(len(users), np.int64)
    for k in np.argsort(t_u, kind="stable"):
        while ptr < len(tr_ts_o) and tr_ts_o[ptr] < t_u[k]:
            pop_c[tr_iid_o[ptr]] += 1
            ptr += 1
        n_rows_used[k] = ptr
        masked = np.setdiff1d(np.append(hist[users[k]], valid_iid[k]), [test_iid[k]])
        rk_causal[k] = ranks_for(pop_c, test_iid[k], masked)

    def hr(r):
        return float((r <= 10).mean())

    # ---- optional: share of the test item's training interactions after its test click
    item_ts = {i: np.sort(g["ts"].to_numpy()) for i, g in train.groupby("iid")}
    item_share = []
    for i, t in zip(test_iid, t_u):
        ts = item_ts.get(i)
        if ts is None or len(ts) == 0:
            continue
        item_share.append((len(ts) - np.searchsorted(ts, t, side="right")) / len(ts))
    item_share = np.array(item_share)

    # ---- gate: logged MostPop run
    pop_run = json.load(open(POP_RUN))
    logged = dict(zip((str(u) for u in pop_run["test_user_ids"]), pop_run["test_ranks"]))
    lr = np.array([logged[u] for u in users], float)
    gate = {
        "logged_file": os.path.relpath(POP_RUN, HARNESS),
        "logged_test_hit@10": pop_run["test"]["hit@10"],
        "logged_hr_from_ranks": float((lr <= 10).mean()),
        "reproduced_hr_strict": hr(rk_full[:, 0]),
        "reproduced_hr_pessimistic": hr(rk_full[:, 1]),
        "users_rank_equal_strict": int((lr == rk_full[:, 0]).sum()),
        "users_rank_equal_pessimistic": int((lr == rk_full[:, 1]).sum()),
        "users_rank_within_tie_band": int(((lr >= rk_full[:, 0]) & (lr <= rk_full[:, 1])).sum()),
        "users_top10_status_equal": int(((lr <= 10) == (rk_full[:, 0] <= 10)).sum()),
        "n_users": len(users),
    }
    gate["passed"] = bool(round(gate["reproduced_hr_strict"], 4) == 0.0357
                          and round(gate["reproduced_hr_pessimistic"], 4) == 0.0357
                          and gate["users_top10_status_equal"] == len(users))

    res = {
        "task": "T1", "markers": ["B2a", "B2b"],
        "inputs": {"inter": os.path.relpath(INTER, HARNESS), "rows": n_rows, "users": len(users),
                   "items": n_items, "train_rows": n_train},
        "B2a": {"mean_share": float(share.mean()), "median_share": float(np.median(share)),
                "own_rows_after_test_total": int(own_after.sum()),
                "definition": "other users' training rows with time > t_u / other users' training rows"},
        "B2b": {"hr10_full_strict": hr(rk_full[:, 0]), "hr10_full_pessimistic": hr(rk_full[:, 1]),
                "hr10_causal_strict": hr(rk_causal[:, 0]),
                "hr10_causal_pessimistic": hr(rk_causal[:, 1]),
                "rel_change_causal_vs_full_pct": 100 * (hr(rk_causal[:, 0]) - hr(rk_full[:, 0]))
                / hr(rk_full[:, 0]),
                "causal_rows_used_mean": float(n_rows_used.mean())},
        "optional_item_share_after_test": {"mean": float(item_share.mean()),
                                           "median": float(np.median(item_share)),
                                           "n_test_clicks": int(len(item_share))},
        "gate": gate,
        "provenance": {"commit": git_head(), "device": f"{platform.node()} CPU",
                       "modal_run_id": None, "wall_s": round(time.time() - t0, 1),
                       "date": time.strftime("%Y-%m-%d %H:%M:%S")},
    }
    with open(os.path.join(HERE, "t1_results.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    np.savez_compressed(os.path.join(HERE, "t1_per_user.npz"), users=np.array(users),
                        share=share, rank_full=rk_full, rank_causal=rk_causal, logged_rank=lr)
    summary = (f"T1: B2a mean {100 * share.mean():.2f}% (median {100 * np.median(share):.2f}%); "
               f"MostPop HR@10 full {hr(rk_full[:, 0]):.4f}, causal {hr(rk_causal[:, 0]):.4f} "
               f"({res['B2b']['rel_change_causal_vs_full_pct']:+.1f}%); gate "
               f"{'PASS' if gate['passed'] else 'FAIL'} (logged 0.0357, "
               f"{gate['users_top10_status_equal']}/{len(users)} users same top-10 status)")
    with open(os.path.join(HERE, "t1_summary.txt"), "w") as fh:
        fh.write(summary + "\n")
    print(json.dumps(res, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
