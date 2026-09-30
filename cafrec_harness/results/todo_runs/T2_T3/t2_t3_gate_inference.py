"""T2 and T3 (TODO.md), author's choice Q11-C (2026-09-30): inference on the three saved
batch-512 CAFREC-NP checkpoints of the ablation seeds, trained on a CPU for the content-profile
comparison. No training; torch.no_grad throughout.

  seed 2020    saved/CAFREC-Sep-17-2026_18-48-42.pth  ref results/local/..._np512_seed2020.json
  seed 2021    saved/CAFREC-Sep-17-2026_22-27-03.pth  ref results/local/..._np512_seed2021.json
  seed 403092  saved/CAFREC-Sep-18-2026_12-18-52.pth  ref results/local/..._np512_seed403092.json

Each checkpoint rebuilds its test split from its own resolved config (as analyze_gate.py does),
so x_ctx is read from the same z-scored dataset columns, with the same persisted scaler, as in
evaluation.

Gate (per seed): test ranks recomputed from the checkpoint equal the logged per-user ranks and
HR@10 of the reference run; partition counts 22,912 / 15,706 openers / 7,206 within /
3,521 high drift / 19,391 low drift (analyze_plan_hypotheses.cohorts, the paper's definitions).

T2  g in [0,1]^64 per test case: mean and SD over all cases and dimensions; mean at openers,
    within sessions, high and low drift. Variance shares: across sessions
    Var_u(mean_d g) / Var(g), across dimensions Var_d(mean_u g) / Var(g).
    Mean over seeds, SD across seeds logged.
T3  For each of the six features, permute that x_ctx column across cases within openers and,
    separately, within within-session cases (other columns kept), five draws per feature and
    seed (numpy default_rng([20260930, seed, feature, draw])). Per-user ΔNDCG@10 = permuted minus
    unpermuted, averaged over draws, then over seeds; paired bootstrap 95% CI with 2,000 user
    resamples (analyze_plan_hypotheses.boot_ci, default_rng(20260917), features in x_ctx order).

Usage (from cafrec_harness/):  <recbole env python> results/todo_runs/T2_T3/t2_t3_gate_inference.py
Writes t2_results.json, t3_results.json, t2_t3_per_user.npz, t2_t3_summary.txt and a run log.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HARNESS)
os.chdir(HARNESS)

import cafrec.runner  # noqa: E402,F401  (patches torch.load for RecBole checkpoints)
from recbole.data import create_dataset, data_preparation  # noqa: E402
from recbole.utils import init_seed  # noqa: E402

import analyze_plan_hypotheses as aph  # noqa: E402
from cafrec.eval.full_rank import full_sort_ranks, metrics_at_k  # noqa: E402
from cafrec.models.cafrec import CAFREC  # noqa: E402

CKPTS = {
    2020: "saved/CAFREC-Sep-17-2026_18-48-42.pth",
    2021: "saved/CAFREC-Sep-17-2026_22-27-03.pth",
    403092: "saved/CAFREC-Sep-18-2026_12-18-52.pth",
}
REF = "results/local/CAFREC_kuairand_pure_ctx_np512_seed{s}.json"
N_DRAWS = 5
EXPECTED = {"all": 22912, "short session": 15706, "longer session": 7206,
            "high drift": 3521, "low drift": 19391}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(HERE, "t2_t3_run.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load(path):
    ck = torch.load(path, map_location="cpu")
    config = ck["config"]
    config["device"] = torch.device("cpu")
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    _, _, test_data = data_preparation(config, dataset)
    model = CAFREC(config, test_data.dataset)
    model.load_state_dict(ck["state_dict"])
    model.load_other_parameter(ck.get("other_parameter"))
    model.eval()
    return config, dataset, test_data, model


def collect(model, test_data):
    """Real-context x_ctx and gate g per test user (internal uid order of the loader)."""
    xs, gs, uids = [], [], []
    uid_field = test_data.dataset.uid_field
    with torch.no_grad():
        for interaction, _, positive_u, _ in test_data:
            ctx = model._build_context(interaction, interaction[model.ITEM_SEQ_LEN])
            gs.append(model.gating_mlp(ctx)[positive_u].numpy())
            xs.append(ctx[positive_u].numpy())
            uids.append(interaction[uid_field][positive_u].numpy())
    return np.concatenate(xs), np.concatenate(gs), np.concatenate(uids)


def ranks_with(model, test_data, dataset, table=None):
    """Full-sort ranks; if `table` is given, x_ctx is looked up per internal user id."""
    orig = model._build_context
    if table is not None:
        t = torch.as_tensor(table, dtype=torch.float32)

        def lookup(self, interaction, item_seq_len):
            return t[interaction[self.USER_ID]]
        model._build_context = types.MethodType(lookup, model)
    try:
        with torch.no_grad():
            uids, ranks = full_sort_ranks(model, test_data, torch.device("cpu"),
                                          dataset.item_num, dataset.uid_field, k=10)
    finally:
        model._build_context = orig
    order = np.argsort(uids)
    return uids[order], ranks[order]


def ndcg(r):
    r = np.asarray(r, float)
    return np.where(r <= 10, 1.0 / np.log2(r + 1.0), 0.0)


def git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HARNESS, text=True).strip()


def main():
    torch.set_num_threads(4)
    t0 = time.time()
    per_seed, gate, t3_deltas, base_ndcg = {}, {}, {}, {}
    users_ref = None
    fields = None
    for seed, path in CKPTS.items():
        log(f"seed {seed}: loading {path}")
        config, dataset, test_data, model = load(path)
        assert config["ablation"] == "no_profiler" and config["train_batch_size"] == 512
        fields = list(config["context_fields"])
        tok = dataset.field2id_token[dataset.uid_field]

        # ---- gate: reproduce the logged ranks and HR@10
        uids, ranks = ranks_with(model, test_data, dataset)
        ref = json.load(open(REF.format(s=seed)))
        logged = dict(zip(map(str, ref["test_user_ids"]), ref["test_ranks"]))
        orig_u = np.array([str(tok[i]) for i in uids])
        lr = np.array([logged[u] for u in orig_u], float)
        m = metrics_at_k(ranks, k=10)
        gate[seed] = {"hr10_recomputed": float(m["hit@10"]), "hr10_logged": ref["test"]["hit@10"],
                      "ranks_equal": int((lr == ranks).sum()), "n_users": int(len(ranks)),
                      "passed": bool(round(float(m["hit@10"]), 4) == ref["test"]["hit@10"]
                                     and (lr == ranks).all())}
        log(f"seed {seed}: gate {gate[seed]}")

        # ---- x_ctx and g per user, aligned to the sorted internal uids of `ranks_with`
        x, g, cu = collect(model, test_data)
        order = np.argsort(cu)
        x, g, cu = x[order], g[order], cu[order]
        assert np.array_equal(cu, uids)
        users = [str(tok[i]) for i in cu]
        if users_ref is None:
            users_ref = users
            masks, _ = aph.cohorts(users)
            counts = {k: int(masks[k].sum()) for k in EXPECTED}
            gate["partition_counts"] = counts
            gate["partition_passed"] = counts == EXPECTED
            log(f"partition counts {counts}")
        assert users == users_ref

        # table lookup must reproduce the real-context ranks exactly
        table = np.zeros((dataset.user_num, x.shape[1]), np.float32)
        table[cu] = x
        _, r_tab = ranks_with(model, test_data, dataset, table)
        assert np.array_equal(r_tab, ranks), "context lookup does not reproduce real ranks"

        # ---- T2
        var_all = g.var()
        per_seed[seed] = {
            "mean": float(g.mean()), "sd": float(g.std()),
            "mean_opener": float(g[masks["short session"]].mean()),
            "mean_within": float(g[masks["longer session"]].mean()),
            "mean_high_drift": float(g[masks["high drift"]].mean()),
            "mean_low_drift": float(g[masks["low drift"]].mean()),
            "share_across_sessions": float(g.mean(1).var() / var_all),
            "share_across_dims": float(g.mean(0).var() / var_all),
            "dim_mean_range": [float(g.mean(0).min()), float(g.mean(0).max())],
        }
        log(f"seed {seed}: T2 {per_seed[seed]}")

        # ---- T3
        base = ndcg(ranks)
        base_ndcg[seed] = base
        groups = [np.where(masks["short session"])[0], np.where(masks["longer session"])[0]]
        for j, f in enumerate(fields):
            acc = np.zeros(len(users))
            for k in range(N_DRAWS):
                rng = np.random.default_rng([20260930, seed, j, k])
                xp = x.copy()
                for idx in groups:
                    xp[idx, j] = x[rng.permutation(idx), j]
                tp = table.copy()
                tp[cu] = xp
                _, rp = ranks_with(model, test_data, dataset, tp)
                acc += ndcg(rp) - base
            t3_deltas.setdefault(f, []).append(acc / N_DRAWS)
            log(f"seed {seed}: T3 {f} mean dNDCG {np.mean(acc / N_DRAWS):+.6f}")

    # ---- aggregate T2
    keys = list(next(iter(per_seed.values())).keys())
    t2 = {"per_seed": per_seed,
          "mean_over_seeds": {k: float(np.mean([per_seed[s][k] for s in per_seed]))
                              for k in keys if k != "dim_mean_range"},
          "sd_across_seeds": {k: float(np.std([per_seed[s][k] for s in per_seed], ddof=1))
                              for k in keys if k != "dim_mean_range"}}

    # ---- aggregate T3 with the IV-F bootstrap
    rng = np.random.default_rng(aph.BOOT_SEED)
    base_mean = float(np.mean([base_ndcg[s].mean() for s in base_ndcg]))
    t3 = {"base_ndcg10_mean": base_mean, "features": {}}
    for f in fields:
        d = np.mean(t3_deltas[f], axis=0)
        lo, hi = aph.boot_ci(d, rng)
        t3["features"][f] = {
            "delta": float(d.mean()), "rel_pct": float(100 * d.mean() / base_mean),
            "ci": [lo, hi], "ci_pct": [100 * lo / base_mean, 100 * hi / base_mean],
            "p": aph.wilcoxon_normal(d), "users_changed": int((d != 0).sum()),
            "delta_openers": float(d[masks["short session"]].mean()),
            "delta_within": float(d[masks["longer session"]].mean()),
            "per_seed_delta": [float(v.mean()) for v in t3_deltas[f]]}

    prov = {"commit": git_head(), "device": f"{platform.node()} CPU", "modal_run_id": None,
            "checkpoints": CKPTS, "seeds": list(CKPTS), "wall_s": round(time.time() - t0, 1),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    passed = all(gate[s]["passed"] for s in CKPTS) and gate["partition_passed"]
    with open(os.path.join(HERE, "t2_results.json"), "w") as fh:
        json.dump({"task": "T2", "markers": ["C2a", "C2b", "C2c"], "gate": gate,
                   "gate_passed": passed, **t2, "provenance": prov}, fh, indent=2)
    with open(os.path.join(HERE, "t3_results.json"), "w") as fh:
        json.dump({"task": "T3", "markers": ["C2d"], "gate_passed": passed, **t3,
                   "protocol": {"draws": N_DRAWS, "perm_rng": "default_rng([20260930, seed, feature, draw])",
                                "n_boot": aph.N_BOOT, "boot_seed": aph.BOOT_SEED},
                   "provenance": prov}, fh, indent=2)
    np.savez_compressed(os.path.join(HERE, "t2_t3_per_user.npz"), users=np.array(users_ref),
                        **{f"d_{f}": np.array(t3_deltas[f]) for f in fields},
                        **{f"base_{s}": base_ndcg[s] for s in base_ndcg})
    mo = t2["mean_over_seeds"]
    summary = (f"T2/T3 (3 CPU checkpoints, seeds {list(CKPTS)}): gate {'PASS' if passed else 'FAIL'}; "
               f"g mean {mo['mean']:.3f} sd {mo['sd']:.3f}; opener {mo['mean_opener']:.3f} within "
               f"{mo['mean_within']:.3f}; high {mo['mean_high_drift']:.3f} low {mo['mean_low_drift']:.3f}; "
               f"var share sessions {100 * mo['share_across_sessions']:.1f}% dims "
               f"{100 * mo['share_across_dims']:.1f}%; T3 " + ", ".join(
                   f"{f} {v['rel_pct']:+.2f}% [{v['ci_pct'][0]:+.2f}, {v['ci_pct'][1]:+.2f}]"
                   for f, v in t3["features"].items()))
    with open(os.path.join(HERE, "t2_t3_summary.txt"), "w") as fh:
        fh.write(summary + "\n")
    log(summary)


if __name__ == "__main__":
    main()
