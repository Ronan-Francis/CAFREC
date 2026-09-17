"""Gate-activation export and inference-time gate interventions (TODO item 6).

For each saved CAFREC checkpoint this script rebuilds the test split from the
checkpoint's own resolved config, then

  1. exports the gate g in [0,1]^H for every test user under the REAL context,
     with summary statistics: how much g varies across users, and how much of
     that variation each context feature explains (linear R^2);
  2. re-scores the test set under inference-time interventions on the SAME
     trained weights:
        real      g = f_gate(x_ctx)                   (what the model was trained for)
        shuffled  g = f_gate(x_ctx permuted across users)
        mean      g = mean of f_gate(x_ctx) over test users (a constant gate)
        none      g = 0, i.e. z = z_short
     and runs paired Wilcoxon tests against `real`.

If `shuffled` and `mean` score like `real`, the trained gate's per-session
variation carries no ranking information at inference time.

    .venv/Scripts/python analyze_gate.py            # all known local checkpoints
    .venv/Scripts/python analyze_gate.py saved/X.pth ...

Run from cafrec_harness/. CPU only. Writes results/gate_export_<date>.{txt,json}.
Local checkpoints were trained at train_batch_size=2048 (read from each config).
"""
from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

import cafrec.runner  # noqa: F401  (patches torch.load for RecBole checkpoints)
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed
from scipy.stats import wilcoxon

from cafrec.eval.full_rank import full_sort_ranks, metrics_at_k
from cafrec.models.cafrec import CAFREC

HERE = Path(__file__).resolve().parent

# Local checkpoints (Sep 15-16 overnight queue), named by run start time.
DEFAULT_CKPTS = {
    "CAFREC-Sep-15-2026_22-08-38.pth": "np_s42",
    "CAFREC-Sep-15-2026_23-20-56.pth": "shuffled_s42",
    "CAFREC-Sep-16-2026_00-31-59.pth": "shuffled_s77",
    "CAFREC-Sep-16-2026_01-43-17.pth": "shuffled_s123",
    "CAFREC-Sep-15-2026_22-44-57.pth": "vector_s42",
    "CAFREC-Sep-15-2026_23-56-37.pth": "vector_s77",
    "CAFREC-Sep-16-2026_01-07-42.pth": "vector_s123",
}


def _load(path):
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


def _collect(model, test_data):
    """Real-context gate g and context x_ctx for every test user, eval order."""
    gs, xs, uids = [], [], []
    uid_field = test_data.dataset.uid_field
    with torch.no_grad():
        for interaction, _, positive_u, _ in test_data:
            seq_len = interaction[model.ITEM_SEQ_LEN]
            ctx = model._build_context(interaction, seq_len)
            gs.append(model.gating_mlp(ctx)[positive_u].numpy())
            xs.append(ctx[positive_u].numpy())
            uids.append(interaction[uid_field][positive_u].numpy())
    return np.concatenate(gs), np.concatenate(xs), np.concatenate(uids)


def _r2(y, x):
    """Share of the total variance of multivariate y explained by OLS on x."""
    x = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    tot = ((y - y.mean(0)) ** 2).sum()
    return float(1 - (resid ** 2).sum() / tot) if tot > 0 else float("nan")


def _gate_stats(g, x, fields):
    gbar = g.mean(0)
    dev = np.linalg.norm(g - gbar, axis=1) / max(np.linalg.norm(1 - gbar), 1e-12)
    return {
        "mean_g": float(g.mean()),
        "dim_mean_min": float(gbar.min()), "dim_mean_max": float(gbar.max()),
        "across_user_sd_mean": float(g.std(0).mean()),
        "across_user_sd_max": float(g.std(0).max()),
        # size of the per-user deviation from a constant gate, relative to the
        # constant gate's own modulation of z_short (1 - gbar)
        "rel_deviation_median": float(np.median(dev)),
        "rel_deviation_p90": float(np.quantile(dev, 0.9)),
        "r2_all_features_linear": _r2(g, x),
        "r2_by_feature": {f: _r2(g, x[:, [i]]) for i, f in enumerate(fields)},
    }


def _score(model, test_data, dataset, gate_fn):
    orig = model._gate
    model._gate = types.MethodType(gate_fn, model)
    try:
        uids, ranks = full_sort_ranks(model, test_data, torch.device("cpu"),
                                      dataset.item_num, dataset.uid_field, k=10)
    finally:
        model._gate = orig
    order = np.argsort(uids)
    return uids[order], ranks[order]


def _ndcg(r):
    return np.where(r <= 10, 1.0 / np.log2(r + 1.0), 0.0)


def analyse(path, label):
    t0 = time.time()
    config, dataset, test_data, model = _load(path)
    fields = list(config["context_fields"])
    out = {"checkpoint": Path(path).name, "label": label,
           "ablation": config["ablation"], "seed": config["seed"],
           "train_batch_size": config["train_batch_size"]}

    if config["ablation"] == "np_vector_gate":
        g = torch.sigmoid(model.vector_gate).detach().numpy()
        out["vector_gate"] = {"mean": float(g.mean()), "min": float(g.min()),
                              "max": float(g.max())}
        out["wall_s"] = round(time.time() - t0, 1)
        return out

    g, x, cuids = _collect(model, test_data)
    out["gate"] = _gate_stats(g, x, fields)
    # session position of each held-out interaction: prefix length 0 <=> the smallest
    # z-scored prefix_session_len value (the target opens a new session)
    pz = x[:, fields.index("prefix_session_len_log_z")]
    opener_by_uid = dict(zip(cuids.tolist(), np.isclose(pz, pz.min()).tolist()))
    out["gate_by_session"] = {
        name: _gate_stats(g[m], x[m], fields)
        for name, m in (("opener", np.isclose(pz, pz.min())), ("continuing", ~np.isclose(pz, pz.min())))}
    gbar = torch.tensor(g.mean(0))
    gen = torch.Generator().manual_seed(0)

    def real(self, context, ref):
        return self.gating_mlp(context)

    def shuffled(self, context, ref):
        idx = torch.randperm(context.size(0), generator=gen)
        return self.gating_mlp(context[idx])

    def mean(self, context, ref):
        return gbar.expand(ref.size(0), -1)

    def none(self, context, ref):
        return torch.zeros_like(ref)

    conds = {"real": real, "shuffled": shuffled, "mean": mean, "none": none}
    ranks = {}
    for name, fn in conds.items():
        u, r = _score(model, test_data, dataset, fn)
        ranks[name] = r
        out.setdefault("metrics", {})[name] = metrics_at_k(r, k=10)
    base = _ndcg(ranks["real"])
    opener = np.array([opener_by_uid[int(v)] for v in u])
    for name in ("shuffled", "mean", "none"):
        d = _ndcg(ranks[name]) - base
        for part, m in (("all", np.ones_like(opener)), ("opener", opener), ("continuing", ~opener)):
            dm, bm = d[m], base[m]
            nz = dm[dm != 0]
            p = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
            out.setdefault("vs_real_ndcg", {}).setdefault(part, {})[name] = {
                "delta": float(dm.mean()), "rel_pct": float(100 * dm.mean() / bm.mean()),
                "p": p, "users_changed": int(len(nz)), "n": int(m.sum())}
    out["wall_s"] = round(time.time() - t0, 1)
    return out


def main(argv):
    torch.set_num_threads(4)          # leave cores for the local training queue
    if argv:
        jobs = {p: Path(p).stem for p in argv}
    else:
        jobs = {str(HERE / "saved" / k): v for k, v in DEFAULT_CKPTS.items()}
    stamp = time.strftime("%Y%m%d")
    results = []
    for path, label in jobs.items():
        print(f"[{time.strftime('%H:%M:%S')}] {label} <- {Path(path).name}", flush=True)
        res = analyse(path, label)
        results.append(res)
        print(json.dumps(res, indent=1), flush=True)
    (HERE / "results" / f"gate_export_{stamp}.json").write_text(json.dumps(results, indent=2))
    lines = []
    for r in results:
        lines.append(f"== {r['label']}  ({r['checkpoint']}, ablation={r['ablation']}, "
                     f"batch={r['train_batch_size']})")
        if "vector_gate" in r:
            v = r["vector_gate"]
            lines.append(f"   constant gate: mean {v['mean']:.3f}  range [{v['min']:.3f}, {v['max']:.3f}]")
            continue
        s = r["gate"]
        lines.append(f"   gate mean {s['mean_g']:.3f}; per-dim mean range [{s['dim_mean_min']:.3f}, "
                     f"{s['dim_mean_max']:.3f}]; across-user sd mean {s['across_user_sd_mean']:.4f} "
                     f"(max {s['across_user_sd_max']:.4f})")
        lines.append(f"   deviation from constant gate, rel. to (1-gbar): median "
                     f"{s['rel_deviation_median']:.3f}, p90 {s['rel_deviation_p90']:.3f}")
        lines.append(f"   linear R^2 of g on all six features {s['r2_all_features_linear']:.3f}; by feature: "
                     + ", ".join(f"{k} {v:.3f}" for k, v in s["r2_by_feature"].items()))
        for part in ("opener", "continuing"):
            gs = r["gate_by_session"][part]
            lines.append(f"   [{part}] gate mean {gs['mean_g']:.3f}; across-user sd mean "
                         f"{gs['across_user_sd_mean']:.4f}; R^2 inter_session_gap "
                         f"{gs['r2_by_feature']['inter_session_gap_log_z']:.3f}")
        for name, m in r["metrics"].items():
            lines.append(f"   {name:8s} HR@10 {m['hit@10']:.4f}  NDCG@10 {m['ndcg@10']:.4f}  "
                         f"MRR@10 {m['mrr@10']:.4f}")
            for part in ("all", "opener", "continuing"):
                c = r.get("vs_real_ndcg", {}).get(part, {}).get(name)
                if c:
                    lines.append(f"      {part:<10} n={c['n']:,} dNDCG vs real {c['delta']:+.5f} "
                                 f"({c['rel_pct']:+.2f}%), p={c['p']:.3g}, users changed {c['users_changed']}")
    txt = "\n".join(lines)
    (HERE / "results" / f"gate_export_{stamp}.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main(sys.argv[1:])
