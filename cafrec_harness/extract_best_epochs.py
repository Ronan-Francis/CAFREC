"""Best validation epoch of each locally trained run, from its saved RecBole checkpoint.

RecBole writes a checkpoint each time validation NDCG@10 improves, so the checkpoint's
`epoch` field (0-based) is the epoch of the best validation score; this script reports
it 1-based (epoch 10 is the last of ten). Each result JSON under results/local and
results/local_gpu is matched to the checkpoints under saved/ whose config agrees on
model, dataset, seed, batch size, learning rate, weight decay, depth, dropout, ablation,
profile file, history gate and context fields, and whose best_valid_score equals the JSON's
valid_best ndcg@10 to four decimals. A best epoch is reported only for a unique match.

The A10G runs (results/modal) cannot be covered: Modal persisted only the result JSON,
which records the best validation score but not its epoch.

Usage:  .venv/Scripts/python extract_best_epochs.py
Writes results/best_epochs_<date>.csv and .txt.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ("model", "dataset", "seed", "train_batch_size", "learning_rate", "weight_decay",
        "n_layers", "hidden_dropout_prob", "ablation", "profile", "history_gate", "context_fields")
COLS = ["file", "model", "dataset", "seed", "ablation", "n_layers", "hidden_dropout_prob",
        "learning_rate", "weight_decay", "train_batch_size", "epochs", "stopping_step",
        "valid_ndcg", "best_epoch", "n_ckpt", "ckpt", "device"]


def _norm(v):
    return round(float(v), 6) if isinstance(v, float) else v


def _key(cfg, model, dataset, seed):
    prof = cfg.get("llm_profile_path")
    prof = os.path.basename(str(prof).replace("\\", "/")) if prof else None
    d = {"model": model, "dataset": dataset, "seed": seed,
         "ablation": cfg.get("ablation") or "none", "profile": prof,
         "history_gate": bool(cfg.get("history_gate") or False),
         "context_fields": tuple(cfg.get("context_fields") or ())}
    for k in ("train_batch_size", "learning_rate", "weight_decay", "n_layers", "hidden_dropout_prob"):
        d[k] = cfg.get(k)
    return tuple(_norm(d[k]) for k in KEYS)


def _cfg_get(cfg, k):
    try:
        return cfg[k]
    except (KeyError, TypeError):
        return None


def load_checkpoints():
    out = []
    for path in sorted(glob.glob(os.path.join(HERE, "saved", "**", "*.pth"), recursive=True)):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        c = ck["config"]
        cfg = {k: _cfg_get(c, k) for k in ("train_batch_size", "learning_rate", "weight_decay",
                                            "n_layers", "hidden_dropout_prob", "ablation",
                                            "llm_profile_path", "history_gate", "context_fields")}
        key = _key(cfg, _cfg_get(c, "model"), _cfg_get(c, "dataset"), _cfg_get(c, "seed"))
        out.append({"path": os.path.relpath(path, HERE).replace("\\", "/"), "key": key,
                    "epoch": int(ck["epoch"]) + 1, "valid": round(float(ck["best_valid_score"]), 4)})
    return out


def main():
    ckpts = load_checkpoints()
    rows, used = [], set()
    files = sorted(glob.glob(os.path.join(HERE, "results", "local", "*.json"))
                   + glob.glob(os.path.join(HERE, "results", "local_gpu", "*.json")))
    n_cfg = 0
    for f in files:
        d = json.load(open(f))
        cfg = d.get("config")
        if not cfg:
            continue
        n_cfg += 1
        key = _key(cfg, d["model"], d["dataset"], d["seed"])
        vnd = round(float(d["valid_best"]["ndcg@10"]), 4)
        hits = [c for c in ckpts if c["key"] == key and c["valid"] == vnd]
        one = hits[0] if len(hits) == 1 else None
        if one:
            used.add(one["path"])
        rows.append({"file": os.path.relpath(f, HERE).replace("\\", "/"), "model": d["model"],
                     "dataset": d["dataset"], "seed": d["seed"], "ablation": cfg.get("ablation") or "",
                     "n_layers": cfg.get("n_layers"), "hidden_dropout_prob": cfg.get("hidden_dropout_prob"),
                     "learning_rate": cfg.get("learning_rate"), "weight_decay": cfg.get("weight_decay"),
                     "train_batch_size": cfg.get("train_batch_size"), "epochs": cfg.get("epochs"),
                     "stopping_step": cfg.get("stopping_step"), "valid_ndcg": vnd,
                     "best_epoch": one["epoch"] if one else "", "n_ckpt": len(hits),
                     "ckpt": one["path"] if one else "", "device": d.get("device", "")})

    stamp = time.strftime("%Y%m%d")
    with open(os.path.join(HERE, "results", f"best_epochs_{stamp}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    lines = [f"Best validation epoch (1-based) from saved checkpoints, {time.strftime('%Y-%m-%d %H:%M')}",
             f"checkpoints read: {len(ckpts)}; result JSONs with config: {n_cfg}; "
             f"matched uniquely: {sum(1 for r in rows if r['n_ckpt'] == 1)}", ""]
    for r in rows:
        be = f"{r['best_epoch']:>4}" if r["best_epoch"] != "" else "None"
        lines.append(f"{os.path.basename(r['file']):<62} best_epoch={be}  "
                     f"valid_ndcg={r['valid_ndcg']:.4f}  n_ckpt={r['n_ckpt']}")
    orphan = [c for c in ckpts if c["path"] not in used]
    lines += ["", f"checkpoints with no unique result JSON: {len(orphan)}"]
    lines += [f"  {c['path']}: {c['key'][0]} {c['key'][1]} seed {c['key'][2]} bs {c['key'][3]} "
              f"epoch {c['epoch']} valid {c['valid']}" for c in orphan]
    txt = "\n".join(lines)
    with open(os.path.join(HERE, "results", f"best_epochs_{stamp}.txt"), "w") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
