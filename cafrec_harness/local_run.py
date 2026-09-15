"""Local (CPU) counterpart of modal_run.train, for runs that don't need a GPU.

Writes the same JSON schema as the Modal runs (per-user ranks + top-k lists), to
results/local/, so the offline significance/diversity code can pair them with
results/modal/ by original user id.

    # one run
    .venv/Scripts/python local_run.py --model CAFREC --dataset kuairand_pure_ctx \
        --ablation np_vector_gate --seed 42 --tag np_vector_gate

    # the overnight queue (skips jobs whose JSON already exists; logs to results/local/)
    .venv/Scripts/python local_run.py --queue overnight

Run from cafrec_harness/ (base.yaml's data_path is relative to it).
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results" / "local"

# Pure seeds 42/77/123 already have CAFREC-NP (tag noprof_ms) rank dumps on Modal,
# so every control below pairs user-for-user with an existing run.
QUEUES = {
    "overnight": [
        # (a) reproducibility check: does CPU reproduce the Modal CAFREC-NP seed 42?
        dict(model="CAFREC", dataset="kuairand_pure_ctx", ablation="no_profiler", seed=42, tag="noprof_local"),
        # (b) context-free per-dimension gate
        dict(model="CAFREC", dataset="kuairand_pure_ctx", ablation="np_vector_gate", seed=42, tag="np_vector_gate"),
        dict(model="CAFREC", dataset="kuairand_pure_ctx", ablation="np_shuffled_ctx", seed=42, tag="np_shuffled_ctx"),
        dict(model="CAFREC", dataset="kuairand_pure_ctx", ablation="np_vector_gate", seed=77, tag="np_vector_gate"),
        dict(model="CAFREC", dataset="kuairand_pure_ctx", ablation="np_shuffled_ctx", seed=77, tag="np_shuffled_ctx"),
        dict(model="CAFREC", dataset="kuairand_pure_ctx", ablation="np_vector_gate", seed=123, tag="np_vector_gate"),
        dict(model="CAFREC", dataset="kuairand_pure_ctx", ablation="np_shuffled_ctx", seed=123, tag="np_shuffled_ctx"),
    ],
    # 1-epoch smoke of each new ablation (a few minutes each)
    "smoke": [
        dict(model="CAFREC", dataset="kuairand_pure_ctx", ablation="np_vector_gate", seed=42, tag="smoke_vec", epochs=1),
        dict(model="CAFREC", dataset="kuairand_pure_ctx", ablation="np_shuffled_ctx", seed=42, tag="smoke_shuf", epochs=1),
    ],
}


def _out_path(job):
    return OUT_DIR / f"{job['model']}_{job['dataset']}_{job['tag']}_seed{job['seed']}.json"


def run_one(model, dataset, seed, ablation=None, tag=None, epochs=None,
            dump_ranks=True, dump_topk=True):
    from cafrec.runner import run_experiment

    overrides = {"seed": seed}
    if ablation is not None:
        overrides["ablation"] = ablation
    if epochs is not None:
        overrides["epochs"] = epochs
    t0 = time.time()
    metrics = run_experiment(model, dataset=dataset, config_overrides=overrides,
                             return_ranks=dump_ranks, dump_topk=dump_topk)
    metrics["wall_seconds"] = round(time.time() - t0, 1)
    metrics["device"] = "local-cpu"
    metrics["ablation"] = ablation
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _out_path(dict(model=model, dataset=dataset, tag=tag or (ablation or "none"), seed=seed))
    with open(out, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    return metrics, out


def _keep_awake():
    """Ask Windows not to sleep while this process runs (released on exit;
    no power-plan settings are changed). No-op on other platforms."""
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except (AttributeError, OSError):
        pass


def run_queue(name):
    _keep_awake()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = OUT_DIR / f"queue_{name}.log"
    jobs = QUEUES[name]
    for i, job in enumerate(jobs, 1):
        out = _out_path(job)
        if out.exists():
            _log(log, f"[{i}/{len(jobs)}] SKIP (exists) {out.name}")
            continue
        _log(log, f"[{i}/{len(jobs)}] START {job}")
        try:
            m, out = run_one(**job)
            _log(log, f"[{i}/{len(jobs)}] DONE  {out.name}  test={m['test']}  "
                      f"wall={m['wall_seconds']}s")
        except Exception:                         # keep going; one failure shouldn't kill the night
            _log(log, f"[{i}/{len(jobs)}] FAIL  {job}\n{traceback.format_exc()}")
    _log(log, "QUEUE FINISHED")


def _log(path, msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(path, "a") as fh:
        fh.write(line + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--queue", choices=sorted(QUEUES))
    p.add_argument("--model", default="CAFREC")
    p.add_argument("--dataset", default="kuairand_pure_ctx")
    p.add_argument("--ablation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag")
    p.add_argument("--epochs", type=int)
    a = p.parse_args()
    if a.queue:
        run_queue(a.queue)
    else:
        m, out = run_one(a.model, a.dataset, a.seed, a.ablation, a.tag, a.epochs)
        print(out, m["test"], f"{m['wall_seconds']}s")
