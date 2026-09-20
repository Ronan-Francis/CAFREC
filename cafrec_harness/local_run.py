"""Local (CPU) counterpart of modal_run.train, for runs that don't need a GPU.

Writes the same JSON schema as the Modal runs (per-user ranks + top-k lists), to
results/local/, so the offline significance/diversity code can pair them with
results/modal/ by original user id.

    # one run
    .venv/Scripts/python local_run.py --model CAFREC --dataset kuairand_pure_ctx \
        --ablation np_vector_gate --seed 42 --tag np_vector_gate

    # the overnight queue (skips jobs whose JSON already exists; logs to results/local/)
    .venv/Scripts/python local_run.py --queue overnight

    # local GPU (2026-09-19): same script from the CUDA venv; results go to
    # results/local_gpu/ so they never pair with CPU or Modal runs by accident
    .venv-gpu/Scripts/python local_run.py --queue gpu_grid

Run from cafrec_harness/ (base.yaml's data_path is relative to it).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _cuda_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        return None


GPU_NAME = _cuda_name()
OUT_DIR = HERE / "results" / ("local_gpu" if GPU_NAME else "local")

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
    # Control seeds missing from "overnight", which ran no_profiler at seed 42 only.
    # Needed locally (not reused from results/modal/) because the seed-42 CPU vs Modal
    # check in "overnight" did not reproduce: see QUEUE NOTES at the bottom of this file.
    "controls": [
        dict(model="CAFREC", dataset="kuairand_pure_ctx", ablation="no_profiler", seed=77, tag="noprof_local"),
        dict(model="CAFREC", dataset="kuairand_pure_ctx", ablation="no_profiler", seed=123, tag="noprof_local"),
    ],
    # 1-epoch smoke of each new ablation (a few minutes each)
    # 2026-09-17: supervisor-review fixes. SASRec at batch 512 on the ablation seeds, so
    # tab:res:coverage and tab:res:seqlen stop mixing batch sizes / seed sets. Every job
    # pins train_batch_size=512 (the tier default is now 2048, see cafrec/tiers.py).
    "sasrec512": [
        dict(model="SASRec", dataset="kuairand_pure", seed=s, tag=f"bs512_L{L}",
             train_batch_size=512, max_seq_len=L)
        for L in (20, 10, 50) for s in (2020, 2021, 403092)
    ],
    # MostPop baseline (deterministic, so one seed). Reviewers expect a popularity floor.
    "pop": [
        dict(model="Pop", dataset="kuairand_pure", seed=2020, tag="pop", epochs=1),
    ],
    "smoke": [
        dict(model="CAFREC", dataset="kuairand_pure_ctx", ablation="np_vector_gate", seed=42, tag="smoke_vec", epochs=1),
        dict(model="CAFREC", dataset="kuairand_pure_ctx", ablation="np_shuffled_ctx", seed=42, tag="smoke_shuf", epochs=1),
    ],
    # 2026-09-17 content-bearing long-term profiles (build_content_profiles.py) vs a
    # CAFREC-NP reference trained on this same device. Stage 1 = ablation seeds.
    # `profile` names a cache in data/profiles/; run_queue resolves its path and dim.
    "content_profiles": [
        dict(model="CAFREC", dataset="kuairand_pure_ctx", seed=seed, train_batch_size=512, **cond)
        for seed in (2020, 2021, 403092)
        for cond in (dict(ablation="no_profiler", tag="np512"),
                     dict(ablation=None, profile="cat_beyond", tag="catbeyond512"),
                     dict(ablation=None, profile="cap_beyond", tag="capbeyond512"),
                     dict(ablation=None, profile="cap_all", tag="capall512"))
    ],
    "content_smoke": [
        dict(model="CAFREC", dataset="kuairand_pure_ctx", seed=2020, train_batch_size=512,
             ablation=None, profile="cap_beyond", tag="smoke_capbeyond512", epochs=1),
    ],
}

# ---------------------------------------------------------------------------
# 2026-09-19: tuned SASRec vs tuned CAFREC-NP (TODO S1) and the policy-flag ablation,
# on the local RTX 2060 SUPER (.venv-gpu). Every arm runs on this one device at batch
# 512, so nothing here pairs with results/modal/ or results/local/.
#
# Equal search budget: one 16-point grid over the shared encoder settings, the same
# for both models, selected on VALIDATION NDCG@10 at seed 403092 (not an evaluation
# seed). Point 00 is the RecBole default; point 13 is the config the 64-point RON-31
# search picked for SASRec (dropout 0.2, 4 layers / 4 heads, lr 1e-3, wd 1e-5).
# n_heads follows n_layers so that point is inside the grid.
GRID16 = [dict(hidden_dropout_prob=d, attn_dropout_prob=d, n_layers=L, n_heads=L,
               learning_rate=lr, weight_decay=wd)
          for d in (0.5, 0.2) for L in (2, 4) for lr in (1e-3, 3e-3) for wd in (0.0, 1e-5)]
GRID_SEED = 403092
HEADLINE_SEEDS = (42, 77, 123, 256, 512, 1024, 2048)
GRID_ARMS = {   # label -> job fields shared by every point of that model's grid
    "sasrec": dict(model="SASRec", dataset="kuairand_pure"),
    "np": dict(model="CAFREC", dataset="kuairand_pure_ctx", ablation="no_profiler"),
}
# the six-feature x_ctx without prefix_policy_flag (fires on ~1% of context rows)
NO_POLICY_FIELDS = ["prefix_session_len_log_z", "prefix_dwell_entropy_z",
                    "prefix_category_drift_z", "inter_session_gap_log_z", "is_first_session"]

QUEUES["gpu_smoke"] = [
    dict(**GRID_ARMS[arm], seed=GRID_SEED, train_batch_size=512, epochs=1,
         tag=f"smoke512_{arm}", dump_topk=False)
    for arm in GRID_ARMS
]
QUEUES["gpu_grid"] = [
    dict(**GRID_ARMS[arm], seed=GRID_SEED, train_batch_size=512, extra=cfg,
         tag=f"gs16_{arm}_{i:02d}", dump_ranks=False, dump_topk=False)
    for i, cfg in enumerate(GRID16) for arm in GRID_ARMS
]


def grid_winner(arm):
    """(index, config) of the grid point with the best validation NDCG@10."""
    best = None
    for i, cfg in enumerate(GRID16):
        job = dict(**GRID_ARMS[arm], seed=GRID_SEED, tag=f"gs16_{arm}_{i:02d}")
        path = _out_path(job)
        if not path.exists():
            raise FileNotFoundError(f"grid incomplete, run --queue gpu_grid first: {path.name}")
        score = json.loads(path.read_text())["valid_best"]["ndcg@10"]
        if best is None or score > best[0]:            # ties keep the lower index
            best = (score, i, cfg)
    return best[1], best[2]


def _tuned_queue():
    (i_s, cfg_s), (i_n, cfg_n) = grid_winner("sasrec"), grid_winner("np")
    print(f"grid winners: SASRec point {i_s:02d} {cfg_s}; CAFREC-NP point {i_n:02d} {cfg_n}")
    jobs = []
    for s in HEADLINE_SEEDS:
        jobs += [
            dict(**GRID_ARMS["sasrec"], seed=s, train_batch_size=512, extra=cfg_s,
                 tag=f"tuned16_sasrec_gs{i_s:02d}"),
            dict(**GRID_ARMS["np"], seed=s, train_batch_size=512, extra=cfg_n,
                 tag=f"tuned16_np_gs{i_n:02d}"),
            dict(**GRID_ARMS["np"], seed=s, train_batch_size=512,
                 extra=dict(cfg_n, context_fields=NO_POLICY_FIELDS, n_context_features=5),
                 tag=f"tuned16_np_nopolicy_gs{i_n:02d}"),
        ]
    return jobs


def _converge_queue():
    """Convergence check for the two winners: 30 epochs, patience 5, grid seed."""
    jobs = []
    for arm in GRID_ARMS:
        i, cfg = grid_winner(arm)
        jobs.append(dict(**GRID_ARMS[arm], seed=GRID_SEED, train_batch_size=512,
                         extra=dict(cfg, stopping_step=5), epochs=30,
                         tag=f"converge30_{arm}_gs{i:02d}", dump_topk=False))
    return jobs


QUEUES["gpu_tuned"] = _tuned_queue          # built from the grid results at run time
QUEUES["gpu_converge"] = _converge_queue

PROFILE_DIR = HERE.parent / "data" / "profiles"


def resolve_profile(name, dataset="kuairand_pure_ctx"):
    """Profile cache name -> (absolute path, profile_dim), read from the file name
    `<dataset>.profiles.<name>.d<dim>.pt` written by build_content_profiles.py."""
    hits = sorted(PROFILE_DIR.glob(f"{dataset}.profiles.{name}.d*.pt"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one {name} cache in {PROFILE_DIR}, found {hits}")
    dim = int(hits[0].name.rsplit(".d", 1)[1][:-len(".pt")])
    return str(hits[0]), dim


def _out_path(job):
    return OUT_DIR / f"{job['model']}_{job['dataset']}_{job['tag']}_seed{job['seed']}.json"


def run_one(model, dataset, seed, ablation=None, tag=None, epochs=None,
            train_batch_size=None, max_seq_len=None,
            dump_ranks=True, dump_topk=True, llm_profile_path=None, profile_dim=None,
            extra=None):
    from cafrec.runner import run_experiment

    overrides = {"seed": seed}
    if ablation is not None:
        overrides["ablation"] = ablation
    if epochs is not None:
        overrides["epochs"] = epochs
    if train_batch_size is not None:
        overrides["train_batch_size"] = train_batch_size
    if max_seq_len is not None:
        overrides["MAX_ITEM_LIST_LENGTH"] = max_seq_len
    # CAFREC only: frozen profile cache instead of the learnable stand-in (as in
    # modal_run.train); profile_dim must match the cache.
    if llm_profile_path is not None:
        overrides["llm_profile_path"] = llm_profile_path
    if profile_dim is not None:
        overrides["profile_dim"] = profile_dim
    # arbitrary RecBole keys (grid points), applied last as in modal_run.train
    if extra:
        overrides.update(extra)
    # RecBole names checkpoints by the second; a folder per process stops two
    # parallel workers from loading each other's best model at test time
    overrides.setdefault("checkpoint_dir", str(HERE / "saved" / f"pid{os.getpid()}"))
    t0 = time.time()
    metrics = run_experiment(model, dataset=dataset, config_overrides=overrides,
                             return_ranks=dump_ranks, dump_topk=dump_topk)
    metrics["wall_seconds"] = round(time.time() - t0, 1)
    metrics["device"] = f"local-gpu ({GPU_NAME})" if GPU_NAME else "local-cpu"
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
    if callable(jobs):
        jobs = jobs()
    # Several workers may run the same queue at once (one process each): a job is
    # claimed by creating <result>.lock exclusively, so no job runs twice.
    w = f"pid{os.getpid()}"
    for i, job in enumerate(jobs, 1):
        out = _out_path(job)
        if out.exists():
            _log(log, f"[{i}/{len(jobs)}] {w} SKIP (exists) {out.name}")
            continue
        lock = out.with_suffix(".lock")
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue                              # another worker has it
        _log(log, f"[{i}/{len(jobs)}] {w} START {job}")
        try:
            job = dict(job)
            profile = job.pop("profile", None)
            if profile is not None:
                job["llm_profile_path"], job["profile_dim"] = resolve_profile(profile, job["dataset"])
            m, out = run_one(**job)
            _log(log, f"[{i}/{len(jobs)}] {w} DONE  {out.name}  test={m['test']}  "
                      f"wall={m['wall_seconds']}s")
        except Exception:                         # keep going; one failure shouldn't kill the night
            _log(log, f"[{i}/{len(jobs)}] {w} FAIL  {job}\n{traceback.format_exc()}")
        finally:
            lock.unlink(missing_ok=True)
    _log(log, f"{w} QUEUE FINISHED")


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
    p.add_argument("--train-batch-size", type=int)
    p.add_argument("--max-seq-len", type=int)
    p.add_argument("--llm-profile-path")
    p.add_argument("--profile-dim", type=int)
    a = p.parse_args()
    if a.queue:
        run_queue(a.queue)
    else:
        m, out = run_one(a.model, a.dataset, a.seed, a.ablation, a.tag, a.epochs,
                         a.train_batch_size, a.max_seq_len,
                         llm_profile_path=a.llm_profile_path, profile_dim=a.profile_dim)
        print(out, m["test"], f"{m['wall_seconds']}s")


# ---------------------------------------------------------------------------
# QUEUE NOTES
#
# 2026-09-16, "overnight" job (a): CPU does not reproduce Modal on the same
# seed and config. CAFREC no_profiler, kuairand_pure_ctx, seed 42:
#
#     Modal (noprof_ms_s42)  HR@10 0.0828  NDCG@10 0.0425  MRR@10 0.0304
#     local-cpu (this file)  HR@10 0.0817  NDCG@10 0.0411  MRR@10 0.0290
#
# The local run is uniformly lower, and the NDCG/MRR gaps (-3.3%, -4.6%) are
# the same size as the ablation effects these queues are meant to measure.
# So local ablations must be compared against local controls only; do not pair
# them with results/modal/ rank dumps. Hence the "controls" queue above.
#
# CORRECTION 2026-09-17: the comparison above is confounded. The local runs used
# base.yaml's train_batch_size=2048 (read back from the saved checkpoint config),
# whereas noprof_ms_s42 ran at 512. Against Modal at the SAME batch
# (bs2048_noprof seed 42: HR 0.0802, NDCG 0.0406, MRR 0.0288) the CPU run differs by
# +1.9% / +1.2% / +0.7%, within seed-to-seed spread (sd ~0.0015 HR). CPU and GPU are
# still not bitwise identical, so keep pairing within one device where possible.
# ---------------------------------------------------------------------------
