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
            dump_ranks=True, dump_topk=True, llm_profile_path=None, profile_dim=None):
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
            job = dict(job)
            profile = job.pop("profile", None)
            if profile is not None:
                job["llm_profile_path"], job["profile_dim"] = resolve_profile(profile, job["dataset"])
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
