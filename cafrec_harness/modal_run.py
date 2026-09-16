"""Run the CAFREC harness on Modal (cloud GPU).

One-time setup
--------------
    conda activate recbole
    pip install modal
    modal setup                      # browser login (one time)

    # upload the RecBole atomic files to a persistent Modal volume (one time
    # per dataset; re-run after rebuilding an .inter file):
    modal volume put cafrec-data ../data/recbole/kuairand_pure/kuairand_pure.inter recbole/kuairand_pure/kuairand_pure.inter
    modal volume put cafrec-data ../data/recbole/kuairand_1k/kuairand_1k.inter     recbole/kuairand_1k/kuairand_1k.inter
    modal volume put cafrec-data ../data/recbole/kuairand_27k/kuairand_27k.inter   recbole/kuairand_27k/kuairand_27k.inter

Run
---
    modal run modal_run.py --models SASRec --dataset kuairand_pure --epochs 1   # sanity
    modal run modal_run.py --models "SASRec,HGN,CAFREC" --dataset kuairand_1k   # real runs

Results are printed and also written as JSON to the `cafrec-results` volume
(browse with `modal volume ls cafrec-results`).
"""
import json
import time

import modal

app = modal.App("cafrec-harness")

# Mirror the local recbole env (torch 2.8 / recbole 1.2.1 / numpy<2), but with
# CUDA torch. The cafrec package and configs/ ship with the image on each run,
# so local code edits apply without any extra deploy step.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.8.0",
        "recbole==1.2.1",
        "numpy<2",
        "scipy>=1.7",
        "pandas",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("cafrec")
    .add_local_dir("configs", remote_path="/root/configs")
)

data_vol = modal.Volume.from_name("cafrec-data", create_if_missing=True)
results_vol = modal.Volume.from_name("cafrec-results", create_if_missing=True)

REMOTE_BASE_CONFIG = "/root/configs/base.yaml"


@app.function(
    image=image,
    gpu="A10G",  # 24GB; T4's 16GB OOMs on the million-item catalogues (CE loss)
    cpu=8.0,          # RecBole's dataloader/augmentation is CPU-bound
    memory=65536,     # MiB; 27k preprocessing materialises ~18GB+ of sequences
    timeout=6 * 60 * 60,
    volumes={"/data": data_vol, "/results": results_vol},
)
def train(model_key: str, dataset: str, epochs: int = None, seed: int = None,
          train_batch_size: int = None, eval_batch_size: int = None,
          llm_profile_path: str = None, profile_dim: int = None,
          ablation: str = None, dump_ranks: bool = False,
          dump_topk: bool = False, tag: str = None,
          context_fields: str = None, max_seq_len: int = None,
          history_gate: bool = False, history_gate_mode: str = None,
          extra_overrides: dict = None) -> dict:
    from cafrec.runner import run_experiment

    overrides = {"data_path": "/data/recbole"}
    if epochs is not None:
        overrides["epochs"] = epochs
    if seed is not None:
        overrides["seed"] = seed
    # CAFREC only: load the frozen LLM profile cache (built by modal_profiles.py)
    # instead of the learnable z_long stand-in. profile_dim must match the cache.
    if llm_profile_path is not None:
        overrides["llm_profile_path"] = llm_profile_path
    if profile_dim is not None:
        overrides["profile_dim"] = profile_dim
    # CAFREC ablations: none | no_profiler | static_gate | concat, plus the
    # CAFREC-NP gate controls np_vector_gate | np_shuffled_ctx (see
    # cafrec/models/cafrec.py). Passed straight through as an override.
    if ablation is not None:
        overrides["ablation"] = ablation
    # RON-40 x_ctx-set ablation: override the gate's input feature set. Pass a
    # comma-separated subset of the six context columns (e.g. the classic-four
    # R^4 set, dropping inter_session_gap_log_z + is_first_session); the gate MLP
    # input width n_context_features is derived from the count. The _ctx load_col
    # still loads all six columns; the model just selects these by name.
    if context_fields is not None:
        fields = [f.strip() for f in context_fields.split(",") if f.strip()]
        overrides["context_fields"] = fields
        overrides["n_context_features"] = len(fields)
    # Sequence-length sensitivity: override the short-term encoder window (also
    # sizes the position-embedding table). Default lives in base.yaml (=20).
    if max_seq_len is not None:
        overrides["MAX_ITEM_LIST_LENGTH"] = max_seq_len
    # RON-45 history-gated profiler: gate also sees history length (H2 fix).
    if history_gate:
        overrides["history_gate"] = True
    if history_gate_mode is not None:
        overrides["history_gate_mode"] = history_gate_mode
    # Batch sizes default per catalogue TIER, never per dataset name: the old
    # `dataset != "kuairand_pure"` test put kuairand_pure_ctx in the large-
    # catalogue branch and caused the H1 batch-size confound. See cafrec/tiers.py.
    # Every Pure variant now trains at base.yaml's 2048 unless told otherwise;
    # to pair with the pre-fix CAFREC corpus (512), pass train_batch_size=512.
    from cafrec.tiers import batch_size_overrides
    overrides.update(batch_size_overrides(dataset))
    if train_batch_size is not None:
        overrides["train_batch_size"] = train_batch_size
    if eval_batch_size is not None:
        overrides["eval_batch_size"] = eval_batch_size

    # RON-31 grid search: arbitrary RecBole keys (learning_rate, weight_decay,
    # hidden_size, n_layers, n_heads, dropout probs, pooling_type, ...). Applied
    # LAST so a sweep config wins over every convenience flag above.
    if extra_overrides:
        overrides.update(extra_overrides)

    metrics = run_experiment(
        model_key,
        dataset=dataset,
        config_overrides=overrides,
        base_config=REMOTE_BASE_CONFIG,
        return_ranks=dump_ranks,
        dump_topk=dump_topk,
    )

    # `tag` labels the run condition (e.g. prof7b / standin / no_profiler) so
    # multiple CAFREC runs on the same dataset are distinguishable on the volume.
    suffix = f"_{tag}" if tag else ""
    out = (f"/results/{model_key}_{dataset}{suffix}_seed{metrics['seed']}"
           f"_{metrics['timestamp']}.json")
    with open(out, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    results_vol.commit()
    metrics["results_file"] = out
    # keep the returned dict lean over the wire; ranks/top-k are persisted in the JSON
    for big in ("test_ranks", "test_user_ids",
                "test_topk_items", "test_topk_user_ids",
                "fusion_lambda", "fusion_lambda_user_tokens"):
        metrics.pop(big, None)
    return metrics


@app.function(
    image=image,
    gpu="A10G",
    # RON-31 cost fix (2026-09-10): the `train` reservations above (8 CPU /
    # 64 GiB) are sized for the 27k tier's million-item catalogue, and Modal
    # bills reserved CPU+memory ON TOP of GPU time. Pure has ~7.2K items and
    # never comes close to that footprint, so a Pure-only sweep was paying a
    # large multiple of its GPU cost for idle reservation. CPU stays at 4 (the
    # RecBole dataloader is CPU-bound; starving it would idle the GPU and cost
    # MORE), memory drops 64 GiB -> 16 GiB.
    cpu=4.0,
    memory=16384,
    timeout=2 * 60 * 60,
    volumes={"/data": data_vol, "/results": results_vol},
)
def train_pure(**kwargs) -> dict:
    """Pure-tier trainer: identical body to `train`, trimmed reservations.

    `.local()` invokes the undecorated function in this container, so the two
    entry points can never drift apart.
    """
    return train.local(**kwargs)


# RON-?? 2026-09-16: GPU re-run of the 2026-09-15 overnight local CPU queue.
# Those seven ran via local_run.py on base.yaml defaults (train_batch_size=2048)
# and so pair with NOTHING in results/modal/, whose CAFREC runs went down the
# (since fixed) `dataset != "kuairand_pure"` branch and trained at 512. Re-run here
# on the identical code path as noprof_ms/ff_*/tuned so the rank dumps pair
# user-for-user. train_batch_size=512 was the implicit default when these ran;
# after the tier fix the default is 2048, so it is pinned explicitly.
RERUN_JOBS = [
    dict(ablation="no_profiler",     seed=42,  tag="gpu_noprof"),
    dict(ablation="np_vector_gate",  seed=42,  tag="gpu_vector_gate"),
    dict(ablation="np_vector_gate",  seed=77,  tag="gpu_vector_gate"),
    dict(ablation="np_vector_gate",  seed=123, tag="gpu_vector_gate"),
    dict(ablation="np_shuffled_ctx", seed=42,  tag="gpu_shuffled_ctx"),
    dict(ablation="np_shuffled_ctx", seed=77,  tag="gpu_shuffled_ctx"),
    dict(ablation="np_shuffled_ctx", seed=123, tag="gpu_shuffled_ctx"),
]
for _job in RERUN_JOBS:
    _job["train_batch_size"] = 512


@app.local_entrypoint()
def rerun(dataset: str = "kuairand_pure_ctx"):
    """Fan the seven overnight conditions out in parallel on train_pure.

    train_pure (4 CPU / 16 GiB), not train (8 CPU / 64 GiB): this is the Pure
    tier, and `main` never got switched over after the RON-31 cost fix.
    """
    t0 = time.time()
    # Budget guard: train_pure's decorator allows 2h, and seven of those running
    # in parallel on A10G would cost more than the account holds. Pure-tier runs
    # land in ~10 min, so 45 min is generous while capping the worst case.
    fn = train_pure.with_options(timeout=45 * 60)
    calls = []
    for job in RERUN_JOBS:
        calls.append((job, fn.spawn(model_key="CAFREC", dataset=dataset,
                                    dump_ranks=True, dump_topk=True, **job)))
        print(f"  spawned {job['tag']} seed{job['seed']}", flush=True)
    print(f"\n{len(calls)} jobs in flight\n", flush=True)
    for job, c in calls:
        try:
            m = c.get()
            print(f"[{job['tag']} seed{job['seed']}] {m['test']}  -> {m['results_file']}", flush=True)
        except Exception as e:
            print(f"[{job['tag']} seed{job['seed']}] FAILED: {e}", flush=True)
    print(f"\nTotal wall: {time.time() - t0:.0f}s")

# RON-?? 2026-09-16: batch-size confound check for the H1 headline.
# SASRec runs on dataset "kuairand_pure" and CAFREC on "kuairand_pure_ctx".
# The `dataset != "kuairand_pure"` branch in `train` (since replaced by
# cafrec/tiers.py) therefore gave SASRec
# train_batch_size 2048 (base.yaml) and CAFREC 512 -- so "matched default
# hyperparameters" and "identical training schedule" may not hold for the
# paper's primary comparison. The result JSONs record no hyperparameters and
# the launch commands were never logged, so this can only be settled by
# experiment. Cross the two settings and see whether the margin survives.
HEADLINE_SEEDS = [42, 77, 123, 256, 512, 1024, 2048]
BSWEEP_JOBS = (
    [dict(model_key="SASRec", dataset="kuairand_pure", seed=s,
          train_batch_size=512, tag="bs512_sasrec") for s in HEADLINE_SEEDS]
    + [dict(model_key="CAFREC", dataset="kuairand_pure_ctx", ablation="no_profiler",
            seed=s, train_batch_size=2048, tag="bs2048_noprof") for s in HEADLINE_SEEDS]
)


@app.local_entrypoint()
def bsweep():
    """Cross SASRec and CAFREC-NP over the two train_batch_size settings."""
    t0 = time.time()
    fn = train_pure.with_options(timeout=45 * 60)
    calls = []
    for job in BSWEEP_JOBS:
        calls.append((job, fn.spawn(dump_ranks=True, dump_topk=True, **job)))
        print("  spawned " + job["tag"] + " seed" + str(job["seed"]), flush=True)
    print(str(len(calls)) + " jobs in flight", flush=True)
    for job, c in calls:
        try:
            m = c.get()
            print("[" + job["tag"] + " seed" + str(job["seed"]) + "] "
                  + str(m["test"]) + "  -> " + m["results_file"], flush=True)
        except Exception as e:
            print("[" + job["tag"] + " seed" + str(job["seed"]) + "] FAILED: " + str(e), flush=True)
    print("Total wall: %.0fs" % (time.time() - t0))


ALL_DATASETS = ["kuairand_pure", "kuairand_1k", "kuairand_27k"]


@app.local_entrypoint()
def main(models: str = "SASRec", dataset: str = "kuairand_pure",
         epochs: int = None, seed: int = None, train_batch_size: int = None,
         eval_batch_size: int = None, llm_profile_path: str = None,
         profile_dim: int = None, ablation: str = None, dump_ranks: bool = False,
         dump_topk: bool = False, tag: str = None, context_fields: str = None,
         max_seq_len: int = None, history_gate: bool = False,
         history_gate_mode: str = None):
    """`models` and `dataset` are comma-separated; `--dataset all` sweeps
    every dataset in the recbole folder (pure, 1k, 27k). `--llm-profile-path`
    (+ `--profile-dim`) loads CAFREC's frozen profile cache from the volume.
    `--dump-ranks` persists per-user held-out ranks (for significance tests);
    `--tag` labels the result file with the run condition."""
    keys = [m.strip() for m in models.split(",") if m.strip()]
    datasets = (ALL_DATASETS if dataset.strip().lower() == "all"
                else [d.strip() for d in dataset.split(",") if d.strip()])
    from cafrec.tiers import catalogue_tier

    t0 = time.time()
    for ds in datasets:
        # Pure-tier work goes to train_pure's trimmed reservations (RON-31);
        # `train`'s 8 CPU / 64 GiB is sized for the million-item tiers.
        fn = train_pure if catalogue_tier(ds) == "small" else train
        for key in keys:
            metrics = fn.remote(model_key=key, dataset=ds, epochs=epochs, seed=seed,
                                   train_batch_size=train_batch_size,
                                   eval_batch_size=eval_batch_size,
                                   llm_profile_path=llm_profile_path,
                                   profile_dim=profile_dim, ablation=ablation,
                                   dump_ranks=dump_ranks, dump_topk=dump_topk,
                                   tag=tag, context_fields=context_fields,
                                   max_seq_len=max_seq_len, history_gate=history_gate,
                                   history_gate_mode=history_gate_mode)
            print(f"\n[{key} / {ds}] test: {metrics['test']}")
            print(f"  split_sizes: {metrics['split_sizes']}")
            print(f"  valid_best : {metrics['valid_best']}")
            print(f"  saved to   : {metrics['results_file']} (cafrec-results volume)")
    print(f"\nTotal wall time: {time.time() - t0:.0f}s")
