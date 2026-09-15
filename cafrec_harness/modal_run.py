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
    # CAFREC ablations (T3.1): none | no_profiler | static_gate | concat.
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
    # Full-ranking eval materialises a [eval_batch_size x n_items] matrix. Pure's
    # 7.2K items are fine at the base 4096, but 1k/27k's huge catalogues OOM the
    # A10G there (13.5GB+ tensor) -> shrink the eval batch for those tiers.
    if eval_batch_size is not None:
        overrides["eval_batch_size"] = eval_batch_size
    elif dataset != "kuairand_pure":
        overrides["eval_batch_size"] = 256

    if train_batch_size is not None:
        overrides["train_batch_size"] = train_batch_size
    elif dataset != "kuairand_pure":
        # CE loss materialises a (batch x n_items) logits tensor. Pure's 7.5K
        # items are fine at batch 2048, but 1k/27k have million-item catalogues
        # -> 2048 OOMs a 16GB T4. 512 keeps the tensor under ~4GB.
        overrides["train_batch_size"] = 512

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
    t0 = time.time()
    for ds in datasets:
        for key in keys:
            metrics = train.remote(key, ds, epochs=epochs, seed=seed,
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
