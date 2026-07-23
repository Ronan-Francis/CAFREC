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
          train_batch_size: int = None) -> dict:
    from cafrec.runner import run_experiment

    overrides = {"data_path": "/data/recbole"}
    if epochs is not None:
        overrides["epochs"] = epochs
    if seed is not None:
        overrides["seed"] = seed
    if train_batch_size is not None:
        overrides["train_batch_size"] = train_batch_size
    elif dataset != "kuairand_pure":
        # CE loss materialises a (batch x n_items) logits tensor. Pure's 7.5K
        # items are fine at batch 2048, but 1k/27k have million-item catalogues
        # -> 2048 OOMs a 16GB T4. 512 keeps the tensor under ~4GB.
        overrides["train_batch_size"] = 512

    metrics = run_experiment(
        model_key,
        dataset=dataset,
        config_overrides=overrides,
        base_config=REMOTE_BASE_CONFIG,
    )

    out = f"/results/{model_key}_{dataset}_seed{metrics['seed']}_{metrics['timestamp']}.json"
    with open(out, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    results_vol.commit()
    metrics["results_file"] = out
    return metrics


ALL_DATASETS = ["kuairand_pure", "kuairand_1k", "kuairand_27k"]


@app.local_entrypoint()
def main(models: str = "SASRec", dataset: str = "kuairand_pure",
         epochs: int = None, seed: int = None, train_batch_size: int = None):
    """`models` and `dataset` are comma-separated; `--dataset all` sweeps
    every dataset in the recbole folder (pure, 1k, 27k)."""
    keys = [m.strip() for m in models.split(",") if m.strip()]
    datasets = (ALL_DATASETS if dataset.strip().lower() == "all"
                else [d.strip() for d in dataset.split(",") if d.strip()])
    t0 = time.time()
    for ds in datasets:
        for key in keys:
            metrics = train.remote(key, ds, epochs=epochs, seed=seed,
                                   train_batch_size=train_batch_size)
            print(f"\n[{key} / {ds}] test: {metrics['test']}")
            print(f"  split_sizes: {metrics['split_sizes']}")
            print(f"  valid_best : {metrics['valid_best']}")
            print(f"  saved to   : {metrics['results_file']} (cafrec-results volume)")
    print(f"\nTotal wall time: {time.time() - t0:.0f}s")
