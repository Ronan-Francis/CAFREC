"""Build the CAFREC LLM temporal-profile cache on Modal GPU (RON-24).

Runs the offline embedding profiler (`cafrec.features.build_profiles`) over a
`_ctx` dataset's users and writes the frozen `[n_users, profile_dim]` cache to
the `cafrec-data` volume, so `modal_run.py` can load it into CAFREC.

    conda activate recbole
    # 1k_ctx.inter must already be on the volume (modal volume put ...).
    modal run modal_profiles.py --dataset kuairand_1k_ctx \
        --model BAAI/bge-large-en-v1.5

The model is a robust 1024-dim embedder by default; pass a larger one (e.g.
intfloat/e5-mistral-7b-instruct, 4096-d) to scale up — it fits the A10G. HF
weights are cached in the `hf-cache` volume, so a re-run does not re-download.
"""
import modal

app = modal.App("cafrec-profiler")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.8.0",
        "recbole==1.2.1",
        "numpy<2",
        "scipy>=1.7",
        "pandas",
        "sentence-transformers>=3.0",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "HF_HOME": "/hf"})
    .add_local_python_source("cafrec")
    .add_local_dir("configs", remote_path="/root/configs")
)

data_vol = modal.Volume.from_name("cafrec-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

REMOTE_BASE_CONFIG = "/root/configs/base.yaml"


@app.function(
    image=image,
    gpu="A10G",       # 24GB — fits a 7B embedder in fp16 if you scale the model up
    cpu=8.0,
    memory=32768,
    timeout=2 * 60 * 60,
    volumes={"/data": data_vol, "/hf": hf_vol},
)
def profile(dataset: str, model: str = "BAAI/bge-large-en-v1.5",
            dtype: str = None, batch_size: int = 64) -> dict:
    from cafrec.features.build_profiles import build, HFBackend

    # Remap + text rendering run first (cheap); the model loads only after, so a
    # data/remap error fails before any multi-GB weight download.
    stats = build(
        dataset,
        backend_factory=lambda: HFBackend(model_name=model, device="cuda",
                                          dtype=dtype, batch_size=batch_size),
        base_config=REMOTE_BASE_CONFIG,
        inter_root="/data/recbole",
        out_dir="/data/profiles",
        data_path="/data/recbole",
        write=True,
    )
    data_vol.commit()
    hf_vol.commit()
    stats["model"] = model
    return stats


@app.local_entrypoint()
def main(dataset: str = "kuairand_1k_ctx", model: str = "BAAI/bge-large-en-v1.5",
         dtype: str = None, batch_size: int = 64):
    stats = profile.remote(dataset, model=model, dtype=dtype, batch_size=batch_size)
    print("\n--- profile build (Modal) ---")
    for k, v in stats.items():
        print(f"  {k:16s}: {v}")
    print(f"\nCache on cafrec-data volume: {stats.get('path')}")
    print(f"Load into CAFREC with:  --llm_profile_path {stats.get('path')} "
          f"--profile_dim {stats.get('profile_dim')}")
