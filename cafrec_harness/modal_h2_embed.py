"""Embed the generated H2 profiles into the frozen cache CAFREC loads (2026-09-21).

Takes `/data/h2/profiles.<variant>.jsonl` from `modal_h2_generate.py` and writes
`/data/profiles/kuairand_pure_ctx.profiles.llm_<variant>.d1024.pt` plus a sidecar,
in the same contract as every other profile cache: `[n_users, profile_dim]`
float32, row i == RecBole internal user id i, row 0 = padding, users without a
profile all-zero.

The encoder is `BAAI/bge-large-en-v1.5`, the SAME one that produced the template
profile (`.profiles.hf.d1024.pt`). That is the point: the H2 comparison is
LLM-written text vs template-written text through one encoder, so any difference
is the text, not the embedding model. (The content profiles used
`bge-large-zh-v1.5` because they embedded Chinese captions directly; these
profiles are English by construction.)

    modal run modal_h2_embed.py --variant plain
    modal run modal_h2_embed.py --variant kar --batch-size 128
"""
import json
import time

import modal

app = modal.App("cafrec-h2-embed")

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
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_HOME": "/hf"})
    .add_local_python_source("cafrec")
    .add_local_dir("configs", remote_path="/root/configs")
)

data_vol = modal.Volume.from_name("cafrec-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

REMOTE_BASE_CONFIG = "/root/configs/base.yaml"
VARIANTS = ("plain", "kar", "collab", "support")


@app.function(
    image=image,
    gpu="A10G",
    cpu=8.0,
    memory=32768,
    timeout=2 * 60 * 60,
    volumes={"/data": data_vol, "/hf": hf_vol},
)
def embed(variant: str = "plain", dataset: str = "kuairand_pure_ctx",
          model: str = "BAAI/bge-large-en-v1.5",
          batch_size: int = 128, min_chars: int = 40) -> dict:
    import hashlib
    import os

    import numpy as np
    import torch

    from cafrec.features.build_profiles import HFBackend, load_user_item_remap
    from cafrec.features.profiles import save_profiles, validate_profiles

    t0 = time.time()
    data_vol.reload()
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")
    TAG = f"llm_{variant}"
    PROFILES_JSONL = f"/data/h2/profiles.{variant}.jsonl"
    if not os.path.exists(PROFILES_JSONL):
        raise FileNotFoundError(
            f"{PROFILES_JSONL} missing — run "
            f"`modal run modal_h2_generate.py --variant {variant}` first")

    rows = [json.loads(ln) for ln in open(PROFILES_JSONL, encoding="utf-8")]
    # a resumed generation appends, so the same user can appear twice; last wins
    by_user = {r["user_id"]: r for r in rows}
    print(f"  {len(rows):,} lines, {len(by_user):,} distinct users", flush=True)
    # a smoke run with one model followed by a full run with another leaves a
    # mixed corpus, which would confound the condition it is meant to define
    generators = sorted({r.get("model", "unknown") for r in by_user.values()})
    if len(generators) > 1:
        raise ValueError(
            f"profiles.jsonl mixes generators {generators}. Regenerate with "
            f"`modal run modal_h2_generate.py --overwrite --model <one of them>`.")

    # Remap FIRST (cheap): a mismatch here means the prompts were built against a
    # different .inter, and must fail before the model downloads.
    uid_token2id, _iid, n_users = load_user_item_remap(
        dataset, REMOTE_BASE_CONFIG, data_path="/data/recbole")

    user_ids, texts, n_short, n_unmapped, n_truncated = [], [], 0, 0, 0
    for tok, r in by_user.items():
        internal = uid_token2id.get(str(tok))
        if internal is None:
            n_unmapped += 1
            continue
        if "uid" in r and int(r["uid"]) != int(internal):
            raise ValueError(
                f"user {tok}: prompts say internal id {r['uid']}, this RecBole remap "
                f"says {internal} — prompts and dataset disagree, rebuild prompts")
        text = (r.get("text") or "").strip()
        if len(text) < min_chars:          # empty or degenerate generation
            n_short += 1
            continue
        if r.get("finish_reason") == "length":
            n_truncated += 1
        user_ids.append(int(internal))
        texts.append(text)
    print(f"  embedding {len(texts):,} profiles; {n_short:,} too short to use, "
          f"{n_unmapped:,} not in the RecBole remap, {n_truncated:,} hit the token limit",
          flush=True)

    backend = HFBackend(model_name=model, device="cuda", batch_size=batch_size)
    vecs = backend.encode(texts)
    profile_dim = backend.dim

    cache = torch.zeros(n_users, profile_dim, dtype=torch.float32)
    cache[torch.tensor(user_ids, dtype=torch.long)] = torch.from_numpy(np.asarray(vecs, np.float32))
    cache[0] = 0.0
    validate_profiles(cache, n_users, profile_dim)

    out_dir = "/data/profiles"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{dataset}.profiles.{TAG}.d{profile_dim}.pt")
    save_profiles(path, cache, n_users, profile_dim)

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    nonzero = (cache.abs().sum(dim=1) > 0)
    gen_report = {}
    rep_path = f"/data/h2/generation_report.{variant}.json"
    if os.path.exists(rep_path):
        gen_report = json.load(open(rep_path))
    lens = np.array([len(t) for t in texts]) if texts else np.zeros(1)

    meta = dict(
        name=TAG, variant=variant, registered=(variant == "plain"),
        path=path, dataset=dataset, n_users=n_users, profile_dim=profile_dim,
        nonzero_users=int(nonzero.sum()),
        nonzero_share_excl_padding=float(nonzero[1:].float().mean()),
        sha256=h.hexdigest(), built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        script="modal_h2_embed.py", encoder=model, batch_size=batch_size,
        kind="L2-normalised embedding of an LLM-generated English temporal profile",
        profiles_jsonl=PROFILES_JSONL, distinct_users_in_jsonl=len(by_user),
        embedded=len(texts), skipped_short=n_short, skipped_unmapped=n_unmapped,
        generations_hitting_token_limit=n_truncated,
        profile_chars=dict(mean=float(lens.mean()), median=float(np.median(lens)),
                           min=int(lens.min()), max=int(lens.max())),
        generation=gen_report, seconds=round(time.time() - t0, 1),
    )
    with open(path.replace(".pt", ".json"), "w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)
    data_vol.commit()
    meta["size_mb"] = round(os.path.getsize(path) / 1e6, 2)
    return meta


@app.local_entrypoint()
def main(variant: str = "plain", dataset: str = "kuairand_pure_ctx",
         model: str = "BAAI/bge-large-en-v1.5",
         batch_size: int = 128, min_chars: int = 40):
    meta = embed.remote(variant=variant, dataset=dataset, model=model,
                        batch_size=batch_size, min_chars=min_chars)
    print("\n--- H2 profile cache (Modal) ---")
    for k, v in meta.items():
        if k != "generation":
            print(f"  {k:32s}: {v}")
    print(f"\nCache: {meta['path']}  ({meta['nonzero_users']:,} non-zero users)")
    print("\nNext — the four-condition evaluation:")
    print("  modal run modal_run.py::h2       # all arms that have a cache")
