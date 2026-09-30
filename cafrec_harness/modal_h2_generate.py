"""Generate the LLM temporal profiles for the H2 test on Modal (2026-09-21).

Reads `/data/h2/prompts.jsonl` (built by `build_h2_prompts.py`), runs offline
batch inference with vLLM on one A10G, and writes `/data/h2/profiles.jsonl` —
one English profile paragraph per user, row-keyed by the RecBole user token.
`modal_h2_embed.py` then turns those into the frozen `[n_users, 1024]` cache
CAFREC loads.

Choices that matter for the paper
---------------------------------
* **Qwen2.5-7B-Instruct.** The prompts contain Chinese captions and ask for an
  English profile, so the generator has to be genuinely bilingual; a 7B fits an
  A10G in bf16 with room for KV cache.
* **Greedy decoding, fixed seed.** temperature 0, so the profile corpus is a
  deterministic function of the prompts and the model revision, and the run can
  be reproduced or resumed without changing earlier profiles.
* **Resumable.** Profiles are written shard by shard and the volume committed
  after each, so a timeout costs one shard, not the run. Re-running skips users
  already present unless `--overwrite`.
* **Cost.** Prefill dominates (roughly 20M input tokens against ~3M generated).
  ALWAYS run `--limit 64` first: it prints measured throughput and a projected
  full-run time and cost, so the spend is a decision rather than a surprise.

    modal run modal_h2_generate.py --variant plain --limit 64   # smoke + projection
    modal run modal_h2_generate.py --variant plain              # full run
    modal run modal_h2_generate.py --variant kar                # exploratory arm

Inspect a few before spending on training runs:
    python -m modal volume get cafrec-data h2/profiles.plain.jsonl ../data/h2/
"""
import json
import time

import modal

app = modal.App("cafrec-h2-generate")

# vLLM is deliberately unpinned: it ships its own torch build and pinning a
# version this environment has never resolved is the more likely failure. The
# resolved version and the model revision are recorded in the run report, so the
# run can be pinned exactly once it has succeeded.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "huggingface_hub[hf_transfer]")
    .env({"HF_HOME": "/hf",
          "HF_HUB_ENABLE_HF_TRANSFER": "1",
          # vLLM's startup kernel warmup calls FlashInfer's sampler, which JIT
          # compiles a top-k kernel and so needs nvcc — absent from debian_slim,
          # which failed the first run with "Could not find nvcc and default
          # cuda_home='/usr/local/cuda' doesn't exist". Decoding here is greedy
          # (temperature 0, top_p 1), so the FlashInfer sampler contributes
          # nothing: switching to vLLM's native PyTorch sampling path is a
          # cheaper fix than a multi-GB CUDA devel base image. If a JIT failure
          # ever reappears from the ATTENTION backend, add
          # VLLM_ATTENTION_BACKEND=FLASH_ATTN here rather than nvcc.
          "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "VLLM_LOGGING_LEVEL": "WARNING"})
)

data_vol = modal.Volume.from_name("cafrec-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

VARIANTS = ("plain", "kar", "collab", "support")


def _paths(variant):
    """One corpus per prompt variant; `plain` is the registered H2 arm."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")
    return (f"/data/h2/prompts.{variant}.jsonl",
            f"/data/h2/prompts_meta.{variant}.json",
            f"/data/h2/profiles.{variant}.jsonl",
            f"/data/h2/generation_report.{variant}.json")

# Modal A10G list price at the time of writing, for the projection only.
A10G_USD_PER_HOUR = 1.10


@app.function(
    image=image,
    gpu="A10G",
    cpu=4.0,
    memory=16384,
    timeout=5 * 60 * 60,
    volumes={"/data": data_vol, "/hf": hf_vol},
)
def generate(variant: str = "plain", model: str = "Qwen/Qwen2.5-7B-Instruct",
             limit: int = 0,
             max_model_len: int = 4096, max_tokens: int = 200,
             shard_size: int = 4000, overwrite: bool = False,
             gpu_memory_utilization: float = 0.90, seed: int = 20260921) -> dict:
    import os

    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    t_start = time.time()
    data_vol.reload()
    PROMPTS, META, OUT, REPORT = _paths(variant)
    if not os.path.exists(PROMPTS):
        raise FileNotFoundError(
            f"{PROMPTS} missing — build it locally and upload:\n"
            f"  .venv/Scripts/python build_h2_prompts.py --variant {variant}\n"
            f"  python -m modal volume put cafrec-data "
            f"../data/h2/prompts.{variant}.jsonl h2/prompts.{variant}.jsonl")

    import hashlib

    def psha(prompt):
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]

    rows = [json.loads(ln) for ln in open(PROMPTS, encoding="utf-8")]
    want = {r["user_id"]: psha(r["prompt"]) for r in rows}
    done, stale = {}, 0
    if os.path.exists(OUT) and not overwrite:
        for ln in open(OUT, encoding="utf-8"):
            r = json.loads(ln)
            # Resume only where the PROMPT is unchanged. Editing the instruction
            # and re-running would otherwise leave a corpus generated from two
            # different prompts, which the generator-mix check cannot see because
            # the model is identical. Changed prompts are silently regenerated.
            # `r.get("psha") and ...` would be wrong: a profile written before
            # this field existed has no hash, cannot be shown to match the
            # current prompt, and so must count as stale. Treating "no hash" as
            # "fine" is what let 64 pre-gloss smoke profiles into the 2026-09-21
            # plain corpus.
            if r.get("psha") != want.get(r["user_id"]):
                stale += 1
                continue
            done[r["user_id"]] = r
        print(f"  resuming: {len(done):,} profiles reusable, "
              f"{stale:,} discarded as built from an earlier prompt", flush=True)
        if stale:
            # those lines are still in the file; rewrite it without them
            with open(OUT, "w", encoding="utf-8") as fh:
                for r in done.values():
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            data_vol.commit()
    todo = [r for r in rows if r["user_id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"  {len(rows):,} prompts, {len(todo):,} to generate", flush=True)
    if not todo:
        return dict(status="nothing to do", n_prompts=len(rows), n_done=len(done))

    tok = AutoTokenizer.from_pretrained(model)
    system = json.load(open(META, encoding="utf-8"))["system"] \
        if os.path.exists(META) else \
        "You write concise, factual user profiles for a video recommender system."

    def chat(text):
        return tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True)

    # Trim the oldest listed videos out of any prompt that would overflow the
    # context. Dropping whole "- [category] caption" lines keeps the prompt
    # well-formed; the category-mix sentence still covers the full history.
    budget = max_model_len - max_tokens - 16
    texts = [chat(r["prompt"]) for r in todo]
    lengths = [len(ids) for ids in tok(texts).input_ids]   # one batched pass
    over = [i for i, n in enumerate(lengths) if n > budget]
    for i in over:
        lines = todo[i]["prompt"].split("\n")
        first = next((j for j, ln in enumerate(lines) if ln.startswith("- [")), None)
        while (first is not None and first < len(lines)
               and lines[first].startswith("- [")):
            del lines[first]
            texts[i] = chat("\n".join(lines))
            if len(tok(texts[i]).input_ids) <= budget:
                break
    n_trimmed = len(over)
    if n_trimmed:
        print(f"  trimmed {n_trimmed:,} over-long prompts to fit {max_model_len}", flush=True)

    llm = LLM(model=model, max_model_len=max_model_len, seed=seed,
              gpu_memory_utilization=gpu_memory_utilization, enforce_eager=False)
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, seed=seed)
    t_model = time.time()
    print(f"  model loaded in {t_model - t_start:.0f}s ({vllm.__version__})", flush=True)

    n_in = n_out = 0
    t_gen0 = time.time()
    mode = "w" if overwrite else "a"
    for start in range(0, len(todo), shard_size):
        shard, shard_texts = todo[start:start + shard_size], texts[start:start + shard_size]
        outs = llm.generate(shard_texts, params)
        with open(OUT, mode, encoding="utf-8") as fh:
            for r, o in zip(shard, outs):
                n_in += len(o.prompt_token_ids)
                n_out += len(o.outputs[0].token_ids)
                fh.write(json.dumps(dict(user_id=r["user_id"], uid=r["uid"],
                                         n_train=r["n_train"], model=model,
                                         psha=psha(r["prompt"]),
                                         text=o.outputs[0].text.strip(),
                                         finish_reason=o.outputs[0].finish_reason),
                                    ensure_ascii=False) + "\n")
        mode = "a"
        data_vol.commit()
        el = time.time() - t_gen0
        print(f"  {min(start + shard_size, len(todo)):,}/{len(todo):,} in {el:.0f}s "
              f"({n_out / max(el, 1):.0f} out tok/s, {n_in / max(el, 1):.0f} in tok/s)",
              flush=True)

    gen_sec = time.time() - t_gen0
    per_user = gen_sec / len(todo)
    remaining = len(rows) - len(done) - len(todo)
    report = dict(
        variant=variant, registered=(variant == "plain"),
        model=model, vllm=vllm.__version__, seed=seed, temperature=0.0,
        max_model_len=max_model_len, max_tokens=max_tokens,
        n_prompts=len(rows), n_generated=len(todo), n_already_done=len(done),
        n_trimmed=n_trimmed, n_discarded_stale=stale,
        input_tokens=n_in, output_tokens=n_out,
        model_load_seconds=round(t_model - t_start, 1),
        generate_seconds=round(gen_sec, 1),
        seconds_per_user=round(per_user, 3),
        total_seconds=round(time.time() - t_start, 1),
        finished_at=time.strftime("%Y-%m-%d %H:%M:%S"), out=OUT,
    )
    if remaining > 0:
        proj_sec = per_user * remaining
        report["projection"] = dict(
            users_remaining=remaining,
            projected_generate_hours=round(proj_sec / 3600, 2),
            projected_gpu_usd=round(A10G_USD_PER_HOUR * proj_sec / 3600, 2),
            note="GPU list price only; Modal also bills the CPU/memory reservation")
    with open(REPORT, "w") as fh:
        json.dump(report, fh, indent=2)
    data_vol.commit()
    return report


@app.local_entrypoint()
def main(variant: str = "plain", model: str = "Qwen/Qwen2.5-7B-Instruct", limit: int = 0,
         max_model_len: int = 4096, max_tokens: int = 200, shard_size: int = 4000,
         overwrite: bool = False, gpu_memory_utilization: float = 0.90,
         seed: int = 20260921):
    rep = generate.remote(variant=variant, model=model, limit=limit,
                          max_model_len=max_model_len,
                          max_tokens=max_tokens, shard_size=shard_size,
                          overwrite=overwrite,
                          gpu_memory_utilization=gpu_memory_utilization, seed=seed)
    print("\n--- H2 generation (Modal) ---")
    print(json.dumps(rep, indent=2))
    if rep.get("projection"):
        p = rep["projection"]
        print(f"\nPROJECTION for the remaining {p['users_remaining']:,} users: "
              f"~{p['projected_generate_hours']}h, ~${p['projected_gpu_usd']} of GPU time.")
        print(f"Re-run `modal run modal_h2_generate.py --variant {variant}` to continue (it resumes where this stopped).")
    else:
        print("\nNext:")
        print(f"  modal run modal_h2_embed.py --variant {variant}")
