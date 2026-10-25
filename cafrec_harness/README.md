# CAFREC harness

**Documentation:** [Project README](../README.md) · **Harness README** · [Data README](../data/README.md)

A thin, extensible wrapper around RecBole. The built-in baselines and the CAFREC model run through the **same** entry point. The only thing that changes between them is a registry key.

## Setup

```bash
conda activate recbole    # or activate the .venv described in the project README
pip install -e .          # add [dev] for pytest, [wandb] for Weights & Biases
```

## Run

```bash
python run.py --list                                         # registered keys
python run.py --models SASRec HGN --dataset kuairand_pure    # baselines
python run.py --models CAFREC --dataset kuairand_pure_ctx    # CAFREC
```

`--dataset` defaults to `ml-100k`, the MovieLens-100K example bundled with RecBole. Each run writes `results/<model>_<dataset>_seed<seed>_<ts>.json` and appends a row to `results/summary.csv`, so models can be compared at a glance.

## Smoke test

```bash
pytest -q          # one-epoch ml-100k run per model, asserting that metrics come back
```

## Adding a model

Everything lives in `cafrec/registry.py`, with one line per model.

A **built-in** RecBole model points at its name and picks a training contract (`CE` = no negative sampling, `BPR` = uniform negatives):

```python
"SHAN": ModelSpec(model="SHAN", contract=BPR),
```

A **custom** model points at its class instead of a name. The spec has the same shape:

```python
"CAFREC": ModelSpec(model=CAFREC, contract=CE, config={"hidden_size": 64}),
```

The runner branches on whether `model` is a `str` or a class, and nothing else differs. That one branch is what lets CAFREC run alongside the baselines.

## CAFREC model

`cafrec/models/cafrec.py` holds the full architecture: a SASRec-style short-term encoder (`z_short`), a frozen precomputed long-term profile (`z_long`), a context-adaptive gating MLP over session-context features (`g`), and the fusion `z = g * z_long + (1 - g) * z_short`. Ablations are selected by the `ablation` config key: `none`, `no_profiler`, `static_gate`, `concat`, `np_shuffled_ctx` and `np_vector_gate`. The model's docstring describes each one.

## Context-feature runs

The plain `<dataset>.inter` holds only (user, item, timestamp), which is enough for the baselines. CAFREC's gate also needs causal session-context features, so build a feature-augmented sibling dataset and point CAFREC at it:

```bash
# 1. Build data/recbole/kuairand_pure_ctx/kuairand_pure_ctx.inter
#    (the same rows as the base .inter, plus the context columns)
python -m cafrec.features.build_context_inter --tier light   # or medium / heavy

# 2. Run CAFREC on the _ctx dataset
python run.py --models CAFREC --dataset kuairand_pure_ctx
```

For a dataset whose name ends in `_ctx`, the runner loads the context columns automatically (see `cafrec.features.context.CTX_FIELDS`).

Features are computed **causally**: each row summarises only the session history *before* it, so nothing leaks from the held-out target. `tests/test_features.py` checks this.

## Long-term profile cache

`cafrec/features/profiles.py` defines the frozen `[n_users, profile_dim]` cache that CAFREC loads through `llm_profile_path`, with shape, NaN and sparse-user validation. `cafrec/features/build_profiles.py` builds a real cache from each user's training history. `build_placeholder_profiles(...)` gives a synthetic cache for exercising the profile path in tests.

## Scripts in this folder

| Script | Purpose |
|---|---|
| `run.py` | Train and evaluate any registered model |
| `local_run.py`, `gpu_chain.py` | Queue runs on a local CPU or GPU; results go to `results/local/` and `results/local_gpu/` |
| `modal_run.py`, `modal_*.py` | Cloud runs, profile generation and embedding on Modal |
| `analyze_*.py` | Paired tests and tables for each hypothesis, ablation and control |
| `audit_paper_claims.py` | Recompute the thesis's numbers from the stored runs |
| `demo_recommend.py` | Per-user demo: context, held-out item, ranks and top-10 lists |
| `build_*.py` | Build datasets, prompts and profiles |

Data preparation, including the KuaiRand download and the exploratory figures, is described in the [data README](../data/README.md).
