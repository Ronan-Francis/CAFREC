# CAFREC harness

A thin, expandable wrapper around RecBole. Built-in baselines and your own
CAFREC model run through the **same** entry point — the only variable that
changes is a registry key.

## Setup

```bash
conda activate recbole
pip install -e .          # add [dev] for pytest, [wandb] for W&B
```

## Run

```bash
python run.py --models SASRec HGN --dataset ml-100k   # baselines
python run.py --models CAFREC --dataset ml-100k       # your model
python run.py --list                                  # see registered keys
```

Each run writes `results/<model>_<dataset>_seed<seed>_<ts>.json` and appends a
row to `results/summary.csv`, so models are diffable at a glance.

## Smoke test

```bash
pytest -q          # 1-epoch ml-100k run per model, asserts metrics come back
```

## Adding a model

Everything lives in `cafrec/registry.py`. One line each.

A **built-in** RecBole model — point at its name and pick the training contract
(`CE` = no negative sampling, `BPR` = uniform negatives):

```python
"SHAN": ModelSpec(model="SHAN", contract=BPR),
```

**Your own** model — point at the class instead of a name. Same spec shape:

```python
"CAFREC": ModelSpec(model=CAFREC, contract=CE, config={"hidden_size": 64}),
```

The runner branches on whether `model` is a `str` or a class; nothing else
differs. That branch is the whole trick that lets CAFREC sit next to the
baselines.

## CAFREC model

`cafrec/models/cafrec.py` is the full architecture: a SASRec-style short-term
encoder (`z_short`), a frozen precomputed LLM profile (`z_long`), a
context-adaptive gating MLP over session-context features (`g`), and the fusion
`z = g * z_long + (1 - g) * z_short`. Ablations (`none | no_profiler |
static_gate | concat`) are selected by the `ablation` config key.

## Context-feature runs (RON-16/17/18)

The plain `<dataset>.inter` has only (user, item, timestamp) — enough for the
baselines. CAFREC's gate additionally needs causal session-context features, so
build a feature-augmented sibling dataset and point CAFREC at it:

```bash
# 1. Build data/recbole/kuairand_pure_ctx/kuairand_pure_ctx.inter
#    (same rows as the base .inter + session_len/dwell_entropy/cat_drift columns)
python -m cafrec.features.build_context_inter --tier light   # or medium/heavy

# 2. Run CAFREC on the _ctx dataset with the extra columns loaded:
python run.py --models CAFREC --dataset kuairand_pure_ctx \
  # (pass load_col + context_fields overrides; see cafrec.features.context.CTX_FIELDS)
```

Features are computed CAUSALLY (each row summarises only its session history
*before* it), so nothing leaks from the held-out target — see
`tests/test_features.py`.

## LLM profile cache (RON-25)

`cafrec/features/profiles.py` defines the frozen `[n_users, profile_dim]` cache
CAFREC loads via `llm_profile_path`, with shape/NaN/sparse-user validation.
`build_placeholder_profiles(...)` gives a synthetic cache so the profile path can
be exercised before the real profiler (RON-24) exists.

## What's deliberately NOT here

Full-data cloud runs (Modal, see `modal_run.py`) and the reasoning-model /
profiler *inference* pipelines are separate, credential/compute-gated tickets.
