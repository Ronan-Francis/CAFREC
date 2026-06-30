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

## Where CAFREC goes

`cafrec/models/cafrec.py` is a minimal valid sequential model today (a single
GRU encoder) so the pipeline runs. Replace the marked block with the real
architecture: SASRec-style short-term encoder, precomputed LLM profile, the
context-adaptive gating MLP over session-context features, and the fusion
`z = g * z_long + (1 - g) * z_short`.

## What's deliberately NOT here

KuaiRand conversion, full-data runs, the 99-negative leave-one-out sampler,
and ILD/Coverage are separate tickets. This harness is the reusable spine they
plug into.
