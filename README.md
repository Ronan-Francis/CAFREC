# CAFREC

**Context-Adaptive Fusion for Long- and Short-Term Video Recommendation**

CAFREC is a sequential recommender that mixes two views of a user:

- a **short-term** SASRec-style encoder over the current interaction sequence (`z_short`)
- a **long-term**, frozen, precomputed user profile, such as an LLM-generated text profile embedded with bge-large (`z_long`)

A small gating MLP reads causal **session-context features** (session length, dwell-time entropy, category drift, policy transitions). It decides how much weight each view gets for every prediction:

```
z = g(x_ctx) * z_long + (1 - g(x_ctx)) * z_short
```

All experiments run on [KuaiRand](https://kuairand.com/) (mainly the Pure tier, with the 1K tier used to check generalisation). Every model is compared against a SASRec baseline trained with the same loss.

> This is thesis research code. Negative and null results are reported as they came out. See **Findings** below.

## Findings (KuaiRand-Pure)

| Question | Result |
|---|---|
| **H1** Does CAFREC beat a loss-matched SASRec? | **No aggregate gain detected.** The data rule out gains larger than about 2% of SASRec's accuracy. This still holds after an identical 16-point tuning budget for both models (7 seeds, no test survives Holm correction). An earlier "win" came from a batch-size confound and disappeared once batch sizes were matched. |
| **H2** Do LLM profiles help the users with the least activity (lowest quartile)? | **Not supported.** Plain LLM profiles vs a learnable stand-in: −0.4% NDCG@10, Holm p = 0.80. They also rank below template profiles and below the no-profile model. |
| **H3** Does a context-adaptive gate beat simpler ways of combining the two views? | **Partly supported.** It beats a static gate and plain concatenation with significance. Shuffling the context features costs 2.1% NDCG@10 overall and 5.0% at session openers, so the gate does read the current session's context. |
| **H4** Does it improve diversity? | **Partly supported.** Coverage improves. ILD improves only in the high-drift stratum. |

Every numeric claim in the paper is recomputed from the stored runs by `audit_paper_claims.py`. The latest audit: 77 reproduce, 0 differ, 2 partial (seeds missing on disk).

## Repository layout

```
Foundation/          T1.1 notebooks: KuaiRand prep, session segmentation, .inter builder
cafrec_harness/      RecBole-based harness: models, features, runners, analysis
  cafrec/            package: registry, runner, models/cafrec.py, features/, eval/
  configs/           base RecBole config
  results/           per-run JSON + analysis outputs (*.txt / *.json)
  tests/             pytest smoke + causal-feature leakage tests
  H2_RUNBOOK.md      how the LLM-profile (H2) experiment was built and run
PROJECT_LOG.md       dated decision and results log for every cycle
data/                local only, not committed (see Data)
```

## Setup

```bash
conda activate recbole          # Python 3.11 recommended
cd cafrec_harness
pip install -e .[dev]           # recbole==1.2.1, torch, numpy<2, scipy
pytest -q                       # 1-epoch smoke run per model
```

## Data

The raw data is not committed. Download **KuaiRand-Pure** from Zenodo, then run `Foundation/CAFREC_T1_1_KuaiRand_Preparation.ipynb` and `CAFREC_Dataset_Builder_KuaiRand_inter.ipynb` to produce the RecBole `.inter` files. Then build the context-feature version of the dataset:

```bash
python -m cafrec.features.build_context_inter --tier light
```

Context features are computed causally: each row summarises only the session history before it. `tests/test_features.py` checks this.

## Running

```bash
python run.py --list                                         # registered models
python run.py --models SASRec HGN --dataset kuairand_pure    # baselines
python run.py --models CAFREC --dataset kuairand_pure_ctx    # CAFREC
```

Each run writes `results/<model>_<dataset>_seed<seed>_<ts>.json` and adds a row to `results/summary.csv`. Ablations (`no_profiler | static_gate | concat`) are chosen with the `ablation` config key. Full-scale runs use Modal (`modal_run.py`). `local_run.py` and `gpu_chain.py` drive local GPU job queues.

See [`cafrec_harness/README.md`](cafrec_harness/README.md) for how to add models, the profile cache format and the context-feature pipeline.

## Author

Ronan Francis. (Atlantic Technological University) MSc in Computing thesis, 2026. Supervisor: Michael Duignan 
