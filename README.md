# CAFREC

**Context-Adaptive Fusion for Long- and Short-Term Video Recommendation**

CAFREC is a sequential recommender that combines a short-term self-attentive encoder with a frozen long-term user profile. A gate driven by session context sets the weight of each. This repository contains the model, the RecBole-based experiment harness, the analysis scripts, and the stored results behind the accompanying MSc thesis paper (Atlantic Technological University, 2026).

## Overview

- **Short-term view** (`z_short`): a SASRec-style self-attentive encoder over the last 20 interactions.
- **Long-term view** (`z_long`): a frozen, precomputed user profile, for example a templated behavioural summary embedded with `bge-large-en-v1.5`.
- **Gate**: a small MLP over causal session-context features (session length so far, dwell-time entropy, category drift, policy transitions, inter-session gap, first-session flag).

```
z = g(x_ctx) * z_long + (1 - g(x_ctx)) * z_short
```

Setting `z_long = 0` gives **CAFREC-NP** (context-gated SASRec). As `g -> 0` the model reduces to SASRec. All experiments use [KuaiRand-Pure](https://kuairand.com/) with per-user leave-one-out splitting and full-catalogue ranking. Every model is compared against a SASRec trained with the same loss, batch size (512) and schedule.

## Results

Summary of the paper's hypothesis outcomes (paper Table XIII). Negative and null results are reported as observed.

| Hypothesis | Outcome | Evidence |
|---|---|---|
| **H1** CAFREC outperforms SASRec | Not supported | Neither CAFREC nor CAFREC-NP differs significantly from SASRec after Holm correction. A gain of up to about 2% of SASRec's HR@10 remains compatible with the data. An earlier +3.7% advantage came from a batch-size mismatch (SASRec at 2048, CAFREC at 512). |
| **H2** LLM profiles help the lowest-activity quartile | Inconclusive | Generated profiles: -0.4% NDCG@10 against a trainable stand-in, -12.6% against CAFREC-NP. The training-target leak was not measured. |
| **H3** Context gate beats static fusion | Partly supported; mechanism not established | Beats concatenation in every stratum. Beats the static scalar gate only overall and in the low-drift stratum. |
| **H4** Context gate improves diversity | Partly supported | Higher coverage than both static alternatives in every seed, by margins about the size of the seed-to-seed variation. Higher ILD on high-drift sessions only. |

## Components

RecBole 1.2.1 provides the data loader, trainer, SASRec and HGN. The components below were written for this project.

| Area | Files |
|---|---|
| Model | `cafrec/models/cafrec.py`: CAFREC and its ablations (`no_profiler`, `static_gate`, `concat`, vector gate, shuffled context, history gate) as config switches on one class. `cafrec/models/hgru4rec.py`: U-GRU, a user-seeded GRU that approximates HGRU4Rec. |
| Harness | `cafrec/registry.py` (one spec per model), `cafrec/runner.py`, `cafrec/tiers.py` (batch size by catalogue tier) |
| Data and features | `Foundation/` notebooks (KuaiRand preparation, 30-minute session cutting, `.inter` builder). `cafrec/features/context.py` and `build_context_inter.py`: causal prefix features, scaler fitted on training rows only. `cafrec/data/leave_one_out.py`. |
| Profiles | `cafrec/features/profiles.py` and `build_profiles.py` (profile cache with shape, NaN and row checks). `build_content_profiles.py`. `build_h2_prompts.py`, `modal_h2_generate.py` and `modal_h2_embed.py` (Qwen2.5-7B generated profiles; see `H2_RUNBOOK.md`). |
| Evaluation | `cafrec/eval/full_rank.py` (per-user rank and top-10 dumps keyed by original user ID). `metrics.py`, `diversity.py` (coverage, ILD), `significance.py` (paired bootstrap, Wilcoxon, Holm, per-seed tests). |
| Launchers | `modal_run.py` (Modal A10G), `local_run.py` and `gpu_chain.py` (local CPU/GPU queues), `run_*.py` sweeps |
| Analysis and provenance | `analyze_*.py`, `audit_paper_claims.py`, `extract_best_epochs.py`, `results/MANIFEST.csv` |

## Installation

```bash
cd cafrec_harness
python -m venv .venv                 # Python 3.11 used; the Modal image uses torch 2.8
.venv/Scripts/activate               # Windows (source .venv/bin/activate on Linux/macOS)
pip install -e .[dev]                # recbole==1.2.1, torch, numpy<2, scipy, pytest
```

- **Local GPU runs:** `gpu_chain.py` expects a CUDA-enabled environment at `cafrec_harness/.venv-gpu`.
- **Cloud runs:** `pip install modal`, run `modal setup`, then upload the `.inter` files to the `cafrec-data` volume (commands at the top of `modal_run.py`).

## Data

`data/` is git-ignored apart from `ctx_scaler_params.json`.

1. Download **KuaiRand-Pure** from Zenodo. Run `Foundation/CAFREC_T1_1_KuaiRand_Preparation.ipynb`, then `Foundation/CAFREC_Dataset_Builder_KuaiRand_inter.ipynb`. This writes `data/recbole/kuairand_pure/kuairand_pure.inter`.
2. Build the context-feature dataset:
   ```bash
   python -m cafrec.features.build_context_inter --tier light
   ```
3. Category and caption files (used by content profiles, ILD and H2): Zenodo record [18159199](https://zenodo.org/records/18159199). Provenance and MD5s are in `data/content_pure/PROVENANCE.json`.
4. Profile caches (`data/profiles/*.pt`) are not committed and must be rebuilt with the profile builders.

## Usage

```bash
python run.py --list                                         # registered models
python run.py --models SASRec HGN --dataset kuairand_pure    # baselines
python run.py --models CAFREC --dataset kuairand_pure_ctx    # CAFREC
```

Each run writes `results/<model>_<dataset>_seed<seed>_<ts>.json` and appends a row to `results/summary.csv`. The `ablation` config key selects an ablation. See [`cafrec_harness/README.md`](cafrec_harness/README.md) for adding models and the profile cache format.

## Tests

```bash
cd cafrec_harness
python -m pytest -q                                   # 70 tests in 9 files
python -m pytest -q --ignore=tests/test_smoke.py      # skip the 1-epoch training runs
```

`test_smoke.py` trains each model for one epoch on RecBole's `ml-100k`.

| Coverage | Tests |
|---|---|
| Split integrity | 8 |
| Prefix causality | 19 |
| Scaler fitting | 1 |
| Profile row alignment | 9 |
| Ranks and metrics against hand-computed values | 4 |
| One-epoch run per model | 4 |
| Batch-size defaults | 12 |
| U-GRU losses | 3 |
| Statistics module | 4 |
| Diversity metrics | 6 |

## Reproducing Tables V and VI

- Table V (`tab:res:main`): top-10 accuracy, mean ± SD over seven seeds.
- Table VI (`tab:res:tests`): paired per-user tests against SASRec.

`results/MANIFEST.csv` lists the result files, commit, launch script, batch size, seeds and device for each row. Match rows on the LaTeX label, because the manifest's `paper_table` numbers come from an earlier draft.

### From stored runs (CPU, under a minute)

The result JSONs for both tables, including per-user rank dumps, are committed under `results/modal/` and `results/local_gpu/`.

```bash
cd cafrec_harness
python audit_paper_claims.py      # recomputes every paper number -> results/paper_audit_<date>.txt
python analyze_paper_512.py       # Table V default rows; Table VI deltas, CIs, p, Holm, Sig., Sign
python analyze_tuned_gpu.py       # Table V tuned rows and the equal-budget search
```

The scripts need `kuairand_pure.inter` to build the cohorts. The table means use only the rank dumps.

- **Audit labels:** the audit output uses an older numbering. Table V's default rows appear as `Table I`, its tuned rows as `Table VI`, and Table VI's tests as `Table II`. Latest run: `results/paper_audit_20261003.txt` (77 reproduce, 0 differ, 2 partial).
- **Holm adjustment:** `analyze_paper_512.py` applies Holm over nine tests, three of them CAFREC-H. For Table VI's six tests the adjusted values equal a six-test Holm, because the CAFREC-H p-values are the smallest. Reference output: `results/paper_512_20260916.txt`.

### From scratch

Check out the commit listed for the row. Batch-size handling changed after these runs, so HEAD does not reproduce their configuration.

| Table V rows | Commit | Command | Device |
|---|---|---|---|
| SASRec (CE) | `6c5482b` | `modal run modal_run.py::bsweep` | Modal A10G |
| CAFREC-NP, CAFREC (CE) | `ae63ec4` | `python run_manyseed.py` | Modal A10G |
| HGN, U-GRU (BPR) | `b54452d` | `modal run modal_run.py::basesweep` | Modal A10G |
| HGN, U-GRU (CE) | `b54452d` | `modal run modal_run.py::basesweep --loss ce` | Modal A10G |
| SASRec, CAFREC-NP tuned | `4841de5` | `python gpu_chain.py --workers 3 --threads 4` | RTX 2060 SUPER |

- **Seeds:** 42, 77, 123, 256, 512, 1024, 2048. The tuned search selects at seed 403092.
- **`run_manyseed.py`** also trains SASRec at batch size 2048 and CAFREC-H.
- **`gpu_chain.py`** runs `local_run.py --queue gpu_grid`, then `--queue gpu_tuned`.

Retrained runs are expected to differ slightly from the stored ones (see [Reproducibility limitations](#reproducibility-limitations)).

## Configuration

| Setting | Location |
|---|---|
| Shared settings (leave-one-out, full ranking, history length 20, 10 epochs, patience 3, NDCG@10 validation) | `cafrec_harness/configs/base.yaml` |
| Per-model specs and loss contracts | `cafrec_harness/cafrec/registry.py` |
| Batch size by catalogue tier | `cafrec_harness/cafrec/tiers.py` |
| Context-feature fields | `cafrec/features/context.py` (`CTX_FIELDS`) |
| Sweep definitions | `modal_run.py` entry points (`bsweep`, `basesweep`, `gatectl`, `hashsweep`, `h2`, `main`), `local_run.py` queues, `run_manyseed.py`, `run_parallel_pure.py` |
| Resolved per-run config | inside each result JSON (runs after the batch-size fix only) |
| Commit, script, batch size, seeds and device per paper value | `cafrec_harness/results/MANIFEST.csv` |

## Analysis status

Analyses are classified by when they were specified, relative to the project plan submitted in May 2026, before any experiment. The plan was not publicly registered. The full classification is paper Table III.

| Status | Analyses |
|---|---|
| **Pre-specified** | CAFREC-NP and CAFREC vs SASRec. CAFREC vs HGN and U-GRU at native loss. Profile removal. Sparse and dense cohorts. Static gate, concatenation and four-feature ablations. Policy-flag ablation. H3 within activity and drift strata (strata fixed after the aggregate results). |
| **Added after first results** | H4 (August 2026 plan revision). HGN and U-GRU under cross-entropy. Equal-budget tuning. Trainable stand-in, 7B-encoder and hashing profiles. Content-bearing profiles. LLM-generated profile for H2 (design fixed before generation). Vector gate and shuffled context. |
| **Exploratory** | Session-position analyses, including H1's short-session clause. Shuffled context by position, and the pooled ten-seed estimate. Minimum detectable effect. Seed-level tests of the batch-size mismatch. Constant-profile control. Gate values. Within-position permutation. Timeline-leak query. CAFREC-H and the partial grid search (both selected on the test split). |

## Compute requirements

| Job | Hardware | Time / cost |
|---|---|---|
| Recomputing paper numbers | any CPU | about 15 s |
| Headline, baseline, ablation and H2 runs | NVIDIA A10G (24 GB) via Modal, in parallel | a 14-job sweep took 314 to 545 s wall clock; $0.42 to $0.61 per job |
| Content-bearing profile runs | 4-core Intel Xeon 8272CL VM | about 41 min per 10-epoch run |
| Equal-budget search and retraining (53 runs) | one RTX 2060 SUPER, 3 to 5 concurrent | about 31 min per run; 7 h 13 min wall clock |
| Generated profiles (Qwen2.5-7B-Instruct, greedy, 22,912 users) | Modal GPU | not recorded |

## Reproducibility limitations

- **Determinism is partial.** RecBole's reproducibility switch is set, but `torch.use_deterministic_algorithms` was never called. Results also vary by device: CPU vs GPU moved HR@10 by 1.9% for one configuration, and two GPU models differed by 4% in NDCG@10.
- **Best validation epochs of the A10G runs are unrecoverable.** Modal kept only the result JSONs. Local runs are covered by `extract_best_epochs.py`.
- **Two audited claims are partial** because seeds are missing on disk: tuned SASRec at batch 2048 (9 of 10 seeds) and tuned CAFREC-H over tuned SASRec (8 of 10).
- **Some launch commands were not logged.** Manifest rows marked "command not logged" were started from the `modal_run.py::main` CLI, and their settings are reconstructed from code.
- **Batch sizes of the reported runs come from versioned launch code.** These runs predate per-run config logging.
- **The MostPop coverage row comes from an untracked script:** `results/todo_runs/MostPop/mostpop_true_count.py`.
- **Data and profile caches are not distributed.** Rebuilding the generated profiles requires a Modal account. The corpus is deterministic given the prompts and the model revision.
- **Not run:** context-feature ranking by retraining, KuaiRand-1K results (that tier needs a k-core floor), content-bearing profiles at seven seeds, and measurement of the training-target leak.

## Repository layout

```
Foundation/          KuaiRand preparation, session segmentation, .inter builder (notebooks)
cafrec_harness/      RecBole-based harness
  cafrec/            package: registry, runner, tiers, models/, features/, data/, eval/
  configs/base.yaml  shared RecBole config
  results/           per-run JSON (modal/, local/, local_gpu/), analysis outputs, MANIFEST.csv
  tests/             pytest suite
  H2_RUNBOOK.md      construction and runs of the LLM-profile (H2) experiment
PROJECT_LOG.md       dated decision and results log
data/                local only, not committed
```

## Author

Ronan Francis MSc in Computing, Atlantic Technological University, 2026. *Supervisor: Michael Duignan*
