# Diversity (ILD + Coverage) across all Pure conditions

Date: 2026-08-26. Offline join of the dumped top-k lists (stored as original
KuaiRand video_id tokens) to the local multihot video-tag matrix (7,583 items ×
46 tags). No GPU — reuses `cafrec.eval.diversity` + `category_matrix`. Coverage
denominator = real catalogue (n_items_catalog, ~7,210). 3-seed means (2 where
noted). Script: `pure_diversity.py`. **Validation:** the three reference rows
reproduce the reported pure_3seed numbers exactly.

| Condition | ILD@10 | Coverage@10 | seeds |
|---|---|---|---|
| SASRec (seq20, ref) | 0.7930 | 0.1889 | 3 |
| noprof (seq20, ref) | 0.7862 | 0.2336 | 3 |
| bge (seq20, ref) | 0.7920 | 0.2294 | 3 |
| static_gate | 0.7928 | 0.2007 | 3 |
| concat | 0.8005 | 0.1858 | 3 |
| standin | 0.8099 | 0.2026 | 2 |
| prof7b | 0.7891 | 0.2358 | 2 |
| SASRec seq10 | 0.7962 | 0.1872 | 3 |
| bge seq10 | 0.7973 | 0.2252 | 3 |
| noprof seq10 | 0.7952 | 0.2302 | 3 |
| SASRec seq50 | 0.7863 | 0.1852 | 3 |
| bge seq50 | 0.7898 | 0.2395 | 3 |
| noprof seq50 | 0.7817 | 0.2357 | 3 |

## Reading

- **ILD is flat (~0.78–0.81) everywhere** — intra-list category spread does not
  separate the models, consistent with the original finding. Coverage carries the
  diversity story.
- **CAFREC's catalogue-coverage advantage is robust to sequence length.** At every
  window CAFREC (bge/noprof) reaches ~0.23–0.24 vs SASRec's ~0.185–0.189 — a
  stable ~+4–5 pp (~+22% relative). The broader catalogue spread reported at
  seq_len 20 is not an artefact of that operating point; it holds at 10 and 50.
- **The coverage advantage is ARCHITECTURAL — it tracks the gated fusion.** Removing
  the gated fusion (concat) collapses coverage to **0.186 ≈ SASRec's 0.189**, and a
  context-independent gate (static_gate) partially reduces it (0.201), while the
  full gated models keep ~0.23. So the same context-adaptive gated fusion that the
  accuracy ablations validate (analysis_pure_ablations.md) is also what makes
  CAFREC spread recommendations across more of the catalogue. The frozen profile
  itself is not the driver (bge ≈ noprof on coverage; prof7b ≈ bge).
- **Thesis use.** Report coverage as CAFREC's clean, robust secondary win: at
  accuracy statistically tied-or-better vs SASRec, CAFREC recommends ~22% more of
  the catalogue, the effect is stable across sequence lengths, and ablations
  localise it to the gated-fusion mechanism rather than the LLM profile.
