# Sequence-length sensitivity — the SASRec tie is specific to seq_len 20 (Pure)

Date: 2026-08-25. SASRec vs CAFREC (bge / noprof) at short-term window seq_len ∈
{10, 20, 50}, 3 seeds, mode:full. seq_len 20 reuses the reported topk dumps; 10
and 50 are the new sweep (run_seqlen_sweep.py, `--max-seq-len`). Per-user
NDCG@10/HIT@10 reconstructed from dumped ranks (== RecBole aggregate); paired
per-user Wilcoxon of (CAFREC − SASRec) per seed. Script: `pure_seqlen_sig.py`.

## 1. 3-seed means

| seq_len | SASRec NDCG | bge NDCG | noprof NDCG | SASRec HR | bge HR | noprof HR |
|---|---|---|---|---|---|---|
| **10** | 0.0388 | 0.0408 | 0.0406 | 0.0773 | 0.0792 | 0.0797 |
| 20 | 0.0396 | 0.0405 | 0.0406 | 0.0787 | 0.0791 | 0.0799 |
| **50** | 0.0396 | 0.0406 | **0.0414** | 0.0791 | 0.0799 | **0.0819** |

## 2. CAFREC − SASRec (paired per-user Wilcoxon, per seed)

| seq_len | model | mean ΔNDCG | p (2020 / 2021 / 403092) | verdict |
|---|---|---|---|---|
| 10 | bge | +0.00198 | 1.8e-5 / 0.060 / 0.062 | sig 1/3, +ve all |
| 10 | **noprof** | +0.00175 | 1.0e-3 / 7.6e-3 / 3.0e-2 | **sig 3/3** |
| 20 | bge | +0.00087 | 0.48 / 0.18 / 0.36 | ns (the known tie) |
| 20 | noprof | +0.00103 | 0.063 / 0.056 / 0.90 | ns |
| 50 | bge | +0.00101 | 0.55 / 3.6e-4 / 0.66 | sig 1/3 |
| 50 | **noprof** | +0.00176 | 4.4e-2 / 9.1e-4 / 3.8e-2 | **sig 3/3** |

## 3. Reading

- **The "CAFREC ties SASRec" headline is an artefact of seq_len 20.** At both a
  shorter (10) and a longer (50) window, CAFREC-**noprof** beats the strong
  short-term baseline with paired significance on ALL three seeds (ΔNDCG ≈ +0.002,
  ~+5% relative). seq_len 20 is the one window where the gap is a non-significant
  tie. This is the clearest CAFREC-over-SASRec evidence in the project: the
  context-gated architecture has a real, if modest, edge across the sequence-length
  regime, obscured at the single default operating point.
- **Short window (10): the gate compensates for a starved encoder.** When only 10
  recent items feed the short-term encoder, SASRec drops (NDCG 0.0396→0.0388) while
  CAFREC holds (≈0.0407). The context gate supplies structure the truncated
  sequence no longer carries.
- **Long window (50): profiler-off wins outright.** noprof reaches the project's
  best Pure accuracy (NDCG 0.0414, HR 0.0819) and significantly beats SASRec on all
  seeds. bge (profiler on) does NOT — it trails noprof — echoing the H2 / ablation
  result that the frozen profile adds noise more than signal here.
- **It is the CONTEXT GATING, not the LLM profiler, that drives the edge.** At every
  window bge ≈ or < noprof. So the sequence-length advantage belongs to CAFREC's
  context-adaptive gating (RQ3 mechanism), consistent with the static_gate ablation;
  the profiler's separate, history-dependent effect is characterised in
  analysis_h2_sparse.md.
- **Honest caveats.** Magnitudes stay small (~0.002 NDCG); the default seq_len 20
  remains a genuine tie; bge's per-seed significance is noisy. Report as: "CAFREC's
  context gating yields a small but significant improvement over SASRec at
  non-default sequence lengths (10 and 50); at the tuned seq_len 20 the two are
  statistically tied, and the LLM profiler does not contribute to this edge."
