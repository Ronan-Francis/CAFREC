# Stage 4 — 10-seed significance: logfull beats SASRec (Pure)

Date: 2026-08-26. The 3 original seeds (2020/2021/403092) + 7 new (42/77/123/256/
512/1024/2048) = 10 seeds, for SASRec, CAFREC-noprof, CAFREC-bge, and CAFREC
history_gate=logfull. mode:full, seq_len 20. Script: `manyseed_sig.py`.

## 1. 10-seed mean NDCG@10

| Model | NDCG@10 | sd | 
|---|---|---|
| SASRec | 0.0397 | 0.0007 |
| CAFREC noprof | 0.0413 | 0.0008 |
| CAFREC bge | 0.0412 | 0.0009 |
| **CAFREC logfull** | **0.0422** | 0.0010 |

## 2. Across-seed paired test vs SASRec (n=10 seed means)

| Model | mean Δ | wins | Wilcoxon p | t-test p |
|---|---|---|---|---|
| noprof | +0.00162 | 9/10 | 7.7e-3 | 8.7e-5 |
| bge | +0.00151 | 10/10 | 2.0e-3 | 2.7e-4 |
| **logfull** | **+0.00254** | **10/10** | **2.0e-3** | **5.1e-5** |

(Wilcoxon p=2.0e-3 is the floor for n=10 — i.e. a perfect 10/10 directional split.)

Per-seed per-user Wilcoxon: **logfull > SASRec significant on 8/10 seeds**
(the two misses, seeds 2020 and 256, are still positive: Δ +0.0011, +0.0016).

## 3. Reading — two honest conclusions

- **logfull decisively beats the strong baseline.** +0.00254 NDCG (+6.4% relative),
  winning ALL 10 seeds (t-test p=5e-5) and significant per-user on 8/10. The 10/10
  directional consistency is the key point: this is a small but ROBUST effect, not a
  significance-fishing artefact. logfull is the best CAFREC configuration and the
  headline positive result of the study.
- **The earlier "SASRec tie" was an under-powered read, now corrected.** With only 3
  seeds, a ~0.0015 NDCG effect against ~0.0008 seed-to-seed sd was not resolvable, so
  bge/noprof looked tied (analysis_pure_3seed). At 10 seeds, plain bge and noprof
  ALSO significantly beat SASRec (10/10 and 9/10 wins, p<0.01). So the honest update
  is: CAFREC's context-adaptive fusion gives a small but consistent and significant
  lift over SASRec across seeds, and the history-gated profile (logfull) enlarges it.
- **Caveats.** Absolute magnitudes remain small (~0.0016-0.0025 NDCG); the result is
  on the Pure tier only; per-user significance is 8/10 not 10/10. Report as a
  robust-but-modest win, with the seed-count correction stated plainly (it is a
  methodological strength, not a weakness — more power, same conclusion direction).

Supersedes the seq_len-20 "tie" framing for the aggregate SASRec comparison; the
seq-length and cohort analyses stand as the mechanism/where-it-helps story.
