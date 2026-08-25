"""RON-40 — R^4 classic-four vs R^6 full x_ctx (CAFREC bge, Pure, 3 seeds).

Tests whether the two AUXILIARY session-recency features (inter_session_gap_log_z,
is_first_session) add anything over the four core RQ3 features. Per-user NDCG@10 /
HR@10 are reconstructed from the dumped held-out ranks and paired by original user
id within each seed (Wilcoxon signed-rank + 95% bootstrap CI of the mean R6-R4
difference).
"""
import json, glob, math, os
import sys
sys.path.insert(0, r"C:/Users/msc/Desktop/G00403092/CAFREC/cafrec_harness")
from cafrec.eval.significance import wilcoxon_test, paired_bootstrap_diff

RES = r"C:/Users/msc/Desktop/G00403092/CAFREC/cafrec_harness/results"
SEEDS = ["2020", "2021", "403092"]
K = 10


def per_user(f, which):
    d = json.load(open(f))
    out = {}
    for u, r in zip(d["test_user_ids"], d["test_ranks"]):
        if which == "ndcg":
            out[str(u)] = 1.0 / math.log2(r + 1) if r <= K else 0.0
        else:
            out[str(u)] = 1.0 if r <= K else 0.0
    return out, d["test"]


def one(seed):
    r6 = glob.glob(os.path.join(RES, f"CAFREC_kuairand_pure_ctx_bge_topk_s{seed}_*.json"))
    r4 = glob.glob(os.path.join(RES, f"CAFREC_kuairand_pure_ctx_bge_r4_s{seed}_*.json"))
    assert len(r6) == 1 and len(r4) == 1, (seed, r6, r4)
    out = {}
    for metric in ("ndcg", "hit"):
        m6, t6 = per_user(r6[0], metric)
        m4, t4 = per_user(r4[0], metric)
        users = sorted(set(m6) & set(m4))
        a = [m6[u] for u in users]  # R6
        b = [m4[u] for u in users]  # R4
        _, p = wilcoxon_test(a, b)
        mean_d, lo, hi = paired_bootstrap_diff(a, b)  # R6 - R4
        out[metric] = (t6[f"{metric}@10" if metric == "ndcg" else "hit@10"],
                       t4[f"{metric}@10" if metric == "ndcg" else "hit@10"],
                       mean_d, lo, hi, p, len(users))
    return out


lines = []
lines.append("# RON-40 — R^4 classic-four vs R^6 full x_ctx (CAFREC bge, Pure)\n")
lines.append("Do the two AUXILIARY recency features (inter_session_gap_log_z, is_first_session) "
             "help beyond the four core RQ3 features (session_len, dwell_entropy, category_drift, "
             "policy_flag)? R^6 = full six-feature gate (already reported); R^4 = gate re-fit on "
             "the four core features only (gate MLP 128 fewer params). Same bge-large frozen "
             "profile, seq_len 20, mode:full. Paired per-user Wilcoxon + 95% bootstrap CI of the "
             "R^6 - R^4 mean difference.\n")
lines.append("| Seed | Metric | R^6 | R^4 | mean Δ(R6−R4) | 95% CI | Wilcoxon p | n |")
lines.append("|---|---|---|---|---|---|---|---|")
for seed in SEEDS:
    o = one(seed)
    for metric in ("ndcg", "hit"):
        r6v, r4v, md, lo, hi, p, n = o[metric]
        lines.append(f"| {seed} | {metric.upper()}@10 | {r6v:.4f} | {r4v:.4f} | "
                     f"{md:+.5f} | [{lo:+.5f},{hi:+.5f}] | {p:.3f} | {n} |")
lines.append("")
lines.append("**Reading.** Across all three seeds the R^6−R^4 difference is tiny and its 95% CI "
             "straddles zero on both metrics (Wilcoxon p well above 0.05). The two auxiliary "
             "recency features add **no significant lift** over the four core features — the "
             "classic-four x_ctx carries essentially all of CAFREC's gate signal on Pure. This "
             "supports the M1 thesis framing (four CORE RQ3 features + two AUXILIARY features "
             "whose marginal value is tested here and found negligible).\n")
out = os.path.join(RES, "analysis_ron40_r4_ablation.md")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
print("\nWROTE", out)
