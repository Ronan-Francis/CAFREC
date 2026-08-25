"""RON-44 — session-intent PROXY stratification (Pure, 3 seeds).

Proxy: category-drift rate (how much a user hops across content categories
within sessions) as an exploratory-vs-focused session-intent signal, taken from
each user's TEST-row prefix feature (causal: summarises history before the
held-out target). Dwell-entropy reported as a secondary proxy.

For each model we reconstruct per-user hit@10 / ndcg@10 from the dumped
per-user held-out ranks, join to the intent stratum by ORIGINAL user id, and
report the mean per stratum, averaged over the three seeds. The RQ3 question:
does CAFREC's context-adaptive gating concentrate its lift on high-drift
(harder, intent-shifting) users rather than spreading it uniformly?
"""
import json, glob, math, os
from collections import defaultdict

RES = r"C:/Users/msc/Desktop/G00403092/CAFREC/cafrec_harness/results"
INTER = r"C:/Users/msc/Desktop/G00403092/CAFREC/data/recbole/kuairand_pure_ctx/kuairand_pure_ctx.inter"

MODELS = {
    "SASRec":        "SASRec_kuairand_pure_topk_s{seed}_*.json",
    "CAFREC_noprof": "CAFREC_kuairand_pure_ctx_noprof_topk_s{seed}_*.json",
    "CAFREC_bge":    "CAFREC_kuairand_pure_ctx_bge_topk_s{seed}_*.json",
}
SEEDS = ["2020", "2021", "403092"]
K = 10


def per_user_metrics(ranks, uids):
    """rank r (1-indexed position of the held-out target) -> hit/ndcg/mrr@K."""
    hit, ndcg, mrr = {}, {}, {}
    for u, r in zip(uids, ranks):
        u = str(u)
        if r is not None and r <= K:
            hit[u] = 1.0
            ndcg[u] = 1.0 / math.log2(r + 1)
            mrr[u] = 1.0 / r
        else:
            hit[u] = ndcg[u] = mrr[u] = 0.0
    return hit, ndcg, mrr


def load_test_row_ctx():
    """Per user (original id) -> test-row (last timestamp) drift + dwell z-scores."""
    best = {}  # uid -> (ts, drift, dwell)
    with open(INTER) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h.split(":")[0]: i for i, h in enumerate(header)}
        iu, it = idx["user_id"], idx["timestamp"]
        idr, idw = idx["prefix_category_drift_z"], idx["prefix_dwell_entropy_z"]
        for line in f:
            p = line.rstrip("\n").split("\t")
            u = p[iu]; ts = float(p[it])
            if u not in best or ts > best[u][0]:
                best[u] = (ts, float(p[idr]), float(p[idw]))
    return {u: (v[1], v[2]) for u, v in best.items()}


def binary_labels(values_by_user, names):
    """Both behavioural context features are ~85-90% concentrated at their floor
    (short/single-category histories on Pure), so a tertile split collapses.
    A binary split at the floor is the honest cut: names[0] = at-floor (the
    focused / low-dispersion mass), names[1] = above-floor (the intent-shifting
    minority — the group RQ3's context gating is meant to help)."""
    floor = min(values_by_user.values())
    lab = {u: (names[1] if v > floor + 1e-6 else names[0])
           for u, v in values_by_user.items()}
    return lab, floor


def main():
    ctx = load_test_row_ctx()
    drift = {u: v[0] for u, v in ctx.items()}
    dwell = {u: v[1] for u, v in ctx.items()}
    drift_lab, dfloor = binary_labels(drift, ["Focused", "Exploratory"])
    dwell_lab, wfloor = binary_labels(dwell, ["LowDisp", "HighDisp"])

    from collections import Counter
    dc = Counter(drift_lab.values())
    wc = Counter(dwell_lab.values())

    # metric[model][stratum] -> list over seeds of mean ndcg (and hit)
    def collect(label_map, order):
        agg_ndcg = {m: defaultdict(list) for m in MODELS}
        agg_hit = {m: defaultdict(list) for m in MODELS}
        overall_ndcg = {m: [] for m in MODELS}
        for m, pat in MODELS.items():
            for seed in SEEDS:
                fs = glob.glob(os.path.join(RES, pat.format(seed=seed)))
                assert len(fs) == 1, (m, seed, fs)
                d = json.load(open(fs[0]))
                hit, ndcg, mrr = per_user_metrics(d["test_ranks"], d["test_user_ids"])
                # verify reconstruction vs reported aggregate (RecBole rounds to 4dp)
                rec_hit = sum(hit.values()) / len(hit)
                assert abs(round(rec_hit, 4) - d["test"]["hit@10"]) < 1e-9, (m, seed, rec_hit, d["test"]["hit@10"])
                overall_ndcg[m].append(sum(ndcg.values()) / len(ndcg))
                by_s_n = defaultdict(list); by_s_h = defaultdict(list)
                for u in ndcg:
                    s = label_map.get(u)
                    if s is None:
                        continue
                    by_s_n[s].append(ndcg[u]); by_s_h[s].append(hit[u])
                for s in order:
                    if by_s_n[s]:
                        agg_ndcg[m][s].append(sum(by_s_n[s]) / len(by_s_n[s]))
                        agg_hit[m][s].append(sum(by_s_h[s]) / len(by_s_h[s]))
        return agg_ndcg, agg_hit, overall_ndcg

    def mean(xs):
        return sum(xs) / len(xs)

    lines = []
    lines.append("# RON-44 — Session-intent proxy stratification (Pure, 3-seed means)\n")
    lines.append("Session-intent is proxied from the **test-row prefix context features** "
                 "(causal: they summarise each user's history *before* the held-out target). "
                 "Per-user HR@10 / NDCG@10 are reconstructed from the dumped held-out ranks "
                 "(reconstruction asserted == RecBole's reported aggregate) and averaged over "
                 "seeds 2020 / 2021 / 403092. Both behavioural features are ~85-90% "
                 "concentrated at their floor on Pure (short, single-category histories), so "
                 "each proxy is a BINARY split at the floor rather than tertiles.\n")

    def emit(proxy_title, label_map, order, counts, note):
        agg_ndcg, agg_hit, overall = collect(label_map, order)
        lines.append(f"## {proxy_title}\n")
        lines.append(note + f" Group sizes: " +
                     ", ".join(f"{s}={counts[s]}" for s in order) + ".\n")
        lines.append("| Model | Overall NDCG | " +
                     " | ".join(f"{s} NDCG" for s in order) + " | " +
                     " | ".join(f"{s} HR" for s in order) + " |")
        lines.append("|---|" + "---|" * (1 + 2 * len(order)))
        for m in MODELS:
            row = [f"{mean(overall[m]):.4f}"]
            row += [f"{mean(agg_ndcg[m][s]):.4f}" for s in order]
            row += [f"{mean(agg_hit[m][s]):.4f}" for s in order]
            lines.append(f"| {m} | " + " | ".join(row) + " |")
        lines.append("")
        lines.append(f"**NDCG@10 Δ vs SASRec** — " +
                     "; ".join(
                         f"{m.replace('CAFREC_','CAFREC ')}: " +
                         ", ".join(f"{s} {mean(agg_ndcg[m][s]) - mean(agg_ndcg['SASRec'][s]):+.4f}"
                                   for s in order)
                         for m in MODELS if m != "SASRec"))
        lines.append("")

    emit("Category-drift proxy (Focused = never switches category; Exploratory = switches)",
         drift_lab, ["Focused", "Exploratory"], dc,
         "Category-drift rate is the RQ3 policy/intent signal: Exploratory users cross "
         "content categories within their session history, the case context-adaptive gating "
         "is meant to help.")
    emit("Dwell-entropy proxy (Low vs High dwell dispersion)",
         dwell_lab, ["LowDisp", "HighDisp"], wc,
         "Dwell-time entropy proxies engagement dispersion across a session.")

    lines.append("## Reading (RQ3 / H3)\n")
    lines.append(
        "- The *Exploratory* (category-switching) and *HighDisp* users are the harder-history "
        "cohort the context gate was hypothesised to help most (H3). They post HIGHER absolute "
        "accuracy for every model (more history to exploit), and on them **SASRec wins**: "
        "CAFREC's NDCG Δ vs SASRec is NEGATIVE for both variants (drift −0.0015/−0.0018; "
        "dwell −0.0007/−0.0008).\n"
        "- CAFREC's small aggregate edge is instead carried by the *Focused* / *LowDisp* "
        "majority (Δ ≈ +0.0011…+0.0015). So on Pure the context-adaptive fusion helps the "
        "focused mass, not the intent-shifting minority — the reverse of the H3 expectation.\n"
        "- Honest RQ3 read: context gating is not a differentiated win for high-intent-shift "
        "users on this tier; the strong short-term attention encoder already captures the "
        "longer exploratory histories. Report as a nuanced/limitation finding, not support "
        "for H3. Magnitudes are small (<0.002 NDCG) and per-seed noise is comparable — pair "
        "with the paired per-user tests before any directional claim.\n")
    out = os.path.join(RES, "analysis_intent_strata.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("\nWROTE", out)


if __name__ == "__main__":
    main()
