"""Shuffled-context control split by session position, 7 headline seeds (CPU only).

Follow-up to TODO S13. CAFREC-NP beats SASRec when the held-out interaction opens a
session and loses within sessions (results/plan_hypotheses_20260917.txt). The 3-seed
control split in that file was inconclusive. The four missing shuffled-context seeds
(256/512/1024/2048) ran on Modal on 2026-09-16 and were fetched on 2026-09-19, so the
control now covers the same 7 headline seeds as the SASRec and CAFREC-NP runs.

Question: does the opener gain need the RIGHT session's context, or only a gate?
  * If Shuffled ctx matches CAFREC-NP on openers, the gain does not depend on the
    context values (any gate input would do).
  * If Shuffled ctx falls back to SASRec on openers, the gain comes from the context.

Contrasts (per-user metric averaged over seeds; batch 512, Modal A10G, 7 seeds):
  CAFREC-NP    vs SASRec        reference, as reported in the paper
  Shuffled ctx vs SASRec        does a gate fed shuffled context keep the opener gain?
  CAFREC-NP    vs Shuffled ctx  the contribution of the context values themselves
Each is reported for openers (prefix session length 0) and within-session users
(prefix > 0), together with the opener-minus-within interaction.

Batch-size provenance: seeds 256-2048 of the shuffled control record
train_batch_size=512 in their config block. Seeds 42/77/123 predate the config block;
PROJECT_LOG Cycle 10 records that they ran on the same code path as noprof_ms, which
put every _ctx dataset at 512. The script checks the recorded value where one exists.

Usage:  .venv/Scripts/python analyze_shuffled_position.py
Writes results/shuffled_position_<date>.txt and .json.
"""
from __future__ import annotations

import glob
import json
import os
import time

import numpy as np
from scipy.stats import mannwhitneyu, ttest_1samp, wilcoxon as sp_wilcoxon

import analyze_plan_hypotheses as aph
from analyze_plan_hypotheses import HEADLINE, N_BOOT, Runs, cohorts, compare

aph.PATS["Shuffled ctx (7)"] = ["CAFREC_kuairand_pure_ctx_gpu_shuffled_ctx_seed{s}_*.json"]

CONTRASTS = [
    ("CAFREC-NP", "SASRec"),
    ("Shuffled ctx (7)", "SASRec"),
    ("CAFREC-NP", "Shuffled ctx (7)"),
]
OPEN, WITHIN = "short session", "longer session"      # cohort keys in aph.cohorts


def check_batch():
    notes = []
    for s in HEADLINE:
        f = sorted(glob.glob(os.path.join(aph.RES, aph.PATS["Shuffled ctx (7)"][0].format(s=s))))[-1]
        cfg = json.load(open(f)).get("config") or {}
        bs = cfg.get("train_batch_size")
        if bs is not None and bs != 512:
            raise SystemExit(f"shuffled ctx seed {s} ran at batch {bs}, not 512")
        notes.append(f"seed {s}: batch {bs if bs is not None else 'not recorded (512 per PROJECT_LOG)'}")
    return notes


def position_split(runs, a, b, metric, masks, rng):
    op = compare(runs, a, b, HEADLINE, metric, masks[OPEN], rng)
    wi = compare(runs, a, b, HEADLINE, metric, masks[WITHIN], rng)
    do, dw = op.pop("_diffs"), wi.pop("_diffs")
    boot = np.array([do[rng.integers(0, len(do), len(do))].mean()
                     - dw[rng.integers(0, len(dw), len(dw))].mean() for _ in range(N_BOOT)])
    A, B = runs.get(a, HEADLINE)[metric], runs.get(b, HEADLINE)[metric]
    per_seed = ((A - B)[:, masks[OPEN]].mean(1) - (A - B)[:, masks[WITHIN]].mean(1))
    return {
        "a": a, "b": b, "metric": metric, "openers": op, "within": wi,
        "interaction": float(do.mean() - dw.mean()),
        "interaction_ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
        "interaction_p_mannwhitney": float(mannwhitneyu(do, dw).pvalue),
        "interaction_seed_p_W": float(sp_wilcoxon(per_seed, method="exact").pvalue),
        "interaction_seed_p_t": float(ttest_1samp(per_seed, 0.0).pvalue),
        "interaction_seed_sign": int((np.sign(per_seed) == np.sign(per_seed.mean())).sum()),
        "interaction_per_seed": per_seed.tolist(),
    }


def fmt_part(name, r):
    return (f"    {name:<8} d={r['delta']:+.5f} ({r['rel_pct']:+.1f}%) "
            f"CI[{r['ci'][0]:+.5f},{r['ci'][1]:+.5f}] user-level p={r['p']:.3g} "
            f"sign {r['sign']}/7 sig {r['sig']}/7 | seed-level W p={r['seed_p_wilcoxon_exact']:.3g} "
            f"t p={r['seed_p_ttest']:.3g}  (n={r['n_users']:,})")


def main():
    t0 = time.time()
    rng = np.random.default_rng(aph.BOOT_SEED)
    batch_notes = check_batch()
    probe = aph._load_json("SASRec", HEADLINE[0])
    users = sorted(str(u) for u in probe["test_user_ids"])
    runs = Runs(users)
    masks, _ = cohorts(users)

    results = [position_split(runs, a, b, m, masks, rng)
               for a, b in CONTRASTS for m in ("hit@10", "ndcg@10")]

    means = {name: {coh: {m: float(runs.get(name, HEADLINE)[m][:, masks[coh]].mean())
                          for m in ("hit@10", "ndcg@10")}
                    for coh in ("all", OPEN, WITHIN)}
             for name in ("SASRec", "CAFREC-NP", "Shuffled ctx (7)")}

    lines = ["Shuffled-context control by session position, 7 headline seeds "
             f"{HEADLINE}, batch 512, Modal A10G",
             f"openers (prefix 0): {int(masks[OPEN].sum()):,} users; "
             f"within session (prefix > 0): {int(masks[WITHIN].sum()):,} users",
             "Batch check: " + "; ".join(batch_notes), "",
             "== Means (per-user metric averaged over seeds) =="]
    for name, d in means.items():
        lines.append(f"  {name:<17} " + "  ".join(
            f"{coh.replace('short session', 'openers').replace('longer session', 'within')}: "
            f"HR {d[coh]['hit@10']:.4f} NDCG {d[coh]['ndcg@10']:.4f}"
            for coh in ("all", OPEN, WITHIN)))
    lines.append("")
    lines.append("== Contrasts (not Holm-corrected; follow-up analysis) ==")
    for r in results:
        lines.append(f"  {r['a']} vs {r['b']}  [{r['metric']}]")
        lines.append(fmt_part("openers", r["openers"]))
        lines.append(fmt_part("within", r["within"]))
        lines.append(f"    opener-minus-within {r['interaction']:+.5f} "
                     f"CI[{r['interaction_ci'][0]:+.5f},{r['interaction_ci'][1]:+.5f}] "
                     f"MW p={r['interaction_p_mannwhitney']:.3g} | seed-level W p={r['interaction_seed_p_W']:.3g} "
                     f"t p={r['interaction_seed_p_t']:.3g} (same sign {r['interaction_seed_sign']}/7)")
    lines.append(f"\nwall {time.time() - t0:.0f}s")
    text = "\n".join(lines)
    print(text)

    stamp = time.strftime("%Y%m%d")
    out = os.path.join(aph.HERE, "results", f"shuffled_position_{stamp}")
    with open(out + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"seeds": HEADLINE, "batch_check": batch_notes, "means": means,
                   "contrasts": results}, fh, indent=2)
    print(f"-> {out}.txt / .json")


if __name__ == "__main__":
    main()
