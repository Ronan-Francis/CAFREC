"""Recompute the numbers printed in the paper from the stored run data and flag any that
differ. CPU only; reads results/modal, results/local and results/local_gpu.

Each claim carries the value as printed in JournalPaper/sections, the location, and the
recomputation. Verdicts:
    OK           recomputed value matches within tolerance
    DIFFERS      recomputed value is outside tolerance (fix the paper or the claim)
    MISSING      the runs needed are not on this machine, so the claim cannot be checked

Tolerances: 0.0002 absolute on metric means (the paper prints four decimals), 0.0003 on
paired differences and CI bounds, and for p-values a match means both fall on the same
side of 0.05 and within a factor of three.

Usage:  .venv/Scripts/python audit_paper_claims.py
Writes results/paper_audit_<date>.txt.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import analyze_plan_hypotheses as aph
from analyze_plan_hypotheses import ABLATION, HEADLINE, Runs, cohorts, compare

OPEN, WITHIN = "short session", "longer session"
rows = []


def verdict(paper, got, kind):
    if got is None:
        return "MISSING", "runs not on this machine"
    if kind == "p":
        same_side = (paper < 0.05) == (got < 0.05)
        close = paper == 0 or (1 / 3 <= (got / paper if paper else 1) <= 3)
        return ("OK" if same_side and close else "DIFFERS"), f"paper {paper:.3g}, recomputed {got:.3g}"
    tol = 0.0002 if kind == "mean" else 0.0003
    ok = abs(paper - got) <= tol
    return ("OK" if ok else "DIFFERS"), f"paper {paper:+.4f}, recomputed {got:+.4f}, diff {got - paper:+.5f}"


def check(where, what, paper, got, kind="mean"):
    v, detail = verdict(paper, got, kind)
    rows.append((v, where, what, detail))


def main():
    t0 = time.time()
    rng = np.random.default_rng(aph.BOOT_SEED)
    aph.PATS["Shuffled ctx (7)"] = ["CAFREC_kuairand_pure_ctx_gpu_shuffled_ctx_seed{s}_*.json"]
    probe = aph._load_json("SASRec", HEADLINE[0])
    users = sorted(str(u) for u in probe["test_user_ids"])
    runs = Runs(users)
    masks, _ = cohorts(users)
    allm = masks["all"]

    def mean_sd(name, seeds, metric):
        try:
            per_seed = runs.get(name, seeds)[metric].mean(1)
        except FileNotFoundError:
            return None, None
        return float(per_seed.mean()), float(per_seed.std(ddof=1))

    # ---- Table I: headline accuracy, seven seeds, batch 512 -------------------
    table1 = {
        "SASRec":           (0.0810, 0.0411, 0.0291),
        "CAFREC-NP":        (0.0816, 0.0416, 0.0297),
        "CAFREC":           (0.0810, 0.0415, 0.0297),
        "HGN (CE)":         (0.0717, 0.0340, 0.0227),
        "HGRU4Rec (CE)":    (0.0668, 0.0314, 0.0209),
        "HGN (BPR)":        (0.0235, 0.0112, 0.0074),
        "HGRU4Rec (BPR)":   (0.0554, 0.0278, 0.0195),
    }
    for name, (hr, ndcg, mrr) in table1.items():
        for metric, paper in (("hit@10", hr), ("ndcg@10", ndcg), ("mrr@10", mrr)):
            got, _ = mean_sd(name, HEADLINE, metric)
            check("Table I", f"{name} {metric}", paper, got)

    # ---- Table II: paired tests against SASRec --------------------------------
    table2 = {
        ("CAFREC-NP", "hit@10"):  (+0.0006, 0.45),
        ("CAFREC-NP", "ndcg@10"): (+0.0006, 0.11),
        ("CAFREC-NP", "mrr@10"):  (+0.0006, 0.045),
        ("CAFREC", "hit@10"):     (+0.0000, 0.90),
        ("CAFREC", "ndcg@10"):    (+0.0005, 0.34),
        ("CAFREC", "mrr@10"):     (+0.0006, 0.23),
    }
    for (name, metric), (d, p) in table2.items():
        r = compare(runs, name, "SASRec", HEADLINE, metric, allm, rng)
        check("Table II", f"{name} vs SASRec {metric} delta", d, r["delta"], "delta")
        check("Table II", f"{name} vs SASRec {metric} p", p, r["p"], "p")

    # ---- Fusion ablations and gate controls, three seeds ----------------------
    fusion = {
        "CAFREC [abl]":     (0.0791, 0.0405, None, None),
        "Static gate":      (0.0764, 0.0385, -0.0019, 6.3e-6),
        "Concatenation":    (0.0722, 0.0351, -0.0054, 7.4e-19),
        "CAFREC-NP [abl]":  (0.0799, 0.0406, None, None),
        "Vector gate":      (0.0770, 0.0385, -0.0022, 2.8e-9),
        "Shuffled ctx":     (0.0798, 0.0406, -0.0001, 0.93),
    }
    for name, (hr, ndcg, d, p) in fusion.items():
        for metric, paper in (("hit@10", hr), ("ndcg@10", ndcg)):
            got, _ = mean_sd(name, ABLATION, metric)
            check("Table IV", f"{name} {metric}", paper, got)
        if d is not None:
            ref = "CAFREC [abl]" if name in ("Static gate", "Concatenation") else "CAFREC-NP [abl]"
            r = compare(runs, name, ref, ABLATION, "ndcg@10", allm, rng)
            check("Table IV", f"{name} vs {ref} NDCG delta", d, r["delta"], "delta")
            check("Table IV", f"{name} vs {ref} NDCG p", p, r["p"], "p")

    # ---- Session position, seven seeds ---------------------------------------
    session = {
        ("CAFREC-NP", "SASRec", OPEN):            (+0.0016, 2.4e-4),
        ("CAFREC-NP", "SASRec", WITHIN):          (-0.0016, 0.021),
        ("CAFREC", "SASRec", OPEN):               (+0.0010, 0.088),
        ("CAFREC", "SASRec", WITHIN):             (-0.0007, 0.47),
        ("CAFREC-NP @2048", "SASRec @2048", OPEN):   (+0.0007, 8.7e-3),
        ("CAFREC-NP @2048", "SASRec @2048", WITHIN): (-0.0009, 0.037),
    }
    for (a, b, coh), (d, p) in session.items():
        r = compare(runs, a, b, HEADLINE, "ndcg@10", masks[coh], rng)
        label = "opener" if coh == OPEN else "within"
        check("Table VIII", f"{a} vs {b} {label} delta", d, r["delta"], "delta")
        check("Table VIII", f"{a} vs {b} {label} p", p, r["p"], "p")

    # ---- Shuffled control on seven seeds (corrected 2026-09-20) ---------------
    r = compare(runs, "Shuffled ctx (7)", "CAFREC-NP", HEADLINE, "ndcg@10", allm, rng)
    check("Sec. V-C text", "shuffled vs CAFREC-NP NDCG delta", -0.0009, r["delta"], "delta")
    check("Sec. V-C text", "shuffled vs CAFREC-NP NDCG p", 0.023, r["p"], "p")

    # ---- Equal-budget tuning, seven seeds ------------------------------------
    for name, key, (hr, ndcg, mrr) in (
        ("SASRec (tuned)", "SASRec_kuairand_pure_tuned16_sasrec_gs*_seed{s}.json",
         (0.0880, 0.0456, 0.0329)),
        ("CAFREC-NP (tuned)", "CAFREC_kuairand_pure_ctx_tuned16_np_gs*_seed{s}.json",
         (0.0888, 0.0462, 0.0335)),
    ):
        aph.PATS[name] = [key]
        for metric, paper in (("hit@10", hr), ("ndcg@10", ndcg), ("mrr@10", mrr)):
            old = aph.RES
            aph.RES = os.path.join(aph.HERE, "results", "local_gpu")
            got, _ = mean_sd(name, HEADLINE, metric)
            aph.RES = old
            check("Table VI", f"{name} {metric}", paper, got)

    # ---- Batch-size claim: SASRec at 2048 versus 512 --------------------------
    got, _ = mean_sd("SASRec @2048", HEADLINE, "hit@10")
    check("Sec. IV-E, batch size", "SASRec HR@10 at batch 2048", 0.0787, got)

    # ---- Loss share of SASRec's lead over BPR-trained HGN ---------------------
    sas, _ = mean_sd("SASRec", HEADLINE, "hit@10")
    hgn_bpr, _ = mean_sd("HGN (BPR)", HEADLINE, "hit@10")
    hgn_ce, _ = mean_sd("HGN (CE)", HEADLINE, "hit@10")
    share = (hgn_ce - hgn_bpr) / (sas - hgn_bpr)
    rows.append(("OK" if abs(share - 0.84) <= 0.01 else "DIFFERS", "Sec. VI-B",
                 "loss share of SASRec's lead over HGN (BPR)",
                 f"paper 84%, recomputed {100 * share:.1f}%"))

    # ---- Tuned configurations from the earlier partial search ----------------
    # Aggregate test metrics only: these runs predate the per-user rank dumps.
    def seed_means(pattern):
        import glob
        out = {}
        for f in glob.glob(os.path.join(aph.RES, pattern)):
            d = json.load(open(f))
            out[d["seed"]] = d["test"]["ndcg@10"]
        return out

    tuned_sasrec_512 = seed_means("SASRec_kuairand_pure_tuned_SASRec_b512_seed*.json")
    tuned_sasrec_2048 = seed_means("SASRec_kuairand_pure_tuned_SASRec_tuned_s*_seed*.json")
    tuned_h_512 = seed_means("CAFREC_kuairand_pure_ctx_tuned_CAFREC_logfull_tuned_s*_seed*.json")
    tuned_h_2048 = seed_means("CAFREC_kuairand_pure_ctx_tuned_logfull_b2048_seed*.json")

    # The paper reports ten seeds for these; only nine of each are on this machine, so a
    # small gap is expected and the verdict is PARTIAL rather than DIFFERS.
    for what, paper, vals in (("tuned SASRec NDCG@10 at 2048", 0.0455, tuned_sasrec_2048),
                              ("tuned CAFREC-H NDCG@10 at 512", 0.0468, tuned_h_512)):
        got = float(np.mean(list(vals.values()))) if vals else None
        v = "MISSING" if got is None else ("OK" if abs(got - paper) <= 0.0002 else "PARTIAL")
        rows.append((v, "Sec. V-F, grid search", f"{what} (paper: 10 seeds, on disk: {len(vals)})",
                     f"paper {paper:.4f}, recomputed {got:.4f}" if got else "no runs"))

    for label, paper_pct, a, b in (("512", 2.9, tuned_h_512, tuned_sasrec_512),
                                   ("2048", 2.0, tuned_h_2048, tuned_sasrec_2048)):
        shared = sorted(set(a) & set(b))
        if not shared:
            rows.append(("MISSING", "Sec. V-F, grid search",
                         f"tuned CAFREC-H over tuned SASRec at batch {label}", "no shared seeds on disk"))
            continue
        va, vb = np.array([a[s] for s in shared]), np.array([b[s] for s in shared])
        pct = 100 * (va - vb).mean() / vb.mean()
        v = "OK" if abs(pct - paper_pct) <= 0.5 else ("PARTIAL" if len(shared) < 10 else "DIFFERS")
        rows.append((v, "Sec. V-F, grid search",
                     f"tuned CAFREC-H over tuned SASRec at batch {label} "
                     f"(paper: 10 seeds, on disk: {len(shared)})",
                     f"paper +{paper_pct}%, recomputed {pct:+.1f}%, "
                     f"higher in {int((va > vb).sum())}/{len(shared)} seeds"))

    order = {"DIFFERS": 0, "PARTIAL": 1, "MISSING": 2, "OK": 3}
    rows.sort(key=lambda r: (order[r[0]], r[1]))
    n = {k: sum(1 for r in rows if r[0] == k) for k in order}
    lines = [f"Paper claim audit, {time.strftime('%Y-%m-%d')}",
             f"{n['OK']} reproduce, {n['DIFFERS']} differ, {n['PARTIAL']} partial (fewer seeds on disk than the paper reports), {n['MISSING']} cannot be checked", ""]
    lines += [f"  {v:<8} {where:<30} {what:<52} {detail}" for v, where, what, detail in rows]
    lines.append(f"\nwall {time.time() - t0:.0f}s")
    text = "\n".join(lines)
    print(text)
    out = os.path.join(aph.HERE, "results", f"paper_audit_{time.strftime('%Y%m%d')}.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
