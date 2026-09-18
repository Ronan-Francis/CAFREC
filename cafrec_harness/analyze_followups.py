"""Paired tests for the 2026-09-16 budget follow-ups (modal_run.py basesweep /
gatectl / hashsweep). Same statistics as analyze_batch_confound.py (Section IV-F):
per-user metric averaged over shared seeds, Wilcoxon signed-rank, 95% paired
bootstrap CI over users, per-seed significance and sign counts.

(A) Matched-batch four-model comparison, all at train_batch_size 512, 7 headline seeds.
(B) CAFREC-NP gate controls at the ablation seeds {2020, 2021, 403092}.
(C) Hashing-backend profile vs bge-large profile, ablation seeds.
(D) HGN / HGRU4Rec retrained under SASRec's CE loss (basesweep --loss ce), batch
    512, 7 headline seeds: separates architecture from the BPR-vs-CE loss confound.

No multiplicity correction is applied here; p-values are per comparison.

Usage:  python analyze_followups.py [results_dir] [--only A,B,C,D]
The bootstrap is pure Python and slow (~minutes per comparison); --only skips sections.
"""
from __future__ import annotations

import glob
import json
import os
import sys

from analyze_batch_confound import boot_ci, mean, per_user, sd, wilcoxon

_args = sys.argv[1:]
ONLY = None
if "--only" in _args:
    i = _args.index("--only")
    ONLY = {x.strip().upper() for x in _args[i + 1].split(",")}
    del _args[i:i + 2]
RES = _args[0] if _args else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results")

HEADLINE = [42, 77, 123, 256, 512, 1024, 2048]
ABLATION = [2020, 2021, 403092]
METRICS = ("hit@10", "ndcg@10", "mrr@10")

M = "modal/"
PATS = {
    # --- (A) batch 512, headline seeds
    "SASRec @512":    M + "SASRec_kuairand_pure_bs512_sasrec_seed{s}_*.json",
    "HGN @512":       M + "HGN_kuairand_pure_bs512_hgn_seed{s}_*.json",
    "HGRU4Rec @512":  M + "HGRU4Rec_kuairand_pure_bs512_hgru4rec_seed{s}_*.json",
    "CAFREC-NP @512": M + "CAFREC_kuairand_pure_ctx_noprof_ms_s{s}_seed{s}_*.json",
    "CAFREC @512":    M + "CAFREC_kuairand_pure_ctx_bge_ms_s{s}_seed{s}_*.json",
    # --- (B)/(C) ablation seeds, batch 512
    "CAFREC-NP":      M + "CAFREC_kuairand_pure_ctx_noprof_topk_s{s}_seed{s}_*.json",
    "Vector gate":    M + "CAFREC_kuairand_pure_ctx_vector_gate_s{s}_seed{s}_*.json",
    "Shuffled ctx":   M + "CAFREC_kuairand_pure_ctx_shuffled_ctx_s{s}_seed{s}_*.json",
    "Static gate":    M + "CAFREC_kuairand_pure_ctx_static_gate_s{s}_seed{s}_*.json",
    "CAFREC (bge)":   M + "CAFREC_kuairand_pure_ctx_bge_topk_s{s}_seed{s}_*.json",
    "CAFREC (hash)":  M + "CAFREC_kuairand_pure_ctx_hash_s{s}_seed{s}_*.json",
    # --- (D) CE-loss baselines, batch 512, headline seeds
    "HGN-CE @512":      M + "HGN_kuairand_pure_bs512_ce_hgn_seed{s}_*.json",
    "HGRU4Rec-CE @512": M + "HGRU4Rec_kuairand_pure_bs512_ce_hgru4rec_seed{s}_*.json",
}

SECTIONS = [
    ("(A) Matched batch 512, seven headline seeds", HEADLINE,
     ["SASRec @512", "HGN @512", "HGRU4Rec @512", "CAFREC-NP @512", "CAFREC @512"],
     [("CAFREC-NP @512", "SASRec @512",   "gate effect (re-stated, matched)"),
      ("SASRec @512",    "HGN @512",      "CE transformer vs BPR gated baseline"),
      ("SASRec @512",    "HGRU4Rec @512", "CE transformer vs BPR hierarchical RNN"),
      ("CAFREC-NP @512", "HGN @512",      ""),
      ("CAFREC-NP @512", "HGRU4Rec @512", ""),
      ("CAFREC @512",    "HGN @512",      "")]),
    ("(B) Gate controls, ablation seeds", ABLATION,
     ["CAFREC-NP", "Vector gate", "Shuffled ctx", "Static gate"],
     [("Vector gate",  "CAFREC-NP",    "remove gate input, keep per-dim freedom"),
      ("Shuffled ctx", "CAFREC-NP",    "scramble context across sessions"),
      ("Vector gate",  "Shuffled ctx", "")]),
    ("(C) Profile embedder, ablation seeds", ABLATION,
     ["CAFREC-NP", "CAFREC (bge)", "CAFREC (hash)"],
     [("CAFREC (hash)", "CAFREC (bge)", "does text semantics matter?"),
      ("CAFREC (hash)", "CAFREC-NP",    "hashing profile vs no profile"),
      ("CAFREC (bge)",  "CAFREC-NP",    "bge profile vs no profile (re-stated)")]),
    ("(D) Loss-matched baselines (CE), batch 512, seven headline seeds", HEADLINE,
     ["SASRec @512", "CAFREC-NP @512", "HGN @512", "HGN-CE @512",
      "HGRU4Rec @512", "HGRU4Rec-CE @512"],
     [("HGN-CE @512",      "HGN @512",         "loss effect on HGN (CE vs BPR)"),
      ("HGRU4Rec-CE @512", "HGRU4Rec @512",    "loss effect on HGRU4Rec (CE vs BPR)"),
      ("SASRec @512",      "HGN-CE @512",      "architecture gap, loss matched"),
      ("SASRec @512",      "HGRU4Rec-CE @512", "architecture gap, loss matched"),
      ("CAFREC-NP @512",   "HGN-CE @512",      "loss matched"),
      ("CAFREC-NP @512",   "HGRU4Rec-CE @512", "loss matched")]),
]


def load(pattern, seed):
    hits = sorted(glob.glob(os.path.join(RES, pattern.format(s=seed))))
    if not hits:
        return None
    with open(hits[-1]) as fh:
        return json.load(fh)


def batch_of(d):
    """Resolved train_batch_size if the JSON records it (post-fix runs), else '?'."""
    return (d.get("config") or {}).get("train_batch_size", "?")


def main():
    print(f"results dir : {RES}\n")
    for title, seeds, conds, pairs in SECTIONS:
        if ONLY and title[1] not in ONLY:
            continue
        k = len(seeds)
        print("=" * 90)
        print(f"{title}   seeds={seeds}")
        print("=" * 90)
        loaded = {c: {s: load(PATS[c], s) for s in seeds} for c in conds}
        print(f"{'Condition':<16} {'bs':>5} {'HR@10':>17} {'NDCG@10':>17} {'MRR@10':>17}")
        for c in conds:
            ds = loaded[c]
            miss = [s for s in seeds if ds[s] is None]
            if miss:
                print(f"{c:<16} MISSING seeds {miss}")
                continue
            bs = sorted({str(batch_of(ds[s])) for s in seeds})
            cells = [f"{mean([ds[s]['test'][m] for s in seeds]):.4f} +- "
                     f"{sd([ds[s]['test'][m] for s in seeds]):.4f}" for m in METRICS]
            print(f"{c:<16} {'/'.join(bs):>5} {cells[0]:>17} {cells[1]:>17} {cells[2]:>17}")

        for var, ref, what in pairs:
            dv, dr = loaded[var], loaded[ref]
            if any(v is None for v in dv.values()) or any(v is None for v in dr.values()):
                print(f"\n{var} vs {ref}: MISSING")
                continue
            pv = {s: per_user(dv[s]) for s in seeds}
            pr = {s: per_user(dr[s]) for s in seeds}
            common = sorted(set.intersection(*[set(pv[s]) for s in seeds],
                                             *[set(pr[s]) for s in seeds]))
            label = f"   [{what}]" if what else ""
            print(f"\n{var}  vs  {ref}{label}   n = {len(common):,}")
            for m in METRICS:
                diffs = [mean([pv[s][u][m] for s in seeds]) - mean([pr[s][u][m] for s in seeds])
                         for u in common]
                d = mean(diffs); pval = wilcoxon(diffs); lo, hi = boot_ci(diffs)
                sig = sign = 0
                for s in seeds:
                    ds_ = [pv[s][u][m] - pr[s][u][m] for u in common]
                    if wilcoxon(ds_) < 0.05: sig += 1
                    if (mean(ds_) < 0) == (d < 0): sign += 1
                base = mean([mean([pr[s][u][m] for s in seeds]) for u in common])
                print(f"  {m:<9} delta={d:+.5f} ({100.0 * d / base:+.1f}%)  "
                      f"95% CI [{lo:+.5f}, {hi:+.5f}]  p={pval:.3g}  "
                      f"sig {sig}/{k}  sign {sign}/{k}")
        print()


if __name__ == "__main__":
    main()
