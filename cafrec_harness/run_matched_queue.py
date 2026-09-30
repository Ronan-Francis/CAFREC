"""Matched-batch completion queue (2026-09-16): every remaining CAFREC-vs-SASRec
claim onto one same-day, same-code-path corpus with train_batch_size explicit.

Follows run_diversity_matched.py (SASRec + CAFREC-NP, 2 x 2 x 7, done). Groups:

  A  sasrec_b512_old    SASRec defaults @512, seeds 2020/2021/403092 (+ topk).
                        Completes SASRec@512 to the 10 seeds of the defaults
                        headline (logfull_ms / hglogfull ran at 512).      3 legs
  B  sasrec_tuned_b512  SASRec TUNED cfg (run_tuned_confirm.py) @512, 10 seeds.
                        Matched partner for tuned_CAFREC_logfull (512). 10 legs
  C  logfull_tuned_b2048  CAFREC logfull TUNED cfg @2048, 10 seeds. The
                        reverse arm, so the batch effect is measured for the
                        tuned pair in both directions.                   10 legs
  D  logfull_b512       CAFREC logfull defaults @512 + topk, 7 ms seeds.
                        Diversity for the headline model.                 7 legs
     logfull_b2048      CAFREC logfull defaults @2048 + topk, all 10 seeds.
                        Batch effect + diversity + 10-seed test at 2048. 10 legs
  E  shuffled_ctx_b512  np_shuffled_ctx @512, seeds 256/512/1024/2048 (+topk),
                        extending the 3-seed gate control to 7 (RQ3).     4 legs
                                                                    total 44

eval_batch_size is pinned to 4096 everywhere (it does not change results).
Ranks are dumped for every leg; top-k for A, D, E. train_pure only. Each leg
is ~8-13 min on A10G; the 09-16 28-leg batch cost ~$5-9, so this is ~$12-15.
Results carry the resolved config (runner.py "config") from this date on.

Run from cafrec_harness/ with the recbole env python and PYTHONUTF8=1:
  python run_matched_queue.py --dry-run
  python run_matched_queue.py --group A         # one group
  python run_matched_queue.py                   # everything pending (resumes)
"""
import argparse
import csv
import time
from pathlib import Path

from modal_run import app, train_pure

OUT = Path(__file__).parent / "results" / "matched_queue.csv"
BGE = "/data/profiles/kuairand_pure_ctx.profiles.hf.d1024.pt"
EVAL_BATCH = 4096

MS_SEEDS = (42, 77, 123, 256, 512, 1024, 2048)
OLD_SEEDS = (2020, 2021, 403092)
ALL_SEEDS = MS_SEEDS + OLD_SEEDS

SASREC = dict(model_key="SASRec", dataset="kuairand_pure")
LOGFULL = dict(model_key="CAFREC", dataset="kuairand_pure_ctx",
               llm_profile_path=BGE, profile_dim=1024, history_gate_mode="logfull")
SHUFFLED = dict(model_key="CAFREC", dataset="kuairand_pure_ctx",
                ablation="np_shuffled_ctx")

# tuned configs exactly as run_tuned_confirm.py selected them (RON-31)
SASREC_TUNED = {"learning_rate": 1e-3, "weight_decay": 1e-5, "hidden_size": 64,
                "inner_size": 256, "hidden_dropout_prob": 0.2,
                "attn_dropout_prob": 0.2, "n_layers": 4, "n_heads": 4}
LOGFULL_TUNED = dict(SASREC_TUNED, n_layers=2)

# group -> list of (label, batch, seed, kwargs)
GROUPS = {
    "A": [("SASRec", 512, s, dict(SASREC, train_batch_size=512, dump_topk=True,
                                 tag="div_b512")) for s in OLD_SEEDS],
    "B": [("SASRec_tuned", 512, s, dict(SASREC, train_batch_size=512,
                                       extra_overrides=SASREC_TUNED,
                                       tag="tuned_SASRec_b512")) for s in ALL_SEEDS],
    "C": [("logfull_tuned", 2048, s, dict(LOGFULL, train_batch_size=2048,
                                         extra_overrides=LOGFULL_TUNED,
                                         tag="tuned_logfull_b2048")) for s in ALL_SEEDS],
    "D": [("logfull", 512, s, dict(LOGFULL, train_batch_size=512, dump_topk=True,
                                  tag="div_b512")) for s in MS_SEEDS]
       + [("logfull", 2048, s, dict(LOGFULL, train_batch_size=2048, dump_topk=True,
                                   tag="div_b2048")) for s in ALL_SEEDS],
    "E": [("shuffled_ctx", 512, s, dict(SHUFFLED, train_batch_size=512, dump_topk=True,
                                       tag="gpu_shuffled_ctx")) for s in (256, 512, 1024, 2048)],
}


def build_jobs(groups):
    jobs = []
    for g in groups:
        for label, bs, s, kw in GROUPS[g]:
            jobs.append((g, label, bs, s, dict(kw, seed=s, eval_batch_size=EVAL_BATCH,
                                              dump_ranks=True)))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--group", default="ABCDE", help="subset of ABCDE")
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    jobs = build_jobs([g for g in args.group.upper() if g in GROUPS])
    if OUT.exists():
        with OUT.open(encoding="utf-8") as fh:
            have = {(r["label"], int(r["batch"]), int(r["seed"]))
                    for r in csv.DictReader(fh)}
        before = len(jobs)
        jobs = [j for j in jobs if (j[1], j[2], j[3]) not in have]
        if before != len(jobs):
            print(f"resume: skipping {before - len(jobs)} leg(s) already in {OUT}")
    if args.n is not None:
        jobs = jobs[:args.n]

    per_group = {g: sum(1 for j in jobs if j[0] == g) for g in GROUPS}
    print(f"legs: {len(jobs)}  by group: {per_group}")
    for g, label, bs, s, _ in jobs[:8]:
        print(f"  [{g}] {label} b{bs} seed {s}")
    if args.dry_run or not jobs:
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT.exists()
    fh = OUT.open("a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(["group", "label", "batch", "seed", "valid_ndcg@10",
                    "test_ndcg@10", "test_hit@10", "test_mrr@10", "n_params",
                    "wall_s", "results_file"])
        fh.flush()

    t0 = time.time()
    with app.run():
        calls = [(g, label, bs, s, time.time(), train_pure.spawn(**kw))
                 for g, label, bs, s, kw in jobs]
        print(f"spawned {len(calls)}; waiting...", flush=True)
        done = fail = 0
        for g, label, bs, s, ts, c in calls:
            try:
                m = c.get()
                done += 1
                cfg = m.get("config") or {}
                w.writerow([g, label, bs, s, m["valid_best"].get("ndcg@10"),
                            m["test"].get("ndcg@10"), m["test"].get("hit@10"),
                            m["test"].get("mrr@10"), m.get("n_params"),
                            round(time.time() - ts), m.get("results_file")])
                fh.flush()
                print(f"[{done + fail}/{len(calls)}] [{g}] {label} b{bs} s{s}: "
                      f"hit={m['test'].get('hit@10')} ndcg={m['test'].get('ndcg@10')}"
                      f"  (json says train_batch_size={cfg.get('train_batch_size')})",
                      flush=True)
            except Exception as e:
                fail += 1
                print(f"[{done + fail}/{len(calls)}] FAIL [{g}] {label} b{bs} s{s}: {e!r}",
                      flush=True)
    fh.close()
    print(f"\n{done} ok / {fail} failed in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
