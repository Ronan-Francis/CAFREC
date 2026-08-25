"""Sequence-length sensitivity sweep on Pure (parallel, detached-safe via spawn).

Re-tests the SASRec-vs-CAFREC balance at short (10) and long (50) short-term
windows vs the reported seq_len 20, and probes whether the H2 history-dependence
(profiler helps dense, hurts sparse) shifts as the short-term window changes.

18 legs = {SASRec, CAFREC_bge, CAFREC_noprof} x seq_len{10,50} x seeds{2020,2021,403092}.
Default 10-epoch schedule, ranks + topk dumps. Run from cafrec_harness/:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python run_seqlen_sweep.py
"""
from modal_run import app, train

BGE = "/data/profiles/kuairand_pure_ctx.profiles.hf.d1024.pt"
SEEDS = (2020, 2021, 403092)
LENS = (10, 50)

jobs = []
for L in LENS:
    for s in SEEDS:
        jobs.append((f"SASRec_seq{L}_s{s}", dict(
            model_key="SASRec", dataset="kuairand_pure", seed=s, max_seq_len=L,
            dump_ranks=True, dump_topk=True, tag=f"seq{L}_s{s}")))
        jobs.append((f"bge_seq{L}_s{s}", dict(
            model_key="CAFREC", dataset="kuairand_pure_ctx", seed=s, max_seq_len=L,
            llm_profile_path=BGE, profile_dim=1024,
            dump_ranks=True, dump_topk=True, tag=f"bge_seq{L}_s{s}")))
        jobs.append((f"noprof_seq{L}_s{s}", dict(
            model_key="CAFREC", dataset="kuairand_pure_ctx", seed=s, max_seq_len=L,
            ablation="no_profiler",
            dump_ranks=True, dump_topk=True, tag=f"noprof_seq{L}_s{s}")))


def main():
    with app.run():
        print(f"Spawning {len(jobs)} seq-length legs...", flush=True)
        calls = [(label, train.spawn(**kw)) for label, kw in jobs]
        for label, _ in calls:
            print(f"  spawned {label}", flush=True)
        print("--- waiting for results ---", flush=True)
        for label, c in calls:
            try:
                m = c.get()
                print(f"[DONE] {label}: test={m['test']} -> {m['results_file']}",
                      flush=True)
            except Exception as e:
                print(f"[FAIL] {label}: {e!r}", flush=True)
    print("all legs settled", flush=True)


if __name__ == "__main__":
    main()
