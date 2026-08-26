"""History-gated profiler (RON-45) — the H2 fix, on Pure (parallel spawn).

CAFREC with the frozen bge profile AND history_gate=True: the gate additionally
sees the user's history length, so it can suppress z_long when the history is too
thin to profile. Motivated by analysis_h2_sparse.md (profiler helps dense, hurts
sparse). 3 seeds, ranks + topk dumps -> aggregate significance, H2 cohort re-test,
and diversity all reuse the existing scripts.

Run from cafrec_harness/:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python run_history_gate.py
"""
from modal_run import app, train

BGE = "/data/profiles/kuairand_pure_ctx.profiles.hf.d1024.pt"

jobs = [(f"histgate_s{s}", dict(
    model_key="CAFREC", dataset="kuairand_pure_ctx", seed=s,
    llm_profile_path=BGE, profile_dim=1024, history_gate=True,
    dump_ranks=True, dump_topk=True, tag=f"histgate_s{s}"))
    for s in (2020, 2021, 403092)]


def main():
    with app.run():
        print(f"Spawning {len(jobs)} history-gate legs...", flush=True)
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
