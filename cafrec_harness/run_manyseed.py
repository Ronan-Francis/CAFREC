"""Stage 4 — many-seed significance for the logfull winner (Pure, parallel spawn).

Runs the 4 core configs at 7 ADDITIONAL seeds (existing 3 -> 10 total) so the
logfull-vs-SASRec win can be tested across seeds, not just 3. Aggregate NDCG per
seed is enough for a paired-across-seeds test; ranks dumped too for per-user
paired tests. bge profile for bge/logfull. tag `ms_s{seed}`.

Run from cafrec_harness/:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python run_manyseed.py
"""
from modal_run import app, train

BGE = "/data/profiles/kuairand_pure_ctx.profiles.hf.d1024.pt"
NEW_SEEDS = (42, 77, 123, 256, 512, 1024, 2048)

CONF = {
    "SASRec":  dict(model_key="SASRec", dataset="kuairand_pure"),
    "noprof":  dict(model_key="CAFREC", dataset="kuairand_pure_ctx",
                    ablation="no_profiler"),
    "bge":     dict(model_key="CAFREC", dataset="kuairand_pure_ctx",
                    llm_profile_path=BGE, profile_dim=1024),
    "logfull": dict(model_key="CAFREC", dataset="kuairand_pure_ctx",
                    llm_profile_path=BGE, profile_dim=1024,
                    history_gate_mode="logfull"),
}

jobs = []
for name, base in CONF.items():
    for s in NEW_SEEDS:
        tag = f"ms_s{s}" if name == "SASRec" else f"{name}_ms_s{s}"
        jobs.append((f"{name}_s{s}", dict(base, seed=s, dump_ranks=True, tag=tag)))


def main():
    with app.run():
        print(f"Spawning {len(jobs)} many-seed legs...", flush=True)
        calls = [(label, train.spawn(**kw)) for label, kw in jobs]
        for label, _ in calls:
            print(f"  spawned {label}", flush=True)
        print("--- waiting for results ---", flush=True)
        done = 0
        for label, c in calls:
            try:
                m = c.get(); done += 1
                print(f"[DONE {done}/{len(calls)}] {label}: ndcg={m['test']['ndcg@10']}",
                      flush=True)
            except Exception as e:
                print(f"[FAIL] {label}: {e!r}", flush=True)
    print("all legs settled", flush=True)


if __name__ == "__main__":
    main()
