"""1K generalisation test — does the logfull win hold on a SECOND dataset?

SASRec vs CAFREC-noprof vs CAFREC-logfull on the k-core-floored 1K
(kuairand_1k_kcore_ctx), identical interactions, matched 5-epoch schedule, seed
2020, ranks dumped. Budget probe (1 seed) on the second tier.

Run from cafrec_harness/:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python run_1k_generalization.py
"""
from modal_run import app, train

DS = "kuairand_1k_kcore_ctx"
PROF = "/data/profiles/kuairand_1k_kcore_ctx.profiles.hf.d1024.pt"
EP = 5

jobs = [
    ("SASRec", dict(model_key="SASRec", dataset=DS, epochs=EP, seed=2020,
                    dump_ranks=True, tag="gen")),
    ("noprof", dict(model_key="CAFREC", dataset=DS, epochs=EP, seed=2020,
                    ablation="no_profiler", dump_ranks=True, tag="noprof_gen")),
    ("logfull", dict(model_key="CAFREC", dataset=DS, epochs=EP, seed=2020,
                     llm_profile_path=PROF, profile_dim=1024,
                     history_gate_mode="logfull", dump_ranks=True,
                     tag="logfull_gen")),
]


def main():
    with app.run():
        print(f"Spawning {len(jobs)} 1K-generalisation legs...", flush=True)
        calls = [(label, train.spawn(**kw)) for label, kw in jobs]
        for label, _ in calls:
            print(f"  spawned {label}", flush=True)
        print("--- waiting for results ---", flush=True)
        for label, c in calls:
            try:
                m = c.get()
                print(f"[DONE] {label}: test={m['test']} valid={m['valid_best']}",
                      flush=True)
            except Exception as e:
                print(f"[FAIL] {label}: {e!r}", flush=True)
    print("all legs settled", flush=True)


if __name__ == "__main__":
    main()
