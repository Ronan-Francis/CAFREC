"""Stage 1 — history-gate variant bake-off (Pure, parallel spawn).

seqlen (saturating, already run as `histgate`) vs the two unsaturated variants:
  logfull   append log1p(total user activity)/norm to the gate inputs
  suppress  explicit z_long *= sigmoid((h_log - tau)/beta), tau/beta learnt
CAFREC-bge, 3 seeds, ranks + topk dumps -> reuse histgate_analysis for the winner.

Run from cafrec_harness/:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python run_histgate_variants.py
"""
from modal_run import app, train

BGE = "/data/profiles/kuairand_pure_ctx.profiles.hf.d1024.pt"

jobs = []
for mode in ("logfull", "suppress"):
    for s in (2020, 2021, 403092):
        jobs.append((f"hg{mode}_s{s}", dict(
            model_key="CAFREC", dataset="kuairand_pure_ctx", seed=s,
            llm_profile_path=BGE, profile_dim=1024, history_gate_mode=mode,
            dump_ranks=True, dump_topk=True, tag=f"hg{mode}_s{s}")))


def main():
    with app.run():
        print(f"Spawning {len(jobs)} variant legs...", flush=True)
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
