"""One-off driver: launch the missing Pure experiments in parallel on Modal.

Runs alongside the 1K diagnostic. Uses the committed modal_run.app/train
Function via .spawn() so all legs run concurrently (each its own A10G), with the
DEFAULT base.yaml schedule (epochs 10 / stopping_step 3) so results are directly
comparable to the reported CAFREC bge runs.

Legs (10):
  static_gate ablation x {2020,2021,403092}  (bge profile; isolates RQ3 gating)
  concat      ablation x {2020,2021,403092}  (bge profile; gated-vs-concat fusion)
  prof_7b               x {2021,403092}       (7B cache; complete capacity ladder)
  standin               x {2021,403092}       (learnable z_long; RQ2 robustness)

Run from cafrec_harness/ (so add_local_python_source/add_local_dir resolve):
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python run_parallel_pure.py
"""
from modal_run import app, train

DS = "kuairand_pure_ctx"
BGE = "/data/profiles/kuairand_pure_ctx.profiles.hf.d1024.pt"
SEVENB = "/data/profiles/kuairand_pure_ctx.profiles.hf.d4096.pt"

jobs = []  # (label, kwargs)
for s in (2020, 2021, 403092):
    for abl in ("static_gate", "concat"):
        jobs.append((f"{abl}_s{s}", dict(
            model_key="CAFREC", dataset=DS, seed=s,
            llm_profile_path=BGE, profile_dim=1024, ablation=abl,
            dump_ranks=True, dump_topk=True, tag=f"{abl}_s{s}")))
for s in (2021, 403092):
    jobs.append((f"prof7b_s{s}", dict(
        model_key="CAFREC", dataset=DS, seed=s,
        llm_profile_path=SEVENB, profile_dim=4096,
        dump_ranks=True, dump_topk=True, tag=f"prof7b_s{s}")))
    jobs.append((f"standin_s{s}", dict(
        model_key="CAFREC", dataset=DS, seed=s,
        dump_ranks=True, dump_topk=True, tag=f"standin_s{s}")))


def main():
    with app.run():
        print(f"Spawning {len(jobs)} parallel Pure legs...", flush=True)
        calls = [(label, train.spawn(**kw)) for label, kw in jobs]
        for label, _ in calls:
            print(f"  spawned {label}", flush=True)
        print("--- waiting for results ---", flush=True)
        for label, c in calls:
            try:
                m = c.get()
                print(f"[DONE] {label}: test={m['test']} -> {m['results_file']}",
                      flush=True)
            except Exception as e:  # keep collecting the rest
                print(f"[FAIL] {label}: {e!r}", flush=True)
    print("all legs settled", flush=True)


if __name__ == "__main__":
    main()
