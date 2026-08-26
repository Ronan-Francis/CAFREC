"""1K generalisation, take 2 — 10 epochs to remove the under-convergence confound.

logfull vs SASRec on kuairand_1k_kcore_ctx, 10 epochs, seed 2020, ranks dumped.
Tells us whether CAFREC closes the 5-epoch gap when properly trained.
"""
from modal_run import app, train

DS = "kuairand_1k_kcore_ctx"
PROF = "/data/profiles/kuairand_1k_kcore_ctx.profiles.hf.d1024.pt"

jobs = [
    ("SASRec", dict(model_key="SASRec", dataset=DS, epochs=10, seed=2020,
                    dump_ranks=True, tag="gen10")),
    ("logfull", dict(model_key="CAFREC", dataset=DS, epochs=10, seed=2020,
                     llm_profile_path=PROF, profile_dim=1024,
                     history_gate_mode="logfull", dump_ranks=True,
                     tag="logfull_gen10")),
]


def main():
    with app.run():
        calls = [(l, train.spawn(**kw)) for l, kw in jobs]
        for l, _ in calls:
            print(f"  spawned {l}", flush=True)
        print("--- waiting ---", flush=True)
        for l, c in calls:
            try:
                m = c.get()
                print(f"[DONE] {l}: test={m['test']} valid={m['valid_best']}", flush=True)
            except Exception as e:
                print(f"[FAIL] {l}: {e!r}", flush=True)
    print("settled", flush=True)


if __name__ == "__main__":
    main()
