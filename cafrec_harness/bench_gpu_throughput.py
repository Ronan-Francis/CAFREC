"""How many parallel workers, and how many CPU threads each, get the most training done
on the local GPU? Run it while nothing else is training (about 10 minutes).

For each setup (workers x threads) it starts that many worker processes, each training
SASRec at the grid's default point (batch 512) on kuairand_pure, waits until all have
loaded their data, then times the same number of training steps in all of them at once.
The total batches/s across workers is the figure that matters. Prints a table and the
gpu_chain.py command for the fastest setup.

    .venv-gpu/Scripts/python bench_gpu_throughput.py [--setups 3x4,1x1,3x1,5x1] [--batches 150]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BATCHES_PER_EPOCH = 1138          # 582,363 training sequences / batch 512
RUNS_LEFT_EPOCHS = 45 * 10        # rough: remaining grid + tuned runs x 10 epochs


def worker(start_at, n):
    os.chdir(HERE)
    sys.path.insert(0, str(HERE))
    import torch
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed

    from cafrec.registry import get_spec
    from cafrec.runner import _resolve_model_class

    spec = get_spec("SASRec")
    cd = dict(spec.config); cd.update(spec.contract); cd.update(seed=403092, train_batch_size=512)
    cfg = Config(model=spec.model, dataset="kuairand_pure",
                 config_file_list=["configs/base.yaml"], config_dict=cd)
    init_seed(cfg["seed"], cfg["reproducibility"])
    train, _, _ = data_preparation(cfg, create_dataset(cfg))
    model = _resolve_model_class(spec, cfg)(cfg, train.dataset).to(cfg["device"])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    it = iter(train)

    def step():
        nonlocal it
        try:
            b = next(it)
        except StopIteration:
            it = iter(train); b = next(it)
        b = b.to(cfg["device"])
        opt.zero_grad(); loss = model.calculate_loss(b); loss.backward(); opt.step()
        return loss.item()                            # RecBole's trainer syncs on the loss too

    for _ in range(5):
        step()
    if time.time() > start_at:
        print("LATE", flush=True)                     # data loading overran the start time
    while time.time() < start_at:
        time.sleep(0.05)
    t0 = time.perf_counter()
    for _ in range(n):
        step()
    print(f"RATE {n / (time.perf_counter() - t0):.3f}", flush=True)


def run_setup(k, threads, n, load_s):
    env = dict(os.environ, PYTHONUTF8="1", OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    start_at = time.time() + load_s
    procs = [subprocess.Popen([sys.executable, __file__, "--worker", str(start_at), str(n)],
                              cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              text=True) for _ in range(k)]
    rates, late = [], False
    for p in procs:
        out, _ = p.communicate()
        late |= "LATE" in out
        rates += [float(line.split()[1]) for line in out.splitlines() if line.startswith("RATE")]
    return rates, late


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setups", default="3x4,1x1,3x1,5x1",
                    help="comma-separated WORKERSxTHREADS, e.g. 3x4,5x1")
    ap.add_argument("--batches", type=int, default=150, help="timed batches per worker")
    ap.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        return worker(float(a.worker[0]), int(a.worker[1]))

    setups = [tuple(int(x) for x in s.lower().split("x")) for s in a.setups.split(",")]
    print(f"Benchmarking {len(setups)} setups, {a.batches} timed batches per worker "
          f"(about 2-3 min each). Do not train anything else meanwhile.\n", flush=True)
    results = []
    for k, t in setups:
        print(f"  {k} workers x {t} threads ... ", end="", flush=True)
        rates, late = run_setup(k, t, a.batches, load_s=60 + 15 * k)
        if len(rates) < k:
            print(f"FAILED ({len(rates)}/{k} workers reported)", flush=True)
            continue
        total = sum(rates)
        results.append((k, t, total))
        print(f"{total:6.2f} batches/s total = {60 * total / BATCHES_PER_EPOCH:4.2f} epochs/min "
              f"(per worker: {', '.join(f'{r:.2f}' for r in rates)})"
              + ("  [warning: a worker started late]" if late else ""), flush=True)
    if not results:
        return 1

    base = next((r for r in results if (r[0], r[1]) == (3, 4)), results[0])
    print("\n  setup          batches/s   vs 3x4   rough training time for the remaining runs")
    for k, t, total in sorted(results, key=lambda r: -r[2]):
        hours = RUNS_LEFT_EPOCHS * BATCHES_PER_EPOCH / total / 3600
        print(f"  {k} x {t} threads   {total:8.2f}   {total / base[2]:5.2f}x   ~{hours:.1f} h "
              f"(training steps only; validation adds more)")
    k, t, _ = max(results, key=lambda r: r[2])
    py = HERE / ".venv-gpu" / "Scripts" / "python.exe"
    print(f"\nFastest: {k} workers x {t} threads. Command:\n"
          f"\"{py}\" \"{HERE / 'gpu_chain.py'}\" --workers {k} --threads {t}")


if __name__ == "__main__":
    sys.exit(main())
