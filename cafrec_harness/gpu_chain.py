"""Controller for the local GPU queues: gpu_grid, then gpu_tuned, then the analysis.

Run it from a terminal and leave the window open. It keeps Windows awake, optionally
waits for earlier worker processes to exit, clears leftover .lock files, and runs each
queue with N parallel local_run.py workers (jobs are claimed through <result>.lock files).
Finished runs are skipped, so stopping (Ctrl+C) and rerunning the same command resumes.

Progress is printed to the console and appended to results/local_gpu/chain.log:
every run start / finish / failure as it happens, plus a status line (done, running,
failed, elapsed, rough time left) every 10 minutes and after each finished run.

    .venv-gpu/Scripts/python gpu_chain.py --workers 3 --threads 4 [--wait-pids 123,456]
"""
from __future__ import annotations

import argparse
import ctypes
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "local_gpu"
CHAIN = OUT / "chain.log"
GPU_PY = HERE / ".venv-gpu" / "Scripts" / "python.exe"
CPU_PY = HERE / ".venv" / "Scripts" / "python.exe"

# queue -> (number of jobs, result-file glob). Keep in sync with local_run.py:
# gpu_grid = 16 grid points x 2 models; gpu_tuned = 7 headline seeds x 3 conditions.
QUEUES = {
    "gpu_grid": (32, "*_gs16_*_seed*.json"),
    "gpu_tuned": (21, "*_tuned16_*_seed*.json"),
}
POLL_S, STATUS_EVERY_S = 15, 600


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(CHAIN, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def hm(seconds):
    m = int(seconds // 60)
    return f"{m // 60}h{m % 60:02d}m" if m >= 60 else f"{m}m"


def alive(pid):
    """True while the process exists (OpenProcess + GetExitCodeProcess; never signals it)."""
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, pid)             # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return code.value == 259                          # STILL_ACTIVE


class QueueWatch:
    """Follows results/local_gpu/queue_<name>.log from its current end and turns the
    START / DONE / FAIL lines written by local_run.run_queue into short console events."""

    def __init__(self, name):
        self.path = OUT / f"queue_{name}.log"
        self.offset = self.path.stat().st_size if self.path.exists() else 0
        self.running = {}                             # "tag seed" -> start time
        self.walls, self.failed = [], 0

    def poll(self):
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8", errors="replace") as fh:
            fh.seek(self.offset)
            chunk = fh.read()
            self.offset = fh.tell()
        events = []
        for line in chunk.splitlines():
            if " START " in line:
                tag, seed = re.search(r"'tag': '([^']+)'", line), re.search(r"'seed': (\d+)", line)
                job = f"{tag.group(1) if tag else '?'} seed {seed.group(1) if seed else '?'}"
                self.running[job] = time.time()
                events.append(f"START {job}")
            elif " DONE " in line:
                m = re.search(r"_((?:gs16|tuned16)_\S+?)_seed(\d+)\.json", line)
                job = f"{m.group(1)} seed {m.group(2)}" if m else "?"
                wall = re.search(r"wall=([\d.]+)s", line)
                hr = re.search(r"'hit@10': ([\d.]+)", line)
                self.running.pop(job, None)
                if wall:
                    self.walls.append(float(wall.group(1)))
                events.append(f"DONE  {job} in {hm(float(wall.group(1))) if wall else '?'}"
                              + (f", test HR@10 {hr.group(1)}" if hr else ""))
            elif " FAIL " in line:
                self.failed += 1
                tag, seed = re.search(r"'tag': '([^']+)'", line), re.search(r"'seed': (\d+)", line)
                job = f"{tag.group(1) if tag else '?'} seed {seed.group(1) if seed else '?'}"
                self.running.pop(job, None)
                events.append(f"FAIL  {job} (traceback in {self.path.name})")
        return events


def status(name, watch, workers, t_start, chain_start):
    total, pattern = QUEUES[name]
    done = len(list(OUT.glob(pattern)))
    now = time.time()
    running = ", ".join(f"{job} ({hm(now - t0)})" for job, t0 in watch.running.items()) or "none"
    line = (f"[{name}] {done}/{total} done, {len(watch.running)} running, {watch.failed} failed "
            f"| queue {hm(now - t_start)}, total {hm(now - chain_start)}")
    if watch.walls:
        avg = sum(watch.walls) / len(watch.walls)
        left = max(total - done - len(watch.running), 0) + 0.5 * len(watch.running)
        later = sum(QUEUES[q][0] for q in list(QUEUES)[list(QUEUES).index(name) + 1:])
        line += (f" | avg run {hm(avg)}, rough time left: this queue ~{hm(left * avg / workers)}, "
                 f"all ~{hm((left + later) * avg / workers)}")
    else:
        line += " | time left: estimated after the first run finishes"
    log(line)
    log(f"[{name}] running: {running}")


def run_queue(name, n, threads, chain_start):
    env = dict(os.environ, PYTHONUTF8="1", OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads))
    total, pattern = QUEUES[name]
    log(f"{name} start: {len(list(OUT.glob(pattern)))}/{total} already done, {n} workers")
    watch, t_start = QueueWatch(name), time.time()
    procs = []
    for w in range(1, n + 1):
        out = open(OUT / f"{name}_w{w}_stdout.log", "a")
        err = open(OUT / f"{name}_w{w}_stderr.log", "a")
        procs.append(subprocess.Popen([str(GPU_PY), "local_run.py", "--queue", name],
                                      cwd=HERE, env=env, stdout=out, stderr=err))
        time.sleep(20)                                # stagger data loading
        for e in watch.poll():
            log(f"[{name}] {e}")
    last_status = 0.0
    while any(p.poll() is None for p in procs):
        events = watch.poll()
        for e in events:
            log(f"[{name}] {e}")
        if any(e.startswith("DONE") for e in events) or time.time() - last_status > STATUS_EVERY_S:
            status(name, watch, n, t_start, chain_start)
            last_status = time.time()
        time.sleep(POLL_S)
    for e in watch.poll():
        log(f"[{name}] {e}")
    log(f"{name} finished: {len(list(OUT.glob(pattern)))}/{total} done, {watch.failed} failed "
        f"this session, took {hm(time.time() - t_start)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads per worker")
    ap.add_argument("--wait-pids", default="")
    a = ap.parse_args()
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # stay awake
    OUT.mkdir(parents=True, exist_ok=True)
    chain_start = time.time()
    log(f"controller started, pid {os.getpid()}, {a.workers} workers x {a.threads} CPU threads. "
        f"Ctrl+C stops; rerun the same command to resume.")
    try:
        pids = [int(p) for p in a.wait_pids.split(",") if p.strip()]
        if pids:
            log(f"waiting for earlier workers {pids} to finish their current runs")
            while any(alive(p) for p in pids):
                time.sleep(30)
            log("earlier workers have exited")
        for lock in OUT.glob("*.lock"):               # no worker is running at this point
            lock.unlink()

        for name in QUEUES:
            run_queue(name, a.workers, a.threads, chain_start)

        log("running analyze_tuned_gpu.py")
        res = subprocess.run([str(CPU_PY), "analyze_tuned_gpu.py"], cwd=HERE,
                             stdout=open(OUT / "analysis_stdout.log", "w"), stderr=subprocess.STDOUT)
        log(f"analysis written to results/tuned_gpu_{time.strftime('%Y%m%d')}.txt"
            if res.returncode == 0 else "analysis FAILED, see results/local_gpu/analysis_stdout.log")
        log(f"CHAIN DONE after {hm(time.time() - chain_start)}")
    except KeyboardInterrupt:
        log("stopped by Ctrl+C. Finished runs are kept; rerun the same command to resume.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
