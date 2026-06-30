"""Persist metrics to a committed artifact (not just stdout).

Writes one JSON per run AND appends a flat row to results/summary.csv so you
can diff models at a glance. W&B logging is opt-in and silently skipped if
wandb isn't installed.
"""
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def _flatten(metrics):
    row = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    for k, v in metrics.get("test", {}).items():
        row[f"test_{k}"] = v
    return row


def save_metrics(metrics, results_dir=DEFAULT_RESULTS_DIR, use_wandb=False):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{metrics['model']}_{metrics['dataset']}_seed{metrics['seed']}_{metrics['timestamp']}"
    json_path = results_dir / f"{stem}.json"
    json_path.write_text(json.dumps(metrics, indent=2))

    _append_summary(_flatten(metrics), results_dir / "summary.csv")

    if use_wandb:
        _log_wandb(metrics)

    return json_path


def _append_summary(row, csv_path):
    existing = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            existing = list(csv.DictReader(f))
    fieldnames = sorted({*[k for r in existing for k in r], *row})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in (*existing, row):
            writer.writerow(r)


def _log_wandb(metrics):
    try:
        import wandb
    except ImportError:
        return
    run = wandb.init(project="cafrec", name=f"{metrics['model']}-{metrics['timestamp']}",
                     config={"model": metrics["model"], "dataset": metrics["dataset"],
                             "seed": metrics["seed"]})
    run.log({f"test/{k}": v for k, v in metrics.get("test", {}).items()})
    run.finish()
