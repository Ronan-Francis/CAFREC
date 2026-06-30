"""CLI entry point.

Examples
--------
    python run.py --models SASRec HGN --dataset ml-100k
    python run.py --models CAFREC --dataset ml-100k --seed 42
    python run.py --list
"""
import argparse

from cafrec.runner import run_experiment
from cafrec.results import save_metrics
from cafrec.registry import list_models


def main():
    parser = argparse.ArgumentParser(description="Run models through the CAFREC harness.")
    parser.add_argument("--models", nargs="+", default=["SASRec", "HGN"],
                        help="registry keys to run")
    parser.add_argument("--dataset", default="ml-100k")
    parser.add_argument("--seed", type=int, default=None,
                        help="override the seed in base.yaml")
    parser.add_argument("--wandb", action="store_true", help="also log to Weights & Biases")
    parser.add_argument("--list", action="store_true", help="list registered models and exit")
    args = parser.parse_args()

    if args.list:
        print("Registered models:", ", ".join(list_models()))
        return

    overrides = {"seed": args.seed} if args.seed is not None else None
    for key in args.models:
        metrics = run_experiment(key, dataset=args.dataset, config_overrides=overrides)
        path = save_metrics(metrics, use_wandb=args.wandb)
        print(f"\n[{key}] test: {metrics['test']}")
        print(f"      saved -> {path}")


if __name__ == "__main__":
    main()
