from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mhrg.config import load_yaml_config
from mhrg.data.loaders import build_dataloaders
from mhrg.training.trainer import aggregate_runs, train_all_seeds
from mhrg.utils.io import ensure_dir, save_json
from mhrg.utils.seed import seed_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MH-RG DeepONet on the NLSE collision benchmark.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    seed_all(config["runtime"]["base_seed"])
    torch.backends.cudnn.deterministic = bool(config["runtime"].get("deterministic", True))
    torch.backends.cudnn.benchmark = False

    for key in ["checkpoint_root", "metrics_dir", "figures_dir"]:
        ensure_dir(config["experiment"][key])

    stats, train_loader, test_loader = build_dataloaders(config)
    order, all_runs = train_all_seeds(config, stats, train_loader, test_loader, keep_models=bool(config["training"].get("keep_models", False)))
    agg = aggregate_runs(order, all_runs)
    metrics_path = Path(config["experiment"]["metrics_dir"]) / "aggregated_summary.json"
    save_json(metrics_path, agg)
    print(f"Saved aggregated metrics to {metrics_path}")


if __name__ == "__main__":
    main()
