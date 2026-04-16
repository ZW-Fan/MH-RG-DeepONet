from __future__ import annotations

import argparse
from pathlib import Path

from mhrg.config import load_yaml_config
from mhrg.data.loaders import build_dataloaders
from mhrg.training.trainer import aggregate_runs, train_all_seeds
from mhrg.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved NLSE checkpoints and aggregate metrics.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    config["training"]["epochs"] = int(config["training"]["epochs"])
    config["training"]["force_retrain"] = False
    stats, train_loader, test_loader = build_dataloaders(config)
    order, all_runs = train_all_seeds(config, stats, train_loader, test_loader, keep_models=False)
    agg = aggregate_runs(order, all_runs)
    metrics_path = Path(config["experiment"]["metrics_dir"]) / "aggregated_summary_eval.json"
    save_json(metrics_path, agg)
    print(f"Saved evaluation summary to {metrics_path}")


if __name__ == "__main__":
    main()
