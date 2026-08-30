"""Run a reproducible CPU LightGCN learning check on a synthetic graph."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from group_movie_recommender.pipelines.lightgcn_training import (
    LightGCNTrainingConfig,
    build_toy_training_graph,
    train_small_graph,
)
from group_movie_recommender.shared.io import write_csv_gzip, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/lightgcn_toy.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/lightgcn_toy")
    args = parser.parse_args()
    config = LightGCNTrainingConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    graph = build_toy_training_graph()
    _, history = train_small_graph(graph, config)
    initial = float(history.iloc[0]["fixed_training_bpr_loss"])
    final = float(history.iloc[-1]["fixed_training_bpr_loss"])
    summary = {
        "purpose": "Synthetic training sanity check; not validation/test performance",
        "device": "cpu",
        "torch_version": torch.__version__,
        "config": asdict(config),
        "num_users": graph.num_users,
        "num_movies": graph.num_movies,
        "num_positive_edges": graph.num_positive_edges,
        "initial_fixed_training_bpr_loss": initial,
        "final_fixed_training_bpr_loss": final,
        "loss_decreased": final < initial,
    }
    write_csv_gzip(history, args.output_dir / "training_history.csv.gz")
    write_json(summary, args.output_dir / "summary.json")
    print(json.dumps(summary, indent=2))
    print(f"\nTraining outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
