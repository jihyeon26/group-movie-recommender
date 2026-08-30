"""Train LightGCN on a controlled MovieLens subset and export final embeddings."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from group_movie_recommender.pipelines.lightgcn_training import (
    LightGCNTrainingConfig,
    train_lightgcn,
)
from group_movie_recommender.preprocessing.graph_subset import build_controlled_graph_subset
from group_movie_recommender.shared.io import write_csv_gzip, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/processed_movielens32m",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/lightgcn_subset.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/lightgcn_subset",
    )
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    training_config = LightGCNTrainingConfig(**settings["training"])
    subset_settings = settings["subset"]

    train_edges = pd.read_csv(
        args.processed_dir / "train_positive_edges.csv.gz",
        dtype={"userId": "int32", "movieId": "int32"},
    )
    pairs = pd.read_csv(
        args.processed_dir / "dissimilar_pairs.csv.gz",
        usecols=["userA", "userB"],
        dtype={"userA": "int32", "userB": "int32"},
    )
    subset = build_controlled_graph_subset(
        train_edges,
        pairs,
        pair_limit=int(subset_settings["pair_limit"]),
        total_user_limit=int(subset_settings["total_user_limit"]),
        random_seed=training_config.random_seed,
    )
    model, history = train_lightgcn(subset.graph, training_config)
    with torch.no_grad():
        user_embeddings, movie_embeddings = model.propagate()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "final_embeddings.npz",
        user_ids=subset.graph.user_ids,
        user_embeddings=user_embeddings.cpu().numpy(),
        movie_ids=subset.graph.movie_ids,
        movie_embeddings=movie_embeddings.cpu().numpy(),
    )
    subset_catalog = (
        subset.positive_edges.groupby("movieId")
        .size()
        .rename("trainPositiveCount")
        .sort_values(ascending=False)
        .reset_index()
    )
    write_csv_gzip(subset.pairs, args.output_dir / "focus_pairs.csv.gz")
    write_csv_gzip(subset_catalog, args.output_dir / "training_catalog.csv.gz")
    write_csv_gzip(history, args.output_dir / "training_history.csv.gz")
    initial = float(history.iloc[0]["fixed_training_bpr_loss"])
    final = float(history.iloc[-1]["fixed_training_bpr_loss"])
    summary = {
        "purpose": "Controlled MovieLens integration run; not final model performance",
        "torch_version": torch.__version__,
        "device": "cpu",
        "subset": {
            **subset_settings,
            "focus_users": int(len(subset.focus_user_ids)),
            "context_users": int(len(subset.context_user_ids)),
            "graph_users": subset.graph.num_users,
            "graph_movies": subset.graph.num_movies,
            "positive_edges": subset.graph.num_positive_edges,
        },
        "training": asdict(training_config),
        "initial_fixed_training_bpr_loss": initial,
        "final_fixed_training_bpr_loss": final,
        "loss_decreased": final < initial,
    }
    write_json(summary, args.output_dir / "summary.json")
    print(json.dumps(summary, indent=2))
    print(f"\nSubset outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
