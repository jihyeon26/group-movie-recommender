"""Generate and evaluate the popularity baseline on the prepared user pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from group_movie_recommender.filtering.candidates import build_seen_item_sets
from group_movie_recommender.preprocessing.config import PreprocessingConfig
from group_movie_recommender.evaluation.metrics import (
    build_relevant_item_sets,
    evaluate_shared_rankings,
)
from group_movie_recommender.filtering.catalog import positive_events
from group_movie_recommender.shared.io import load_ratings, write_csv_gzip, write_json
from group_movie_recommender.algorithms.popularity import recommend_pairs_by_popularity
from group_movie_recommender.preprocessing.splitting import add_temporal_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "movie_lens32m",
        help="Directory containing the extracted MovieLens 32M CSV files.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "processed_movielens32m",
        help="Directory created by scripts/prepare_data.py.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "preprocessing.json",
        help="JSON file containing preprocessing thresholds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "popularity_baseline",
        help="Directory for recommendations and evaluation metrics.",
    )
    parser.add_argument("--k", type=int, default=10, help="Recommendation cutoff.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PreprocessingConfig.from_json(args.config)

    pairs = pd.read_csv(
        args.processed_dir / "dissimilar_pairs.csv.gz",
        usecols=["userA", "userB"],
        dtype={"userA": "int32", "userB": "int32"},
    )
    warm_catalog = pd.read_csv(
        args.processed_dir / "warm_movies.csv.gz",
        dtype={"movieId": "int32", "trainPositiveCount": "int32"},
    )

    ratings = load_ratings(args.data_dir)
    add_temporal_split(ratings, config, copy=False)
    pair_user_ids = np.unique(
        pairs[["userA", "userB"]].to_numpy(dtype=np.int32).ravel()
    )
    seen_items = build_seen_item_sets(ratings, pair_user_ids)

    recommendations = recommend_pairs_by_popularity(
        pairs,
        warm_catalog,
        seen_items,
        k=args.k,
    )

    test_positives = positive_events(ratings, "test", pair_user_ids)
    relevant_items = build_relevant_item_sets(
        test_positives,
        set(warm_catalog["movieId"].astype(int)),
    )
    pair_metrics, summary = evaluate_shared_rankings(
        recommendations,
        relevant_items,
        catalog_size=len(warm_catalog),
        k=args.k,
    )

    write_csv_gzip(
        recommendations,
        args.output_dir / "recommendations.csv.gz",
    )
    write_csv_gzip(pair_metrics, args.output_dir / "pair_metrics.csv.gz")
    write_json(summary, args.output_dir / "metrics.json")

    print(json.dumps(summary, indent=2))
    print(f"\nPopularity outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
