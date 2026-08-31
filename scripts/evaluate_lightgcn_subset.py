"""Compare popularity and group rankers on the controlled subset validation period."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from group_movie_recommender.algorithms.embedding_inference import recommend_pairs_from_embeddings
from group_movie_recommender.algorithms.popularity import recommend_pairs_by_popularity
from group_movie_recommender.evaluation.metrics import (
    build_pair_relevant_item_sets,
    evaluate_shared_rankings,
)
from group_movie_recommender.evaluation.ranking_diagnostics import (
    compare_rankings_to_reference,
)
from group_movie_recommender.evaluation.uncertainty import (
    paired_bootstrap_mean_difference,
)
from group_movie_recommender.filtering.candidates import build_seen_item_sets
from group_movie_recommender.filtering.catalog import positive_events
from group_movie_recommender.preprocessing.config import PreprocessingConfig
from group_movie_recommender.preprocessing.splitting import add_temporal_split
from group_movie_recommender.shared.io import load_ratings, write_csv_gzip, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "dataset/movie_lens32m")
    parser.add_argument("--embedding-dir", type=Path, default=PROJECT_ROOT / "outputs/lightgcn_subset")
    parser.add_argument("--preprocessing-config", type=Path, default=PROJECT_ROOT / "configs/preprocessing.json")
    parser.add_argument("--ranking-config", type=Path, default=PROJECT_ROOT / "configs/ranking_validation.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/lightgcn_subset_validation")
    parser.add_argument(
        "--method-prefix",
        default="lightgcn",
        help="Prefix used to label personalized ranking methods in exported results.",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.method_prefix):
        raise ValueError("method-prefix must be a lowercase identifier")

    preprocessing = PreprocessingConfig.from_json(args.preprocessing_config)
    ranking = json.loads(args.ranking_config.read_text(encoding="utf-8"))
    k = int(ranking["k"])
    weights = [float(value) for value in ranking["conflict_weights"]]
    bootstrap_resamples = int(ranking["bootstrap_resamples"])
    bootstrap_seed = int(ranking["bootstrap_seed"])
    if 0.0 not in weights:
        raise ValueError("conflict_weights must include 0.0 for the average baseline")

    pairs = pd.read_csv(args.embedding_dir / "focus_pairs.csv.gz")
    artifact = np.load(args.embedding_dir / "final_embeddings.npz")
    user_ids = artifact["user_ids"]
    user_embeddings = artifact["user_embeddings"]
    movie_ids = artifact["movie_ids"]
    movie_embeddings = artifact["movie_embeddings"]
    pair_user_ids = np.unique(pairs[["userA", "userB"]].to_numpy().ravel()).astype(np.int32)

    ratings = load_ratings(args.data_dir)
    add_temporal_split(ratings, preprocessing, copy=False)
    seen_train = build_seen_item_sets(
        ratings,
        pair_user_ids,
        available_splits=("train",),
    )
    validation_positives = positive_events(ratings, "validation", pair_user_ids)
    eligible_movies = set(movie_ids.astype(int))
    pair_relevance = build_pair_relevant_item_sets(
        pairs,
        validation_positives,
        eligible_movies,
        seen_train,
    )

    recommendation_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    subset_catalog = pd.read_csv(args.embedding_dir / "training_catalog.csv.gz")
    if set(subset_catalog["movieId"].astype(int)) != eligible_movies:
        raise ValueError("Subset popularity catalogue and embedding movies do not match")
    popularity = recommend_pairs_by_popularity(pairs, subset_catalog, seen_train, k=k)
    popularity.insert(0, "method", "popularity")
    pair_metrics, summary = evaluate_shared_rankings(
        popularity,
        pair_relevance,
        catalog_size=len(eligible_movies),
        k=k,
    )
    pair_metrics.insert(0, "method", "popularity")
    recommendation_frames.append(popularity)
    metric_frames.append(pair_metrics)
    summaries.append({"method": "popularity", **summary})

    average_method = f"{args.method_prefix}_average"
    for weight in weights:
        method = average_method if weight == 0.0 else f"{args.method_prefix}_conflict_{weight:g}"
        recommendations = recommend_pairs_from_embeddings(
            pairs,
            user_ids,
            user_embeddings,
            movie_ids,
            movie_embeddings,
            seen_train,
            conflict_weight=weight,
            k=k,
        )
        recommendations.insert(0, "method", method)
        pair_metrics, summary = evaluate_shared_rankings(
            recommendations,
            pair_relevance,
            catalog_size=len(eligible_movies),
            k=k,
        )
        pair_metrics.insert(0, "method", method)
        recommendation_frames.append(recommendations)
        metric_frames.append(pair_metrics)
        summaries.append({"method": method, "conflict_weight": weight, **summary})

    total_targets = 0
    available_targets = 0
    positives_by_user = {
        int(user): set(group["movieId"].astype(int))
        for user, group in validation_positives.groupby("userId", sort=False)
    }
    for row in pairs.itertuples(index=False):
        for user in (int(row.userA), int(row.userB)):
            total_targets += len(positives_by_user.get(user, set()))
            available_targets += len(pair_relevance.get((int(row.pairId), user), set()))

    key = f"meanMinimumNDCG@{k}"
    personalized = [row for row in summaries if row["method"] != "popularity"]
    best_primary = max(float(row[key]) for row in personalized)
    primary_ties = [row for row in personalized if float(row[key]) == best_primary]
    selected = max(primary_ties, key=lambda row: float(row[f"meanAverageNDCG@{k}"]))
    all_recommendations = pd.concat(recommendation_frames, ignore_index=True)
    all_pair_metrics = pd.concat(metric_frames, ignore_index=True)
    ranking_diagnostics = compare_rankings_to_reference(
        all_recommendations,
        reference_method=average_method,
        k=k,
    )
    comparison_pairs = list(
        dict.fromkeys(
            [
                (average_method, "popularity"),
                (str(selected["method"]), average_method),
                (str(selected["method"]), "popularity"),
            ]
        )
    )
    bootstrap_comparisons = [
        paired_bootstrap_mean_difference(
            all_pair_metrics,
            method_a=method_a,
            method_b=method_b,
            metric=f"minimumNDCG@{k}",
            n_resamples=bootstrap_resamples,
            random_seed=bootstrap_seed,
        )
        for method_a, method_b in comparison_pairs
        if method_a != method_b
    ]
    report = {
        "purpose": "Subset validation integration; not final full-catalogue performance",
        "k": k,
        "pairs": int(len(pairs)),
        "candidate_movies": int(len(eligible_movies)),
        "pair_validation_target_coverage": available_targets / total_targets if total_targets else 0.0,
        "selected_personalized_method": selected["method"],
        "selection_metric": key,
        "selection_metric_value": best_primary,
        "methods_tied_on_selection_metric": [row["method"] for row in primary_ties],
        "personalized_method_prefix": args.method_prefix,
        "ranking_diagnostics_reference": average_method,
        "ranking_diagnostics": ranking_diagnostics.to_dict(orient="records"),
        "paired_bootstrap_validation_diagnostics": bootstrap_comparisons,
        "methods": summaries,
    }
    write_csv_gzip(all_recommendations, args.output_dir / "recommendations.csv.gz")
    write_csv_gzip(all_pair_metrics, args.output_dir / "pair_metrics.csv.gz")
    write_csv_gzip(
        ranking_diagnostics,
        args.output_dir / "ranking_diagnostics.csv.gz",
    )
    write_json(report, args.output_dir / "validation_report.json")
    print(json.dumps(report, indent=2))
    print(f"\nValidation outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
