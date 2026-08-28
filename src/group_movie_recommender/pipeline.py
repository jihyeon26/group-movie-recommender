"""End-to-end orchestration for reproducible preprocessing outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .candidates import add_joint_test_counts
from .config import PreprocessingConfig
from .filtering import (
    build_warm_catalog,
    filter_edges_to_catalog,
    positive_events,
    positive_training_edges,
    warm_event_coverage,
)
from .io import load_movies, load_ratings, write_csv_gzip, write_json
from .pairing import sample_pair_features, select_dissimilar_pairs
from .splitting import (
    add_temporal_split,
    select_evaluation_users,
    select_training_users,
    split_summary,
    user_split_statistics,
)


def run_preprocessing(
    data_dir: str | Path,
    output_dir: str | Path,
    config: PreprocessingConfig,
) -> dict[str, Any]:
    """Run splitting, graph filtering, pair construction, and artifact export."""

    output_path = Path(output_dir)
    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)
    add_temporal_split(ratings, config, copy=False)

    user_statistics = user_split_statistics(ratings)
    training_user_ids = select_training_users(user_statistics, config)
    evaluation_user_ids = select_evaluation_users(user_statistics, config)

    all_train_edges = positive_training_edges(ratings, training_user_ids)
    warm_catalog = build_warm_catalog(all_train_edges, config)
    train_graph_edges = filter_edges_to_catalog(all_train_edges, warm_catalog)

    evaluation_train_ratings = ratings.loc[
        (ratings["split"] == "train") & ratings["userId"].isin(evaluation_user_ids),
        ["userId", "movieId", "rating"],
    ].copy()
    pair_features = sample_pair_features(
        evaluation_train_ratings,
        movies,
        evaluation_user_ids,
        config,
    )
    selected_pairs, pair_summary = select_dissimilar_pairs(pair_features, config)

    test_positives = positive_events(ratings, "test", evaluation_user_ids)
    selected_pairs = add_joint_test_counts(selected_pairs, test_positives)
    test_coverage = warm_event_coverage(test_positives, warm_catalog)

    write_csv_gzip(train_graph_edges, output_path / "train_positive_edges.csv.gz")
    write_csv_gzip(user_statistics, output_path / "user_split_statistics.csv.gz")
    write_csv_gzip(
        pd.DataFrame({"userId": evaluation_user_ids}),
        output_path / "evaluation_users.csv.gz",
    )
    write_csv_gzip(warm_catalog, output_path / "warm_movies.csv.gz")
    write_csv_gzip(selected_pairs, output_path / "dissimilar_pairs.csv.gz")

    split_records = split_summary(ratings).reset_index().to_dict(orient="records")
    for row in split_records:
        row["split"] = str(row["split"])
        row["interactions"] = int(row["interactions"])
        row["users"] = int(row["users"])
        row["movies"] = int(row["movies"])
        row["positive_rate"] = float(row["positive_rate"])

    manifest: dict[str, Any] = {
        "config": config.to_dict(),
        "split_summary": split_records,
        "training_users": int(len(training_user_ids)),
        "evaluation_users": int(len(evaluation_user_ids)),
        "warm_movies": int(len(warm_catalog)),
        "train_graph_edges": int(len(train_graph_edges)),
        "warm_test_positive_coverage": test_coverage,
        "pair_summary": pair_summary,
        "observable_joint_positive_rate": float(
            (selected_pairs["jointTestPositiveCount"] >= 1).mean()
        ) if len(selected_pairs) else 0.0,
    }
    write_json(manifest, output_path / "manifest.json")
    return manifest
