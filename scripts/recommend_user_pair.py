"""Create an inspectable real-user group recommendation from two rating exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from group_movie_recommender.algorithms.embedding_inference import score_folded_in_pair
from group_movie_recommender.algorithms.group_ranking import (
    normalize_member_scores,
    rank_group_candidates,
)
from group_movie_recommender.preprocessing.external_ratings import (
    load_external_movie_ratings,
)
from group_movie_recommender.shared.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings-a", type=Path, required=True)
    parser.add_argument("--ratings-b", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset/movie_lens32m",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/processed_movielens32m",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/user_pair_recommendation",
    )
    parser.add_argument(
        "--embedding-artifact",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs/validation_selected_model/runs/lightgcn_d32_lr002/final_embeddings.npz"
        ),
    )
    parser.add_argument(
        "--conflict-weight", type=float, default=0.0,
        help="Validation selected 0.0; override only for a declared sensitivity analysis.",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    args = parser.parse_args()

    movies = pd.read_csv(
        args.data_dir / "movies.csv", usecols=["movieId", "title", "genres"]
    )
    links = pd.read_csv(
        args.data_dir / "links.csv",
        usecols=["movieId", "imdbId"],
        dtype={"imdbId": "Int64"},
    )
    warm_catalog = pd.read_csv(
        args.processed_dir / "warm_movies.csv.gz",
        usecols=["movieId", "trainPositiveCount"],
    )
    warm_ids = set(warm_catalog["movieId"].astype(int))
    ratings_a, import_a = load_external_movie_ratings(
        args.ratings_a, links, warm_ids
    )
    ratings_b, import_b = load_external_movie_ratings(
        args.ratings_b, links, warm_ids
    )
    with np.load(args.embedding_artifact) as artifact:
        model_movie_ids = artifact["movie_ids"]
        model_movie_embeddings = artifact["movie_embeddings"]
    raw_scores = score_folded_in_pair(
        ratings_a,
        ratings_b,
        model_movie_ids,
        model_movie_embeddings,
        positive_threshold=float(args.positive_threshold),
    )
    normalized = normalize_member_scores(raw_scores)

    methods = {
        "average": 0.0,
        "conflict_aware": float(args.conflict_weight),
        "least_misery": 1.0,
    }
    recommendations: list[pd.DataFrame] = []
    for method, weight in methods.items():
        ranked = rank_group_candidates(
            normalized,
            conflict_weight=weight,
            k=args.k,
        )
        ranked = ranked.merge(
            movies.loc[:, ["movieId", "title", "genres"]],
            on="movieId",
            how="left",
            validate="one_to_one",
        ).merge(
            warm_catalog,
            on="movieId",
            how="left",
            validate="one_to_one",
        )
        ranked.insert(0, "method", method)
        ranked.insert(1, "conflictWeight", weight)
        recommendations.append(ranked)
    comparison = pd.concat(recommendations, ignore_index=True)
    selected = comparison.loc[comparison["method"].eq("conflict_aware")].copy()

    model_ids = set(np.asarray(model_movie_ids, dtype=np.int64).tolist())
    positive_count_a = int(
        (
            ratings_a["rating"].ge(args.positive_threshold)
            & ratings_a["movieId"].isin(model_ids)
        ).sum()
    )
    positive_count_b = int(
        (
            ratings_b["rating"].ge(args.positive_threshold)
            & ratings_b["movieId"].isin(model_ids)
        ).sum()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "method_comparison.csv", index=False)
    selected.to_csv(args.output_dir / "recommendations.csv", index=False)
    summary = {
        "purpose": "Real-user LightGCN fold-in demonstration; not an offline benchmark result",
        "imports": {"userA": import_a, "userB": import_b},
        "modelArtifact": str(args.embedding_artifact.resolve()),
        "foldIn": {
            "positiveThreshold": float(args.positive_threshold),
            "userAPositiveRatingsInModel": positive_count_a,
            "userBPositiveRatingsInModel": positive_count_b,
        },
        "sharedRatedWarmMovies": int(
            len(set(ratings_a["movieId"].astype(int)) & set(ratings_b["movieId"].astype(int)))
        ),
        "candidateCount": int(len(raw_scores)),
        "k": int(args.k),
        "selectedMethod": (
            "lightgcn_fold_in_average" if args.conflict_weight == 0.0
            else "lightgcn_fold_in_conflict_aware"
        ),
        "conflictWeight": float(args.conflict_weight),
        "outputColumns": list(selected.columns),
    }
    write_json(summary, args.output_dir / "summary.json")

    display_columns = [
        "rank", "title", "genres", "qA", "qB",
        "groupScore", "disagreement",
    ]
    display = selected.loc[:, display_columns].copy()
    print(json.dumps(summary, indent=2))
    label = "average" if args.conflict_weight == 0.0 else "conflict-aware"
    print(f"\nLightGCN fold-in {label} recommendation:\n")
    print(display.to_string(index=False))
    print(f"\nOutputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
