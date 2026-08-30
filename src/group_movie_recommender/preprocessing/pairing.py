"""Training-only construction of synthetic two-user groups."""

from __future__ import annotations

from collections import defaultdict
import gc
from typing import Any

import numpy as np
import pandas as pd

from .config import PreprocessingConfig


def build_centered_genre_profiles(
    train_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_ids: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Build normalized genre preference vectors centered on each user's like rate."""

    genre_matrix = (
        movies.set_index("movieId")["genres"]
        .str.get_dummies("|")
        .astype("float32")
    )
    genre_columns = genre_matrix.columns.tolist()
    expanded = train_ratings.merge(
        genre_matrix.reset_index(),
        on="movieId",
        how="left",
        validate="many_to_one",
    )

    exposure = (
        expanded.groupby("userId")[genre_columns]
        .sum()
        .reindex(user_ids, fill_value=0)
        .to_numpy(dtype="float32")
    )
    likes = (
        expanded.loc[expanded["rating"] >= config.positive_threshold]
        .groupby("userId")[genre_columns]
        .sum()
        .reindex(user_ids, fill_value=0)
        .to_numpy(dtype="float32")
    )
    overall_like_rate = (
        train_ratings.assign(is_positive=train_ratings["rating"] >= config.positive_threshold)
        .groupby("userId")["is_positive"]
        .mean()
        .reindex(user_ids)
        .to_numpy(dtype="float32")
    )

    prior = config.genre_prior_strength
    profiles = (
        (likes + prior * overall_like_rate[:, None]) / (exposure + prior)
        - overall_like_rate[:, None]
    )
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    profiles = np.divide(profiles, norms, out=np.zeros_like(profiles), where=norms > 0)

    del expanded, exposure, likes
    gc.collect()
    return profiles


def sample_pair_features(
    train_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_ids: np.ndarray,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    """Sample user pairs and compute genre similarity and rating correlation."""

    ordered_users = np.sort(np.asarray(user_ids, dtype=np.int32))
    if len(ordered_users) < 2:
        raise ValueError("At least two users are required to construct pairs")

    profiles = build_centered_genre_profiles(train_ratings, movies, ordered_users, config)
    rating_maps = {
        int(user_id): dict(zip(group["movieId"].astype(int), group["rating"].astype(float)))
        for user_id, group in train_ratings.groupby("userId", sort=False)
    }

    rng = np.random.default_rng(config.random_seed)
    n_users = len(ordered_users)
    left = rng.integers(0, n_users, size=config.pair_candidate_sample_size)
    right = rng.integers(0, n_users, size=config.pair_candidate_sample_size)
    same_user = left == right
    while same_user.any():
        right[same_user] = rng.integers(0, n_users, size=int(same_user.sum()))
        same_user = left == right

    pair_indices = np.unique(
        np.column_stack([np.minimum(left, right), np.maximum(left, right)]),
        axis=0,
    )
    left_index = pair_indices[:, 0]
    right_index = pair_indices[:, 1]
    genre_similarity = np.einsum(
        "ij,ij->i",
        profiles[left_index],
        profiles[right_index],
    )

    co_rated_count = np.zeros(len(pair_indices), dtype=np.int32)
    rating_correlation = np.full(len(pair_indices), np.nan, dtype=np.float32)

    for row_index, (left_position, right_position) in enumerate(pair_indices):
        left_user = int(ordered_users[left_position])
        right_user = int(ordered_users[right_position])
        left_ratings = rating_maps[left_user]
        right_ratings = rating_maps[right_user]
        common_movies = left_ratings.keys() & right_ratings.keys()
        n_common = len(common_movies)
        co_rated_count[row_index] = n_common

        if n_common < config.pair_min_co_rated:
            continue
        left_values = np.fromiter(
            (left_ratings[movie_id] for movie_id in common_movies),
            dtype=np.float32,
            count=n_common,
        )
        right_values = np.fromiter(
            (right_ratings[movie_id] for movie_id in common_movies),
            dtype=np.float32,
            count=n_common,
        )
        if left_values.std() > 0 and right_values.std() > 0:
            rating_correlation[row_index] = np.corrcoef(left_values, right_values)[0, 1]

    return pd.DataFrame(
        {
            "userA": ordered_users[left_index],
            "userB": ordered_users[right_index],
            "coRatedTrain": co_rated_count,
            "genreSimilarity": genre_similarity.astype("float32"),
            "ratingCorrelation": rating_correlation,
        }
    )


def select_dissimilar_pairs(
    pair_features: pd.DataFrame,
    config: PreprocessingConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select bottom-quartile pairs and cap repeated use of each user."""

    eligible = pair_features.loc[
        (pair_features["coRatedTrain"] >= config.pair_min_co_rated)
        & pair_features["ratingCorrelation"].notna()
    ].copy()
    if eligible.empty:
        raise ValueError("No sampled pairs have enough co-rated training movies")

    genre_cutoff = float(
        eligible["genreSimilarity"].quantile(config.pair_similarity_quantile)
    )
    rating_cutoff = float(
        eligible["ratingCorrelation"].quantile(config.pair_similarity_quantile)
    )
    candidates = eligible.loc[
        (eligible["genreSimilarity"] <= genre_cutoff)
        & (eligible["ratingCorrelation"] <= rating_cutoff)
    ].sample(frac=1.0, random_state=config.random_seed)

    counts: defaultdict[int, int] = defaultdict(int)
    selected_indices: list[int] = []
    for index, row in candidates.iterrows():
        user_a = int(row["userA"])
        user_b = int(row["userB"])
        if counts[user_a] >= config.max_pairs_per_user:
            continue
        if counts[user_b] >= config.max_pairs_per_user:
            continue
        selected_indices.append(index)
        counts[user_a] += 1
        counts[user_b] += 1

    selected = candidates.loc[selected_indices].reset_index(drop=True)
    summary: dict[str, Any] = {
        "sampled_pair_features": int(len(pair_features)),
        "eligible_pairs": int(len(eligible)),
        "dissimilar_candidates": int(len(candidates)),
        "selected_pairs": int(len(selected)),
        "genre_similarity_cutoff": genre_cutoff,
        "rating_correlation_cutoff": rating_cutoff,
        "max_pairs_per_user": config.max_pairs_per_user,
    }
    return selected, summary
