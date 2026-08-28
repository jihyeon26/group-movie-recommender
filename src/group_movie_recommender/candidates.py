"""Candidate-set and held-out joint-relevance utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_seen_item_sets(
    ratings: pd.DataFrame,
    user_ids: np.ndarray,
    *,
    available_splits: tuple[str, ...] = ("train", "validation"),
) -> dict[int, set[int]]:
    """Map each user to items observed before test deployment."""

    observed = ratings.loc[
        ratings["userId"].isin(user_ids) & ratings["split"].isin(available_splits),
        ["userId", "movieId"],
    ]
    return {
        int(user_id): set(group["movieId"].astype(int))
        for user_id, group in observed.groupby("userId", sort=False)
    }


def candidate_movies_for_pair(
    warm_movie_ids: np.ndarray,
    seen_items: dict[int, set[int]],
    user_a: int,
    user_b: int,
) -> np.ndarray:
    """Return warm movies unseen by both members of a pair."""

    unavailable = seen_items.get(int(user_a), set()) | seen_items.get(int(user_b), set())
    candidates = set(np.asarray(warm_movie_ids, dtype=np.int32).tolist()) - unavailable
    return np.asarray(sorted(candidates), dtype=np.int32)


def add_joint_test_counts(
    pairs: pd.DataFrame,
    test_positive_events: pd.DataFrame,
) -> pd.DataFrame:
    """Add observable common positive counts without changing pair selection."""

    positive_sets = {
        int(user_id): set(group["movieId"].astype(int))
        for user_id, group in test_positive_events.groupby("userId", sort=False)
    }
    result = pairs.copy()
    result["jointTestPositiveCount"] = [
        len(positive_sets.get(int(row.userA), set()) & positive_sets.get(int(row.userB), set()))
        for row in result.itertuples(index=False)
    ]
    return result


def keep_first_k_interactions(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep the earliest k interactions per user for sparse-profile experiments."""

    if k <= 0:
        raise ValueError("k must be a positive integer")
    return (
        frame.sort_values(["userId", "timestamp"])
        .groupby("userId", group_keys=False)
        .head(k)
        .reset_index(drop=True)
    )

