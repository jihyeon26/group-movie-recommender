"""Observed-item histories and warm, unseen candidate sets for two-user groups."""

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

