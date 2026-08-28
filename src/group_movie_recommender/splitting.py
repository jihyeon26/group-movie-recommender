"""Temporal splitting and user-cohort selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PreprocessingConfig


SPLIT_ORDER = ["train", "validation", "test"]


def add_temporal_split(
    ratings: pd.DataFrame,
    config: PreprocessingConfig,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Add reproducible split and positive-feedback columns."""

    result = ratings.copy() if copy else ratings
    result["split"] = pd.Categorical(
        np.select(
            [
                result["timestamp"] < config.train_end_timestamp,
                result["timestamp"] < config.validation_end_timestamp,
            ],
            ["train", "validation"],
            default="test",
        ),
        categories=SPLIT_ORDER,
        ordered=True,
    )
    result["is_positive"] = result["rating"] >= config.positive_threshold
    return result


def split_summary(ratings: pd.DataFrame) -> pd.DataFrame:
    """Summarize interaction, user, item, and positive-feedback counts by split."""

    return ratings.groupby("split", observed=True).agg(
        interactions=("rating", "size"),
        users=("userId", "nunique"),
        movies=("movieId", "nunique"),
        positive_rate=("is_positive", "mean"),
    )


def user_split_statistics(ratings: pd.DataFrame) -> pd.DataFrame:
    """Count all and positive interactions for each user in each split."""

    activity = (
        ratings.groupby(["userId", "split"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SPLIT_ORDER, fill_value=0)
    )
    positives = (
        ratings.loc[ratings["is_positive"]]
        .groupby(["userId", "split"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=activity.index, columns=SPLIT_ORDER, fill_value=0)
    )

    statistics = activity.add_suffix("_interactions")
    for split_name in SPLIT_ORDER:
        statistics[f"{split_name}_positives"] = positives[split_name]
    return statistics.reset_index()


def select_training_users(
    user_statistics: pd.DataFrame,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Keep users with enough positive training evidence for graph learning."""

    selected = user_statistics.loc[
        user_statistics["train_positives"] >= config.training_min_positives,
        "userId",
    ]
    return selected.to_numpy(dtype=np.int32)


def select_evaluation_users(
    user_statistics: pd.DataFrame,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Select users with reliable histories and held-out relevance labels."""

    mask = (
        (user_statistics["train_interactions"] >= config.evaluation_min_train_ratings)
        & (user_statistics["train_positives"] >= config.evaluation_min_train_positives)
        & (user_statistics["validation_positives"] >= config.evaluation_min_validation_positives)
        & (user_statistics["test_positives"] >= config.evaluation_min_test_positives)
    )
    return user_statistics.loc[mask, "userId"].to_numpy(dtype=np.int32)

