"""User, item, and positive-edge filtering for graph recommendation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..preprocessing.config import PreprocessingConfig


def positive_training_edges(
    ratings: pd.DataFrame,
    user_ids: np.ndarray,
) -> pd.DataFrame:
    """Return unique positive training user--movie edges for selected users."""

    return (
        ratings.loc[
            (ratings["split"] == "train")
            & ratings["is_positive"]
            & ratings["userId"].isin(user_ids),
            ["userId", "movieId"],
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def build_warm_catalog(
    train_edges: pd.DataFrame,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    """Create the warm-item catalogue from positive training edge counts."""

    counts = train_edges.groupby("movieId").size().rename("trainPositiveCount")
    return (
        counts.loc[counts >= config.warm_item_min_train_positives]
        .sort_values(ascending=False)
        .reset_index()
    )


def filter_edges_to_catalog(
    train_edges: pd.DataFrame,
    warm_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Remove graph edges pointing to items outside the warm catalogue."""

    return train_edges.loc[
        train_edges["movieId"].isin(warm_catalog["movieId"])
    ].reset_index(drop=True)


def positive_events(
    ratings: pd.DataFrame,
    split_name: str,
    user_ids: np.ndarray,
) -> pd.DataFrame:
    """Return positive held-out events for a split and user cohort."""

    return ratings.loc[
        (ratings["split"] == split_name)
        & ratings["is_positive"]
        & ratings["userId"].isin(user_ids),
        ["userId", "movieId"],
    ].reset_index(drop=True)


def warm_event_coverage(events: pd.DataFrame, warm_catalog: pd.DataFrame) -> float:
    """Measure the share of held-out positives covered by the warm catalogue."""

    if events.empty:
        return 0.0
    return float(events["movieId"].isin(warm_catalog["movieId"]).mean())
