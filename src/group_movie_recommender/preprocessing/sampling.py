"""Interaction sampling for sparse-profile experiments."""

from __future__ import annotations

import pandas as pd


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
