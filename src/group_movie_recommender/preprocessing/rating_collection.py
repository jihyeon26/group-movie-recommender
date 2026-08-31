"""Build MovieLens-compatible rows from one participant's live ratings."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


MOVIELENS_RATING_COLUMNS = ["userId", "movieId", "rating", "timestamp"]


def build_movielens_ratings(
    records: Mapping[int, Mapping[str, float | int]],
    *,
    user_id: int,
) -> pd.DataFrame:
    """Validate session records and return exact MovieLens rating columns."""

    if not 0 < int(user_id) <= np.iinfo(np.int32).max:
        raise ValueError("user_id must be a positive 32-bit integer")

    rows = [
        {
            "userId": int(user_id),
            "movieId": int(movie_id),
            "rating": float(record["rating"]),
            "timestamp": int(record["timestamp"]),
        }
        for movie_id, record in records.items()
    ]
    frame = pd.DataFrame(rows, columns=MOVIELENS_RATING_COLUMNS)
    if frame.empty:
        return frame.astype(
            {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}
        )
    if (frame["movieId"] <= 0).any() or frame["movieId"].duplicated().any():
        raise ValueError("movieId values must be positive and unique")
    ratings = frame["rating"].to_numpy(dtype=float)
    if (
        not np.isfinite(ratings).all()
        or ((ratings < 0.5) | (ratings > 5.0)).any()
        or not np.allclose(ratings * 2, np.round(ratings * 2))
    ):
        raise ValueError("Ratings must use half-star increments from 0.5 to 5.0")
    if (frame["timestamp"] <= 0).any():
        raise ValueError("Timestamps must be positive Unix seconds")
    return frame.sort_values("movieId").reset_index(drop=True)
