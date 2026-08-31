"""Normalize external rating exports to warm MovieLens feedback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_external_movie_ratings(
    path: str | Path,
    links: pd.DataFrame,
    warm_movie_ids: set[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load an app/MovieLens CSV or IMDb ratings export.

    IMDb's 1--10 integer ratings are converted to MovieLens half-star ratings.
    Only rows whose title type is ``Movie`` are used from an IMDb export. All
    returned movies must be represented in the training-only warm catalogue.
    """

    source_path = Path(path)
    source = pd.read_csv(source_path)
    input_rows = len(source)
    if {"movieId", "rating"}.issubset(source.columns):
        source_type = "movielens"
        eligible = source.loc[:, ["movieId", "rating"]].copy()
        eligible["movieId"] = pd.to_numeric(eligible["movieId"], errors="coerce")
        eligible["rating"] = pd.to_numeric(eligible["rating"], errors="coerce")
        mapped_rows = int(eligible["movieId"].notna().sum())
        non_movie_rows = 0
        unmapped_rows = input_rows - mapped_rows
    elif {"Const", "Your Rating", "Title Type"}.issubset(source.columns):
        source_type = "imdb"
        movie_rows = source.loc[source["Title Type"].eq("Movie")].copy()
        non_movie_rows = input_rows - len(movie_rows)
        movie_rows["imdbId"] = pd.to_numeric(
            movie_rows["Const"].astype(str).str.extract(r"^tt(\d+)$", expand=False),
            errors="coerce",
        ).astype("Int64")
        link_table = links.loc[:, ["movieId", "imdbId"]].dropna(
            subset=["imdbId"]
        ).copy()
        link_table["imdbId"] = link_table["imdbId"].astype("Int64")
        eligible = movie_rows.merge(
            link_table,
            on="imdbId",
            how="left",
            validate="many_to_one",
        )
        eligible["rating"] = pd.to_numeric(
            eligible["Your Rating"], errors="coerce"
        ) / 2.0
        mapped_rows = int(eligible["movieId"].notna().sum())
        unmapped_rows = len(movie_rows) - mapped_rows
        eligible = eligible.loc[:, ["movieId", "rating"]]
    else:
        raise ValueError(
            f"{source_path.name} is neither a MovieLens/app export nor an IMDb ratings export"
        )

    valid = eligible.dropna(subset=["movieId", "rating"]).copy()
    valid["movieId"] = valid["movieId"].astype(np.int64)
    valid["rating"] = valid["rating"].astype(float)
    _validate_ratings(valid, source_path.name)
    valid = valid.drop_duplicates("movieId", keep="last")
    warm = valid.loc[valid["movieId"].isin(warm_movie_ids)].copy()
    cold_item_rows = len(valid) - len(warm)
    warm = warm.sort_values("movieId").reset_index(drop=True)
    return warm, {
        "file": source_path.name,
        "sourceType": source_type,
        "inputRows": int(input_rows),
        "nonMovieRows": int(non_movie_rows),
        "mappedRows": int(mapped_rows),
        "unmappedRows": int(unmapped_rows),
        "coldOrUnavailableMovieRows": int(cold_item_rows),
        "usableWarmRatings": int(len(warm)),
    }


def _validate_ratings(ratings: pd.DataFrame, name: str) -> None:
    if ratings.empty:
        raise ValueError(f"{name} contains no mapped numeric ratings")
    values = ratings["rating"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0.5) | (values > 5.0)).any():
        raise ValueError(f"{name} ratings must map to the range 0.5--5.0")
    if not np.allclose(values * 2.0, np.round(values * 2.0)):
        raise ValueError(f"{name} ratings must map to half-star increments")
