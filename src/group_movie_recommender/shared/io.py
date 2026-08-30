"""Shared, memory-aware input and output helpers for MovieLens 32M."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_MOVIELENS_FILES = (
    "ratings.csv",
    "movies.csv",
    "links.csv",
    "tags.csv",
)

RATING_DTYPES = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
    "timestamp": "int64",
}

MOVIE_DTYPES = {
    "movieId": "int32",
    "title": "string",
    "genres": "string",
}

LINK_DTYPES = {
    "movieId": "int32",
    "imdbId": "Int64",
    "tmdbId": "Int64",
}

TAG_DTYPES = {
    "userId": "int32",
    "movieId": "int32",
    "tag": "string",
    "timestamp": "int64",
}


def require_movielens_files(data_dir: str | Path) -> Path:
    """Validate the expected MovieLens directory and return its resolved path."""

    resolved = Path(data_dir).resolve()
    missing = [name for name in REQUIRED_MOVIELENS_FILES if not (resolved / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing MovieLens files in {resolved}: {missing}")
    return resolved


def load_ratings(data_dir: str | Path) -> pd.DataFrame:
    """Load the large ratings table using compact dtypes."""

    data_path = require_movielens_files(data_dir)
    return pd.read_csv(data_path / "ratings.csv", dtype=RATING_DTYPES)


def load_movies(data_dir: str | Path) -> pd.DataFrame:
    """Load movie titles and genres."""

    data_path = require_movielens_files(data_dir)
    return pd.read_csv(data_path / "movies.csv", dtype=MOVIE_DTYPES)


def load_all_tables(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load all four MovieLens tables for data-quality validation."""

    data_path = require_movielens_files(data_dir)
    return {
        "ratings": pd.read_csv(data_path / "ratings.csv", dtype=RATING_DTYPES),
        "movies": pd.read_csv(data_path / "movies.csv", dtype=MOVIE_DTYPES),
        "links": pd.read_csv(data_path / "links.csv", dtype=LINK_DTYPES),
        "tags": pd.read_csv(data_path / "tags.csv", dtype=TAG_DTYPES),
    }


def write_csv_gzip(frame: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame as a compressed CSV, creating its parent directory."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, compression="gzip")


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write a dictionary as formatted UTF-8 JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
