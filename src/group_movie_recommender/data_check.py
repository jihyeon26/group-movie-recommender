"""Data-quality checks for the MovieLens source tables."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = {
    "ratings": ["userId", "movieId", "rating", "timestamp"],
    "movies": ["movieId", "title", "genres"],
    "links": ["movieId", "imdbId", "tmdbId"],
    "tags": ["userId", "movieId", "tag", "timestamp"],
}


def clean_tags(tags: pd.DataFrame) -> pd.DataFrame:
    """Remove rows without usable tag text and normalize surrounding whitespace."""

    cleaned = tags.dropna(subset=["tag"]).copy()
    cleaned["tag"] = cleaned["tag"].str.strip()
    return cleaned.loc[cleaned["tag"].ne("")].reset_index(drop=True)


def _missing_report(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    report: dict[str, dict[str, float | int]] = {}
    for column, count in frame.isna().sum().items():
        report[str(column)] = {
            "count": int(count),
            "fraction": float(count / len(frame)) if len(frame) else 0.0,
        }
    return report


def validate_tables(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Return a JSON-serializable validation report for all MovieLens tables."""

    absent = sorted(set(EXPECTED_COLUMNS) - set(tables))
    if absent:
        raise KeyError(f"Missing tables: {absent}")

    ratings = tables["ratings"]
    movies = tables["movies"]
    links = tables["links"]
    tags = tables["tags"]

    schema = {
        name: {
            "matches": frame.columns.tolist() == EXPECTED_COLUMNS[name],
            "actual": frame.columns.tolist(),
            "expected": EXPECTED_COLUMNS[name],
        }
        for name, frame in tables.items()
    }

    valid_ratings = np.arange(0.5, 5.01, 0.5)
    report: dict[str, Any] = {
        "shape": {
            name: {"rows": int(len(frame)), "columns": int(frame.shape[1])}
            for name, frame in tables.items()
        },
        "schema": schema,
        "missing": {name: _missing_report(frame) for name, frame in tables.items()},
        "duplicates": {
            "ratings_user_movie": int(ratings.duplicated(["userId", "movieId"]).sum()),
            "movies_movie_id": int(movies.duplicated(["movieId"]).sum()),
            "links_movie_id": int(links.duplicated(["movieId"]).sum()),
            "tags_exact": int(tags.duplicated().sum()),
        },
        "value_checks": {
            "invalid_rating_rows": int((~ratings["rating"].isin(valid_ratings)).sum()),
            "non_positive_user_ids": int((ratings["userId"] <= 0).sum()),
            "non_positive_movie_ids": int((ratings["movieId"] <= 0).sum()),
        },
        "referential_integrity": {
            "rating_rows_with_unknown_movie": int((~ratings["movieId"].isin(movies["movieId"])).sum()),
            "rating_rows_without_link": int((~ratings["movieId"].isin(links["movieId"])).sum()),
            "tag_rows_with_unknown_movie": int((~tags["movieId"].isin(movies["movieId"])).sum()),
        },
        "time_range_utc": {
            "ratings_start": pd.to_datetime(ratings["timestamp"].min(), unit="s", utc=True).isoformat(),
            "ratings_end": pd.to_datetime(ratings["timestamp"].max(), unit="s", utc=True).isoformat(),
            "tags_start": pd.to_datetime(tags["timestamp"].min(), unit="s", utc=True).isoformat(),
            "tags_end": pd.to_datetime(tags["timestamp"].max(), unit="s", utc=True).isoformat(),
        },
        "tag_cleaning": {
            "input_rows": int(len(tags)),
            "retained_rows": int(len(clean_tags(tags))),
        },
    }
    report["passed_core_checks"] = bool(
        all(item["matches"] for item in schema.values())
        and report["value_checks"]["invalid_rating_rows"] == 0
        and report["referential_integrity"]["rating_rows_with_unknown_movie"] == 0
    )
    return report


def validation_summary(report: dict[str, Any]) -> pd.DataFrame:
    """Flatten the main validation outcomes for console display."""

    rows = [
        {"check": "core_checks", "value": report["passed_core_checks"]},
        {"check": "rating_rows", "value": report["shape"]["ratings"]["rows"]},
        {"check": "movie_rows", "value": report["shape"]["movies"]["rows"]},
        {"check": "invalid_ratings", "value": report["value_checks"]["invalid_rating_rows"]},
        {
            "check": "unknown_rating_movies",
            "value": report["referential_integrity"]["rating_rows_with_unknown_movie"],
        },
        {"check": "missing_tmdb_ids", "value": report["missing"]["links"]["tmdbId"]["count"]},
        {"check": "missing_tag_text", "value": report["missing"]["tags"]["tag"]["count"]},
    ]
    return pd.DataFrame(rows)

