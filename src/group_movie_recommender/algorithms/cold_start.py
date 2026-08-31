"""Cold-start profiles and recommendations for new two-person groups.

New visitors do not have nodes in a trained collaborative-filtering graph.  This
module therefore turns a small number of explicit MovieLens-style ratings into a
shrunk genre profile and falls back smoothly to training-only popularity.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .group_ranking import normalize_member_scores, rank_group_candidates


DEFAULT_ONBOARDING_GENRES = (
    "Drama",
    "Comedy",
    "Action",
    "Thriller",
    "Adventure",
    "Crime",
    "Sci-Fi",
    "Romance",
    "Animation",
    "Children",
    "Fantasy",
    "Mystery",
    "Documentary",
    "Horror",
    "Musical",
    "War",
    "Western",
    "Film-Noir",
)


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _movie_catalog(movies: pd.DataFrame, warm_catalog: pd.DataFrame) -> pd.DataFrame:
    _require_columns(movies, {"movieId", "title", "genres"}, "Movie catalogue")
    _require_columns(warm_catalog, {"movieId", "trainPositiveCount"}, "Warm catalogue")
    if movies["movieId"].duplicated().any() or warm_catalog["movieId"].duplicated().any():
        raise ValueError("movieId must be unique in both catalogues")

    catalog = warm_catalog.loc[:, ["movieId", "trainPositiveCount"]].merge(
        movies.loc[:, ["movieId", "title", "genres"]],
        on="movieId",
        how="inner",
        validate="one_to_one",
    )
    catalog = catalog.loc[catalog["trainPositiveCount"] >= 0].copy()
    if catalog.empty:
        raise ValueError("The warm and movie catalogues have no valid movies in common")
    catalog["genres"] = catalog["genres"].fillna("(no genres listed)").astype(str)
    return catalog.sort_values(
        ["trainPositiveCount", "movieId"], ascending=[False, True]
    ).reset_index(drop=True)


def select_onboarding_movies(
    movies: pd.DataFrame,
    warm_catalog: pd.DataFrame,
    *,
    n: int = 36,
    popular_pool_size: int = 750,
    genres: Iterable[str] = DEFAULT_ONBOARDING_GENRES,
) -> pd.DataFrame:
    """Choose recognizable, genre-diverse movies for the rating screen.

    Only the most popular warm items are considered.  The selector repeatedly
    takes the most popular still-unused title for each broad genre, then fills any
    remaining positions by popularity.  It never treats an unseen movie as a
    dislike.
    """

    if n <= 0 or popular_pool_size <= 0:
        raise ValueError("n and popular_pool_size must be positive")

    catalog = _movie_catalog(movies, warm_catalog)
    pool = catalog.head(max(n, popular_pool_size)).copy()
    selected: list[int] = []
    selected_set: set[int] = set()
    genre_list = tuple(dict.fromkeys(str(genre) for genre in genres))

    while len(selected) < n:
        added_this_round = False
        for genre in genre_list:
            matches = pool.loc[
                pool["genres"].str.split("|").map(lambda values: genre in values)
                & ~pool["movieId"].isin(selected_set)
            ]
            if matches.empty:
                continue
            movie_id = int(matches.iloc[0]["movieId"])
            selected.append(movie_id)
            selected_set.add(movie_id)
            added_this_round = True
            if len(selected) == n:
                break
        if not added_this_round:
            break

    if len(selected) < n:
        for movie_id in pool.loc[~pool["movieId"].isin(selected_set), "movieId"]:
            selected.append(int(movie_id))
            if len(selected) == n:
                break

    by_id = pool.set_index("movieId")
    result = by_id.loc[selected].reset_index()
    result.insert(0, "onboardingOrder", np.arange(1, len(result) + 1, dtype=np.int32))
    return result


def _validated_ratings(ratings: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    _require_columns(ratings, {"movieId", "rating"}, "Ratings")
    if ratings.empty:
        return pd.DataFrame(columns=["movieId", "rating", "genres"])
    if ratings["movieId"].duplicated().any():
        raise ValueError("A member can have only one rating per movieId")

    values = ratings["rating"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0.5) | (values > 5.0)).any():
        raise ValueError("Ratings must be finite and between 0.5 and 5.0")
    doubled = values * 2
    if not np.allclose(doubled, np.round(doubled)):
        raise ValueError("Ratings must use MovieLens half-star increments")

    known = ratings.loc[:, ["movieId", "rating"]].merge(
        catalog.loc[:, ["movieId", "genres"]],
        on="movieId",
        how="inner",
        validate="one_to_one",
    )
    if len(known) != len(ratings):
        unknown = sorted(set(ratings["movieId"]) - set(known["movieId"]))
        raise ValueError(f"Ratings contain movieIds outside the warm catalogue: {unknown[:5]}")
    return known


def score_cold_start_candidates(
    movies: pd.DataFrame,
    warm_catalog: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    candidate_pool_size: int = 2_000,
    profile_prior: float = 2.0,
    confidence_prior: float = 5.0,
) -> pd.DataFrame:
    """Score warm candidates using a sparse genre profile and popularity fallback."""

    if candidate_pool_size <= 0:
        raise ValueError("candidate_pool_size must be positive")
    if profile_prior < 0 or confidence_prior <= 0:
        raise ValueError("profile_prior must be non-negative and confidence_prior positive")

    catalog = _movie_catalog(movies, warm_catalog)
    known = _validated_ratings(ratings, catalog)
    rated_ids = set(known["movieId"].astype(int))
    candidates = catalog.loc[~catalog["movieId"].isin(rated_ids)].head(candidate_pool_size).copy()
    if candidates.empty:
        raise ValueError("No unrated warm movies remain in the candidate pool")

    # Percentile ranks keep the scale stable if a smaller catalogue is supplied.
    candidates["popularityScore"] = candidates["trainPositiveCount"].rank(
        method="average", pct=True
    )
    candidates["tasteScore"] = 0.5

    if len(known):
        profile_rows: list[tuple[str, float]] = []
        for row in known.itertuples(index=False):
            row_genres = str(row.genres).split("|")
            if row_genres == ["(no genres listed)"]:
                continue
            deviation = float(row.rating) - 3.0
            profile_rows.extend((genre, deviation) for genre in row_genres)

        if profile_rows:
            profile = pd.DataFrame(profile_rows, columns=["genre", "deviation"])
            aggregate = profile.groupby("genre")["deviation"].agg(["sum", "count"])
            preference = (aggregate["sum"] / (aggregate["count"] + profile_prior)).to_dict()

            def taste_score(value: str) -> float:
                item_genres = [
                    genre for genre in str(value).split("|") if genre != "(no genres listed)"
                ]
                if not item_genres:
                    return 0.5
                effects = [float(preference.get(genre, 0.0)) for genre in item_genres]
                return float(np.clip(0.5 + np.mean(effects) / 2.0, 0.0, 1.0))

            candidates["tasteScore"] = candidates["genres"].map(taste_score)

    rating_count = len(known)
    confidence = rating_count / (rating_count + confidence_prior)
    # Retain some popularity even for established profiles; it makes a tiny
    # profile less sensitive to one unusual rating.
    taste_weight = 0.8 * confidence
    candidates["profileConfidence"] = confidence
    candidates["score"] = (
        taste_weight * candidates["tasteScore"]
        + (1.0 - taste_weight) * candidates["popularityScore"]
    )
    return candidates.reset_index(drop=True)


def recommend_cold_start_group(
    movies: pd.DataFrame,
    warm_catalog: pd.DataFrame,
    ratings_a: pd.DataFrame,
    ratings_b: pd.DataFrame,
    *,
    conflict_weight: float = 0.65,
    k: int = 10,
    candidate_pool_size: int = 2_000,
    diversity_weight: float = 0.0,
    diversity_pool_size: int = 100,
) -> pd.DataFrame:
    """Return one conflict-aware list for two new or sparsely rated members."""

    if not 0.0 <= diversity_weight <= 1.0:
        raise ValueError("diversity_weight must be between zero and one")
    if diversity_pool_size < k:
        raise ValueError("diversity_pool_size must be at least k")

    catalog = _movie_catalog(movies, warm_catalog)
    all_rated = set(ratings_a.get("movieId", pd.Series(dtype=int)).astype(int)) | set(
        ratings_b.get("movieId", pd.Series(dtype=int)).astype(int)
    )
    # Score against the complete catalogue so each member's rated titles can be
    # used to build their profile.  Ask for a few extra rows because rated movies
    # are removed before the shared ranking.
    expanded_pool_size = candidate_pool_size + len(all_rated)
    scores_a = score_cold_start_candidates(
        movies,
        warm_catalog,
        ratings_a,
        candidate_pool_size=expanded_pool_size,
    )
    scores_b = score_cold_start_candidates(
        movies,
        warm_catalog,
        ratings_b,
        candidate_pool_size=expanded_pool_size,
    )
    aligned = scores_a.loc[:, ["movieId", "score"]].rename(columns={"score": "scoreA"}).merge(
        scores_b.loc[:, ["movieId", "score"]].rename(columns={"score": "scoreB"}),
        on="movieId",
        how="inner",
        validate="one_to_one",
    )
    aligned = aligned.loc[~aligned["movieId"].isin(all_rated)].head(candidate_pool_size)
    normalized = normalize_member_scores(aligned)
    ranking_size = diversity_pool_size if diversity_weight > 0 else k
    ranked = rank_group_candidates(
        normalized,
        conflict_weight=conflict_weight,
        k=ranking_size,
    )
    details = catalog.loc[:, ["movieId", "title", "genres", "trainPositiveCount"]]
    ranked = ranked.merge(details, on="movieId", how="left", validate="one_to_one")
    if diversity_weight > 0:
        return diversify_group_ranking(
            ranked,
            k=k,
            diversity_weight=diversity_weight,
        )
    return ranked


def diversify_group_ranking(
    candidates: pd.DataFrame,
    *,
    k: int,
    diversity_weight: float,
) -> pd.DataFrame:
    """Greedily reward genre novelty while retaining the group score."""

    required = {"movieId", "genres", "groupScore", "minimumScore", "averageScore"}
    _require_columns(candidates, required, "Ranked candidates")
    if k <= 0 or len(candidates) < k:
        raise ValueError("k must be positive and no larger than the candidate pool")
    if not 0.0 <= diversity_weight <= 1.0:
        raise ValueError("diversity_weight must be between zero and one")

    remaining = candidates.copy().reset_index(drop=True)
    remaining["baseRank"] = np.arange(1, len(remaining) + 1, dtype=np.int32)
    selected: list[pd.Series] = []
    selected_genres: list[set[str]] = []
    while len(selected) < k:
        best_index: int | None = None
        best_key: tuple[float, float, float, int] | None = None
        best_bonus = 0.0
        for index, row in remaining.iterrows():
            genres = _genre_set(str(row["genres"]))
            maximum_overlap = max(
                (_genre_jaccard(genres, prior) for prior in selected_genres),
                default=0.0,
            )
            bonus = 1.0 - maximum_overlap if selected_genres else 0.0
            diversified_score = float(row["groupScore"]) + diversity_weight * bonus
            key = (
                diversified_score,
                float(row["minimumScore"]),
                float(row["averageScore"]),
                -int(row["movieId"]),
            )
            if best_key is None or key > best_key:
                best_index = int(index)
                best_key = key
                best_bonus = bonus
        assert best_index is not None and best_key is not None
        chosen = remaining.loc[best_index].copy()
        chosen["diversityBonus"] = best_bonus
        chosen["diversifiedScore"] = best_key[0]
        selected.append(chosen)
        selected_genres.append(_genre_set(str(chosen["genres"])))
        remaining = remaining.drop(index=best_index).reset_index(drop=True)

    result = pd.DataFrame(selected).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1, dtype=np.int32)
    ordered = ["rank", "baseRank", *[column for column in result.columns if column not in {"rank", "baseRank"}]]
    return result.loc[:, ordered]


def _genre_set(value: str) -> set[str]:
    return {
        genre
        for genre in value.split("|")
        if genre and genre != "(no genres listed)"
    }


def _genre_jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0
