"""Single-person Streamlit interface for collecting MovieLens-style ratings."""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from group_movie_recommender.algorithms.cold_start import select_onboarding_movies  # noqa: E402
from group_movie_recommender.preprocessing.rating_collection import (  # noqa: E402
    build_movielens_ratings,
)
from group_movie_recommender.shared.posters import (  # noqa: E402
    fetch_tmdb_poster_url,
    load_poster_cache,
)


MOVIES_PATH = PROJECT_ROOT / "dataset" / "movie_lens32m" / "movies.csv"
LINKS_PATH = PROJECT_ROOT / "dataset" / "movie_lens32m" / "links.csv"
WARM_CATALOG_PATH = (
    PROJECT_ROOT / "outputs" / "processed_movielens32m" / "warm_movies.csv.gz"
)
POSTER_CACHE_PATH = PROJECT_ROOT / "assets" / "poster_urls.json"
POSTER_PLACEHOLDER_PATH = PROJECT_ROOT / "assets" / "poster_placeholder.svg"
TMDB_LOGO_URL = (
    "https://www.themoviedb.org/assets/v4/logos/v2/blue_short-"
    "8e7b30f73a4020692ccca9c88bafe5dcb6f8a62a4c6bc55cd9ba82bb2cd95f6c.svg"
)
RATING_OPTIONS: list[str | float] = ["Not seen"] + [value / 2 for value in range(1, 11)]
PAGE_SIZE = 12
DEFAULT_USER_ID = 1_000_000_001


@st.cache_data(show_spinner=False)
def load_catalogues() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load titles and build a unique queue that can continue for many batches."""

    movies = pd.read_csv(MOVIES_PATH, usecols=["movieId", "title", "genres"])
    links = pd.read_csv(LINKS_PATH, usecols=["movieId", "tmdbId"], dtype={"tmdbId": "Int64"})
    movies = movies.merge(links, on="movieId", how="left", validate="one_to_one")
    warm = pd.read_csv(
        WARM_CATALOG_PATH, usecols=["movieId", "trainPositiveCount"]
    )
    starter = select_onboarding_movies(movies, warm, n=36).merge(
        links, on="movieId", how="left", validate="one_to_one"
    )
    popular = (
        warm.merge(movies, on="movieId", how="inner", validate="one_to_one")
        .sort_values(["trainPositiveCount", "movieId"], ascending=[False, True])
        .reset_index(drop=True)
    )
    queue = pd.concat([starter, popular], ignore_index=True).drop_duplicates(
        "movieId", keep="first"
    )
    return movies, queue.reset_index(drop=True)


@st.cache_data(ttl=86_400, show_spinner=False)
def poster_url(movie_id: int, tmdb_id: int | None) -> str | None:
    """Use the local starter cache, then look up searched movies on demand."""

    cached = load_poster_cache(POSTER_CACHE_PATH)
    if int(movie_id) in cached:
        return cached[int(movie_id)]
    if tmdb_id is None or pd.isna(tmdb_id):
        return None
    return fetch_tmdb_poster_url(int(tmdb_id))


@st.cache_data(ttl=86_400, show_spinner=False)
def poster_urls_for_batch(
    items: tuple[tuple[int, int | None], ...],
) -> dict[int, str | None]:
    """Resolve one visible batch in parallel instead of one slow request at a time."""

    cached = load_poster_cache(POSTER_CACHE_PATH)
    result: dict[int, str | None] = {
        movie_id: cached[movie_id]
        for movie_id, _ in items
        if movie_id in cached
    }
    missing = [
        (movie_id, tmdb_id)
        for movie_id, tmdb_id in items
        if movie_id not in result and tmdb_id is not None
    ]
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(missing)))) as pool:
        futures = {
            pool.submit(fetch_tmdb_poster_url, tmdb_id): movie_id
            for movie_id, tmdb_id in missing
        }
        for future in as_completed(futures):
            result[futures[future]] = future.result()
    for movie_id, _ in items:
        result.setdefault(movie_id, None)
    return result


def rating_value(movie_id: int) -> str | float:
    record = st.session_state["ratings"].get(int(movie_id))
    return float(record["rating"]) if record else "Not seen"


def set_rating(movie_id: int, value: str | float) -> None:
    """Save one rating and its collection time; unseen titles remain absent."""

    movie_id = int(movie_id)
    records = st.session_state["ratings"]
    if value == "Not seen":
        records.pop(movie_id, None)
        return
    numeric_value = float(value)
    previous = records.get(movie_id)
    if previous is None or float(previous["rating"]) != numeric_value:
        records[movie_id] = {"rating": numeric_value, "timestamp": int(time.time())}


def ratings_frame() -> pd.DataFrame:
    """Return the current session ratings without adding a participant ID."""

    return build_movielens_ratings(
        st.session_state["ratings"], user_id=DEFAULT_USER_ID
    ).drop(columns="userId")


def export_ratings_csv(user_id: int) -> bytes:
    """Export exact MovieLens rating columns for one participant."""

    frame = build_movielens_ratings(st.session_state["ratings"], user_id=user_id)
    return frame.to_csv(index=False).encode("utf-8")


def render_movie_card(
    row: object,
    *,
    key_prefix: str,
    image_url: str | None,
) -> None:
    movie_id = int(getattr(row, "movieId"))
    image = image_url or str(POSTER_PLACEHOLDER_PATH)
    st.image(image, caption=str(getattr(row, "title")), width="stretch")
    st.caption(str(getattr(row, "genres")).replace("|", " · "))
    selected = st.select_slider(
        f"Rating for {getattr(row, 'title')}",
        options=RATING_OPTIONS,
        value=rating_value(movie_id),
        key=f"{key_prefix}_{movie_id}",
        label_visibility="collapsed",
        format_func=lambda choice: choice if choice == "Not seen" else f"{choice:g} ★",
    )
    set_rating(movie_id, selected)


def render_search(movies: pd.DataFrame) -> None:
    st.subheader("Search another movie")
    query = st.text_input(
        "Movie title",
        placeholder="Try Alien, Parasite, or The Godfather",
    ).strip()
    if len(query) < 2:
        st.caption("Type at least two characters to search all MovieLens 32M titles.")
        return

    matches = movies.loc[
        movies["title"].str.contains(query, case=False, regex=False, na=False)
    ].head(50)
    if matches.empty:
        st.info("No matching MovieLens title found.")
        return

    title_by_id = matches.set_index("movieId")["title"].to_dict()
    selected_id = int(
        st.selectbox(
            "Choose a movie",
            options=matches["movieId"].astype(int).tolist(),
            format_func=lambda movie_id: title_by_id[movie_id],
        )
    )
    selected = matches.loc[matches["movieId"].eq(selected_id)].iloc[0]
    poster_column, rating_column = st.columns([1, 2])
    with poster_column:
        raw_tmdb_id = selected["tmdbId"]
        tmdb_id = None if pd.isna(raw_tmdb_id) else int(raw_tmdb_id)
        image = poster_url(selected_id, tmdb_id) or str(POSTER_PLACEHOLDER_PATH)
        st.image(image, caption=str(selected["title"]), width="stretch")
    with rating_column:
        st.write(f"**{selected['title']}**")
        st.caption(str(selected["genres"]).replace("|", " · "))
        searched_rating = st.select_slider(
            "Your rating",
            options=RATING_OPTIONS,
            value=rating_value(selected_id),
            key=f"search_rating_{selected_id}",
            format_func=lambda choice: (
                choice if choice == "Not seen" else f"{choice:g} ★"
            ),
        )
        set_rating(selected_id, searched_rating)


def render_saved_ratings(movies: pd.DataFrame) -> None:
    count = len(st.session_state["ratings"])
    st.subheader(f"Saved ratings ({count})")
    if not count:
        st.info("No ratings yet. Unseen movies are skipped and do not become dislikes.")
        return
    saved = ratings_frame().merge(
        movies.loc[:, ["movieId", "title"]], on="movieId", how="left"
    )
    saved["ratedAt"] = pd.to_datetime(saved["timestamp"], unit="s", utc=True)
    st.dataframe(
        saved.loc[:, ["movieId", "title", "rating", "ratedAt"]],
        hide_index=True,
        width="stretch",
    )
    if st.button("Clear all ratings"):
        st.session_state["ratings"] = {}
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Movie Rating Collector", page_icon="🎬", layout="wide")
    st.title("🎬 Rate movies you have seen")
    st.write(
        "This form collects one person's ratings for the recommendation dataset. "
        "Skip unfamiliar movies instead of guessing."
    )

    required_paths = [MOVIES_PATH, LINKS_PATH, WARM_CATALOG_PATH]
    if any(not path.exists() for path in required_paths):
        st.error("MovieLens data or the processed warm catalogue is missing.")
        st.stop()

    movies, rating_queue = load_catalogues()
    if not isinstance(st.session_state.get("ratings"), dict) or set(
        st.session_state.get("ratings", {})
    ).intersection({"a", "b"}):
        st.session_state["ratings"] = {}

    st.subheader("Movies to rate")
    st.caption(
        "Continue for as many batches as you want. Every batch contains 12 new movies; "
        "stop and download whenever you are ready."
    )
    batch_count = (len(rating_queue) + PAGE_SIZE - 1) // PAGE_SIZE
    st.session_state.setdefault("batch_index", 0)
    batch_index = min(int(st.session_state["batch_index"]), batch_count - 1)
    st.session_state["batch_index"] = batch_index

    previous_column, status_column, next_column = st.columns([1, 2, 1])
    with previous_column:
        if st.button("← Previous 12", disabled=batch_index == 0, width="stretch"):
            st.session_state["batch_index"] = batch_index - 1
            st.rerun()
    with status_column:
        start = batch_index * PAGE_SIZE + 1
        end = min((batch_index + 1) * PAGE_SIZE, len(rating_queue))
        st.markdown(
            f"<p style='text-align:center'><strong>Movies {start}–{end}</strong> "
            f"of {len(rating_queue):,}</p>",
            unsafe_allow_html=True,
        )
    with next_column:
        if st.button(
            "Next 12 →",
            disabled=batch_index == batch_count - 1,
            width="stretch",
            key="next_top",
        ):
            st.session_state["batch_index"] = batch_index + 1
            st.rerun()

    page_movies = rating_queue.iloc[
        batch_index * PAGE_SIZE : (batch_index + 1) * PAGE_SIZE
    ]
    poster_items = tuple(
        (
            int(row.movieId),
            None if pd.isna(row.tmdbId) else int(row.tmdbId),
        )
        for row in page_movies.itertuples(index=False)
    )
    with st.spinner("Loading movie posters…"):
        batch_posters = poster_urls_for_batch(poster_items)
    columns = st.columns(4)
    for position, row in enumerate(page_movies.itertuples(index=False)):
        with columns[position % 4]:
            with st.container(border=True):
                render_movie_card(
                    row,
                    key_prefix="queue",
                    image_url=batch_posters.get(int(row.movieId)),
                )

    if st.button(
        "Show another 12 movies →",
        disabled=batch_index == batch_count - 1,
        type="primary",
        width="stretch",
        key="next_bottom",
    ):
        st.session_state["batch_index"] = batch_index + 1
        st.rerun()

    st.divider()
    render_search(movies)
    st.divider()
    render_saved_ratings(movies)

    # Render the sidebar after rating widgets so its count and CSV reflect the
    # value selected in the current Streamlit rerun.
    with st.sidebar:
        st.header("Participant")
        user_id = int(
            st.number_input(
                "Participant ID",
                min_value=200_949,
                max_value=2_147_483_647,
                value=DEFAULT_USER_ID,
                step=1,
                help="Assign a different positive integer to each participant.",
            )
        )
        count = len(st.session_state["ratings"])
        st.metric("Movies rated", count)
        st.download_button(
            "Download ratings CSV",
            data=export_ratings_csv(user_id),
            file_name=f"movie_ratings_user_{user_id}.csv",
            mime="text/csv",
            disabled=count == 0,
            width="stretch",
        )

    st.download_button(
        "I'm finished — download my ratings CSV",
        data=export_ratings_csv(user_id),
        file_name=f"movie_ratings_user_{user_id}.csv",
        mime="text/csv",
        disabled=count == 0,
        type="primary",
        width="stretch",
        key="download_finished",
    )

    with st.expander("Poster credits"):
        st.image(TMDB_LOGO_URL, width=120)
        st.markdown(
            "Poster images come from [TMDB](https://www.themoviedb.org). "
            "This product uses TMDB and the TMDB APIs but is not endorsed, "
            "certified, or otherwise approved by TMDB."
        )


if __name__ == "__main__":
    main()
