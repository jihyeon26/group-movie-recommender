"""Poster URL extraction for MovieLens titles linked to TMDB movies."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TMDB_MOVIE_URL = "https://www.themoviedb.org/movie/{tmdb_id}"


class _OpenGraphImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "meta":
            return
        values = {name.lower(): value for name, value in attrs if value is not None}
        if values.get("property") == "og:image" and values.get("content"):
            self.images.append(str(values["content"]))


def extract_tmdb_poster_url(page_html: str) -> str | None:
    """Return the first TMDB Open Graph image, which is the movie poster."""

    parser = _OpenGraphImageParser()
    parser.feed(page_html)
    return parser.images[0] if parser.images else None


def fetch_tmdb_poster_url(
    tmdb_id: int,
    *,
    timeout: float = 8.0,
) -> str | None:
    """Fetch one public TMDB movie page and return its poster URL."""

    if int(tmdb_id) <= 0:
        return None
    try:
        request = Request(
            TMDB_MOVIE_URL.format(tmdb_id=int(tmdb_id)),
            headers={"User-Agent": "group-movie-recommender/0.1 poster lookup"},
        )
        with urlopen(request, timeout=timeout) as response:
            page_html = response.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, OSError):
        return None
    return extract_tmdb_poster_url(page_html)


def load_poster_cache(path: str | Path) -> dict[int, str]:
    """Load a checked-in movieId-to-poster-URL cache."""

    cache_path = Path(path)
    if not cache_path.is_file():
        return {}
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    return {
        int(movie_id): str(url)
        for movie_id, url in payload.items()
        if str(movie_id).isdigit() and isinstance(url, str) and url.startswith("https://")
    }
