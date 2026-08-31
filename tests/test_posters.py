"""Tests for local poster metadata parsing and caching."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from group_movie_recommender.shared.posters import (
    extract_tmdb_poster_url,
    load_poster_cache,
)


class PosterTests(unittest.TestCase):
    def test_first_open_graph_image_is_the_poster(self) -> None:
        html = """
        <html><head>
          <meta property="og:image" content="https://images.example/poster.jpg">
          <meta property="og:image" content="https://images.example/backdrop.jpg">
        </head></html>
        """
        self.assertEqual(
            extract_tmdb_poster_url(html),
            "https://images.example/poster.jpg",
        )

    def test_missing_open_graph_image_returns_none(self) -> None:
        self.assertIsNone(extract_tmdb_poster_url("<html><head></head></html>"))

    def test_cache_loader_keeps_only_valid_https_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "posters.json"
            path.write_text(
                json.dumps(
                    {
                        "1": "https://images.example/one.jpg",
                        "bad-id": "https://images.example/two.jpg",
                        "3": "http://images.example/three.jpg",
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                load_poster_cache(path),
                {1: "https://images.example/one.jpg"},
            )


if __name__ == "__main__":
    unittest.main()
