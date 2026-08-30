"""Tests for deterministic graph-subset construction."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from group_movie_recommender.preprocessing.graph_subset import build_controlled_graph_subset


class GraphSubsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.edges = pd.DataFrame({
            "userId": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "movieId": [10, 20, 20, 30, 10, 40, 30, 50, 40, 50],
        })
        self.pairs = pd.DataFrame({"userA": [1, 2], "userB": [2, 3]})

    def test_focus_users_are_always_retained(self) -> None:
        subset = build_controlled_graph_subset(
            self.edges,
            self.pairs,
            pair_limit=1,
            total_user_limit=3,
            random_seed=7,
        )
        self.assertEqual(subset.focus_user_ids.tolist(), [1, 2])
        self.assertTrue(np.isin([1, 2], subset.graph.user_ids).all())
        self.assertEqual(subset.graph.num_users, 3)

    def test_context_sampling_is_reproducible(self) -> None:
        kwargs = dict(pair_limit=1, total_user_limit=4, random_seed=19)
        first = build_controlled_graph_subset(self.edges, self.pairs, **kwargs)
        second = build_controlled_graph_subset(self.edges, self.pairs, **kwargs)
        np.testing.assert_array_equal(first.context_user_ids, second.context_user_ids)
        pd.testing.assert_frame_equal(first.positive_edges, second.positive_edges)

    def test_user_limit_cannot_drop_pair_members(self) -> None:
        with self.assertRaises(ValueError):
            build_controlled_graph_subset(
                self.edges,
                self.pairs,
                pair_limit=2,
                total_user_limit=2,
                random_seed=7,
            )


if __name__ == "__main__":
    unittest.main()
