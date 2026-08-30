"""Tests for graph indexing and BPR negative sampling."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from group_movie_recommender.algorithms.graph_data import (
    BPRBatchSampler,
    build_bipartite_graph,
    positive_movie_sets,
    sample_bpr_batch,
)


class GraphDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.edges = pd.DataFrame(
            {
                "userId": [20, 10, 10, 20, 30],
                "movieId": [200, 100, 200, 300, 100],
            }
        )

    def test_ids_become_contiguous_bipartite_nodes(self) -> None:
        graph = build_bipartite_graph(self.edges)

        self.assertEqual(graph.user_ids.tolist(), [10, 20, 30])
        self.assertEqual(graph.movie_ids.tolist(), [100, 200, 300])
        self.assertEqual(graph.num_nodes, 6)
        self.assertEqual(graph.num_positive_edges, 5)
        self.assertEqual(graph.edge_index.shape, (2, 10))
        self.assertTrue((graph.edge_index[1, :5] >= graph.num_users).all())
        np.testing.assert_array_equal(
            graph.edge_index[:, 5:],
            graph.edge_index[::-1, :5],
        )

    def test_negative_movies_are_unseen_and_reproducible(self) -> None:
        graph = build_bipartite_graph(self.edges)
        first = sample_bpr_batch(graph, batch_size=20, random_seed=7)
        second = sample_bpr_batch(graph, batch_size=20, random_seed=7)
        positives = positive_movie_sets(graph)

        pd.testing.assert_frame_equal(first, second)
        for row in first.itertuples(index=False):
            self.assertNotIn(
                int(row.negativeMovieIndex),
                positives[int(row.userIndex)],
            )

    def test_duplicate_edges_are_removed(self) -> None:
        duplicated = pd.concat([self.edges, self.edges.iloc[[0]]], ignore_index=True)
        graph = build_bipartite_graph(duplicated)
        self.assertEqual(graph.num_positive_edges, 5)

    def test_training_sampler_advances_reproducibly(self) -> None:
        graph = build_bipartite_graph(self.edges)
        first_sampler = BPRBatchSampler(graph, random_seed=19)
        second_sampler = BPRBatchSampler(graph, random_seed=19)
        first_sequence = [first_sampler.sample(batch_size=8) for _ in range(2)]
        second_sequence = [second_sampler.sample(batch_size=8) for _ in range(2)]

        pd.testing.assert_frame_equal(first_sequence[0], second_sequence[0])
        pd.testing.assert_frame_equal(first_sequence[1], second_sequence[1])


if __name__ == "__main__":
    unittest.main()
