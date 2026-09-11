"""Tests for PyTorch LightGCN propagation, gradients, and toy learning."""

from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from group_movie_recommender.algorithms.lightgcn import LightGCN
    from group_movie_recommender.algorithms.lightgcn_math import bpr_loss, lightgcn_propagate
    from group_movie_recommender.pipelines.lightgcn_training import (
        LightGCNTrainingConfig,
        build_toy_training_graph,
        train_small_graph,
    )


@unittest.skipIf(torch is None, "Install the training extra to test PyTorch LightGCN")
class LightGCNTests(unittest.TestCase):
    def test_torch_propagation_matches_numpy_reference(self) -> None:
        graph = build_toy_training_graph()
        rng = np.random.default_rng(11)
        initial = rng.normal(size=(graph.num_nodes, 4)).astype(np.float32)
        expected = lightgcn_propagate(initial, graph.edge_index, num_layers=2)
        model = LightGCN(graph, embedding_dim=4, num_layers=2, random_seed=5)
        with torch.no_grad():
            model.embedding.weight.copy_(torch.from_numpy(initial))
            users, movies = model.propagate()
            actual = torch.cat([users, movies]).numpy()
        np.testing.assert_allclose(actual, expected.final_embeddings, rtol=1e-5, atol=1e-6)

    def test_bpr_objective_produces_embedding_gradients(self) -> None:
        graph = build_toy_training_graph()
        model = LightGCN(graph, embedding_dim=4, num_layers=1, random_seed=7)
        indices = (
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([0, 2], dtype=torch.long),
            torch.tensor([3, 0], dtype=torch.long),
        )
        losses = model.bpr_objective(*indices)
        losses["loss"].backward()
        gradient = model.embedding.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertGreater(float(gradient.abs().sum().item()), 0.0)

    def test_fixed_training_loss_decreases(self) -> None:
        graph = build_toy_training_graph()
        config = LightGCNTrainingConfig(steps=60, batch_size=32, random_seed=7)
        _, history = train_small_graph(graph, config)
        initial = float(history.iloc[0]["fixed_training_bpr_loss"])
        final = float(history.iloc[-1]["fixed_training_bpr_loss"])
        self.assertLess(final, initial)

    def test_bpr_ranking_loss_matches_numpy(self) -> None:
        graph = build_toy_training_graph()
        model = LightGCN(graph, embedding_dim=4, num_layers=2, random_seed=7)
        users = torch.tensor([0, 1], dtype=torch.long)
        positives = torch.tensor([0, 2], dtype=torch.long)
        negatives = torch.tensor([3, 0], dtype=torch.long)
        with torch.no_grad():
            expected = bpr_loss(model(users, positives).numpy(), model(users, negatives).numpy())
            actual = model.bpr_objective(users, positives, negatives, l2_weight=0.0)
        self.assertAlmostEqual(float(actual["loss"]), expected, places=6)

    def test_validation_restores_best_checkpoint_and_stops_early(self) -> None:
        graph = build_toy_training_graph()
        snapshots = {}

        def validate(model, step):
            snapshots[step] = model.embedding.weight.detach().clone()
            return ({0: 0.0, 5: 0.5, 10: 0.4, 15: 0.3}[step], 0.0)

        model, history = train_small_graph(
            graph,
            LightGCNTrainingConfig(steps=20, batch_size=16, random_seed=7),
            validation_callback=validate,
            validation_interval=5,
            patience=2,
        )

        self.assertEqual(history.attrs["best_step"], 5)
        self.assertEqual(history.attrs["stopped_step"], 15)
        self.assertTrue(history.attrs["early_stopped"])
        torch.testing.assert_close(model.embedding.weight, snapshots[5])

    def test_smoothing_ignores_a_single_validation_spike(self) -> None:
        """A noisy primary metric must not end training on one lucky estimate."""

        graph = build_toy_training_graph()
        # Step 5 is an isolated spike; the underlying signal keeps improving.
        scores = {0: 0.0, 5: 0.5, 10: 0.3, 15: 0.35, 20: 0.4}

        def run(selection_smoothing):
            return train_small_graph(
                graph,
                LightGCNTrainingConfig(steps=20, batch_size=16, random_seed=7),
                validation_callback=lambda model, step: (scores[step], 0.0),
                validation_interval=5,
                patience=2,
                selection_smoothing=selection_smoothing,
            )[1]

        unsmoothed = run(1)
        self.assertEqual(unsmoothed.attrs["best_step"], 5)
        self.assertTrue(unsmoothed.attrs["early_stopped"])

        smoothed = run(3)
        self.assertEqual(smoothed.attrs["best_step"], 15)
        self.assertFalse(smoothed.attrs["early_stopped"])
        self.assertEqual(smoothed.attrs["selection_smoothing"], 3)

    def test_selection_smoothing_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            train_small_graph(
                build_toy_training_graph(),
                LightGCNTrainingConfig(steps=5, batch_size=4, random_seed=7),
                selection_smoothing=0,
            )


if __name__ == "__main__":
    unittest.main()
