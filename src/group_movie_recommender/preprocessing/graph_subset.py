"""Deterministic MovieLens graph subsets for safe integration testing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..algorithms.graph_data import BipartiteGraphData, build_bipartite_graph


@dataclass(frozen=True)
class ControlledGraphSubset:
    """A graph retaining focus-pair users plus uniformly sampled context users."""

    graph: BipartiteGraphData
    pairs: pd.DataFrame
    focus_user_ids: np.ndarray
    context_user_ids: np.ndarray
    positive_edges: pd.DataFrame


def build_controlled_graph_subset(
    train_positive_edges: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    pair_limit: int,
    total_user_limit: int,
    random_seed: int,
) -> ControlledGraphSubset:
    """Retain pair members exactly and sample other training users reproducibly."""

    if pair_limit <= 0 or total_user_limit <= 0:
        raise ValueError("pair_limit and total_user_limit must be positive")
    missing_edges = {"userId", "movieId"} - set(train_positive_edges.columns)
    missing_pairs = {"userA", "userB"} - set(pairs.columns)
    if missing_edges or missing_pairs:
        raise ValueError(
            f"Missing edge columns: {sorted(missing_edges)}; pair columns: {sorted(missing_pairs)}"
        )

    selected_pairs = pairs.head(pair_limit).loc[:, ["userA", "userB"]].copy()
    selected_pairs.insert(0, "pairId", np.arange(len(selected_pairs), dtype=np.int32))
    if selected_pairs.empty:
        raise ValueError("No pairs are available for subset construction")
    focus_users = np.unique(selected_pairs[["userA", "userB"]].to_numpy().ravel()).astype(np.int64)
    if len(focus_users) > total_user_limit:
        raise ValueError("total_user_limit is smaller than the number of focus users")

    graph_users = np.sort(train_positive_edges["userId"].unique()).astype(np.int64)
    absent = np.setdiff1d(focus_users, graph_users)
    if len(absent):
        raise ValueError(f"Pair users missing from positive training graph: {absent.tolist()}")
    available_context = np.setdiff1d(graph_users, focus_users, assume_unique=True)
    context_count = min(total_user_limit - len(focus_users), len(available_context))
    rng = np.random.default_rng(random_seed)
    context_users = np.sort(
        rng.choice(available_context, size=context_count, replace=False)
    ).astype(np.int64)
    selected_users = np.concatenate([focus_users, context_users])
    subset_edges = train_positive_edges.loc[
        train_positive_edges["userId"].isin(selected_users),
        ["userId", "movieId"],
    ].reset_index(drop=True)
    graph = build_bipartite_graph(subset_edges)
    return ControlledGraphSubset(
        graph=graph,
        pairs=selected_pairs,
        focus_user_ids=focus_users,
        context_user_ids=context_users,
        positive_edges=subset_edges,
    )
