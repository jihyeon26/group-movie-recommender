"""Reproducible paired uncertainty estimates for ranking metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def paired_bootstrap_mean_difference(
    pair_metrics: pd.DataFrame,
    *,
    method_a: str,
    method_b: str,
    metric: str,
    n_resamples: int,
    random_seed: int,
) -> dict[str, Any]:
    """Bootstrap the mean paired difference, method A minus method B."""

    required = {"method", "pairId", metric}
    missing = required - set(pair_metrics.columns)
    if missing:
        raise ValueError(f"Pair metrics are missing columns: {sorted(missing)}")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if pair_metrics.duplicated(["method", "pairId"]).any():
        raise ValueError("Each method and pair must have at most one metric row")

    pivoted = pair_metrics.pivot(index="pairId", columns="method", values=metric)
    missing_methods = {method_a, method_b} - set(pivoted.columns)
    if missing_methods:
        raise ValueError(f"Methods are absent: {sorted(missing_methods)}")
    differences = (pivoted[method_a] - pivoted[method_b]).dropna().to_numpy(float)
    if not len(differences):
        raise ValueError("The selected methods have no common evaluated pairs")

    rng = np.random.default_rng(random_seed)
    sample_indices = rng.integers(
        0,
        len(differences),
        size=(n_resamples, len(differences)),
    )
    bootstrap_means = differences[sample_indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "methodA": method_a,
        "methodB": method_b,
        "metric": metric,
        "commonPairs": int(len(differences)),
        "meanDifferenceAminusB": float(differences.mean()),
        "ci95Lower": float(lower),
        "ci95Upper": float(upper),
        "bootstrapProbabilityAboveZero": float((bootstrap_means > 0).mean()),
        "resamples": int(n_resamples),
        "randomSeed": int(random_seed),
    }
