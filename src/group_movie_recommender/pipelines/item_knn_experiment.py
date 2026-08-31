"""Validation-selected ItemKNN and exploratory comparison with frozen models."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ..algorithms.embedding_inference import recommend_pairs_from_score_matrix
from ..algorithms.item_knn import (
    fit_item_knn, load_item_knn, save_item_knn, score_item_knn_users,
)
from ..algorithms.popularity import recommend_pairs_by_popularity
from ..evaluation.metrics import evaluate_shared_rankings
from ..evaluation.temporal_protocol import build_temporal_context, load_focus_ratings
from ..evaluation.uncertainty import paired_bootstrap_mean_difference
from ..preprocessing.config import PreprocessingConfig
from ..shared.io import write_csv_gzip, write_json
from .als_experiment import _base_inputs
from .validation_experiment import (
    LIMITATIONS, evaluate_embeddings, load_artifact, selection_key, sha256_file,
)


def evaluate_item_knn(pairs, user_ids, scores, movie_ids, context, *, k):
    recommendations = recommend_pairs_from_score_matrix(
        pairs, user_ids, scores, movie_ids, context.seen,
        conflict_weight=0.0, k=k,
    )
    metrics, summary = evaluate_shared_rankings(
        recommendations, context.relevance, catalog_size=context.catalog_size, k=k,
    )
    return recommendations, metrics, summary


def run_item_knn_selection(
    *, data_dir: Path, processed_dir: Path, base_root: Path,
    output_root: Path, config_path: Path,
) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("Use a new ItemKNN output directory")
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    names = [row["name"] for row in settings["runs"]]
    if len(names) != len(set(names)):
        raise ValueError("ItemKNN run names must be unique")
    base, pairs, graph = _base_inputs(base_root, processed_dir)
    preprocessing = PreprocessingConfig(**base["preprocessing"])
    user_ids = np.unique(pairs[["userA", "userB"]].to_numpy().ravel())
    ratings = load_focus_ratings(
        data_dir / "ratings.csv", user_ids,
        before_timestamp=preprocessing.validation_end_timestamp,
    )
    context = build_temporal_context(
        pairs, ratings, graph.movie_ids, preprocessing, period="validation",
    )
    output_root.mkdir(parents=True)
    write_json(settings, output_root / "resolved_config.json")
    summaries = []
    models = {}
    max_neighbors_by_shrinkage = {
        float(shrinkage): max(int(row["neighbors"]) for row in settings["runs"]
                              if float(row["shrinkage"]) == float(shrinkage))
        for shrinkage in {row["shrinkage"] for row in settings["runs"]}
    }
    for shrinkage, max_neighbors in max_neighbors_by_shrinkage.items():
        print(f"Fitting ItemKNN shrinkage={shrinkage}, max_neighbors={max_neighbors}...", flush=True)
        models[shrinkage] = fit_item_knn(
            graph, max_neighbors=max_neighbors, shrinkage=shrinkage,
        )
    k = int(settings["k"])
    for specification in settings["runs"]:
        started = time.perf_counter()
        model = models[float(specification["shrinkage"])]
        scores = score_item_knn_users(
            model, graph, user_ids, neighbors=int(specification["neighbors"]),
        )
        _, _, validation = evaluate_item_knn(
            pairs, user_ids, scores, graph.movie_ids, context, k=k,
        )
        summary = {
            **specification, "validation": validation,
            "elapsed_scoring_seconds": time.perf_counter() - started,
        }
        summaries.append(summary)
        print(f"{specification['name']}: val min-NDCG="
              f"{validation[f'meanMinimumNDCG@{k}']:.6f}, avg-NDCG="
              f"{validation[f'meanAverageNDCG@{k}']:.6f}", flush=True)
    selected = max(summaries, key=lambda row: selection_key(row["validation"], k))
    model_path = output_root / "selected_item_knn.npz"
    save_item_knn(models[float(selected["shrinkage"])], model_path)
    pd.DataFrame([{"run": row["name"], "neighbors": row["neighbors"],
                   "shrinkage": row["shrinkage"], **row["validation"]}
                  for row in summaries]).to_csv(output_root / "validation_comparison.csv", index=False)
    manifest = {
        "status": "frozen_before_exploratory_test",
        "warning": "Test was already consumed; ItemKNN test results are exploratory.",
        "settings": settings, "selected_run": selected,
        "selection_rule": "Average aggregation; maximize validation minimum NDCG@10, then average NDCG@10.",
        "model": "selected_item_knn.npz", "model_sha256": sha256_file(model_path),
        "base_selection_sha256": sha256_file(base_root / "selection.json"),
        "limitations": LIMITATIONS,
    }
    write_json(manifest, output_root / "selection.json")
    print(f"ItemKNN selection frozen: {selected['name']}", flush=True)
    return manifest


def run_item_knn_exploratory_test(
    *, data_dir: Path, processed_dir: Path, base_root: Path,
    als_root: Path, output_root: Path,
) -> dict:
    report_path = output_root / "test/test_report.json"
    if report_path.exists():
        raise FileExistsError("ItemKNN exploratory test already exists")
    selection = json.loads((output_root / "selection.json").read_text(encoding="utf-8"))
    model_path = output_root / selection["model"]
    if sha256_file(model_path) != selection["model_sha256"]:
        raise ValueError("Frozen ItemKNN artifact changed")
    if sha256_file(base_root / "selection.json") != selection["base_selection_sha256"]:
        raise ValueError("Base selection changed")
    base, pairs, graph = _base_inputs(base_root, processed_dir)
    preprocessing = PreprocessingConfig(**base["preprocessing"])
    user_ids = np.unique(pairs[["userA", "userB"]].to_numpy().ravel())
    ratings = load_focus_ratings(data_dir / "ratings.csv", user_ids)
    context = build_temporal_context(
        pairs, ratings, graph.movie_ids, preprocessing, period="test",
    )
    k = int(selection["settings"]["k"])
    frames, metric_frames, summaries = [], [], []

    def collect(method, recommendations, metrics, summary):
        recommendations.insert(0, "method", method); metrics.insert(0, "method", method)
        frames.append(recommendations); metric_frames.append(metrics)
        summaries.append({"method": method, **summary})

    catalogue = pd.read_csv(base_root / "training_catalog.csv.gz")
    popularity = recommend_pairs_by_popularity(pairs, catalogue, context.seen, k=k)
    metrics, summary = evaluate_shared_rankings(
        popularity, context.relevance, catalog_size=context.catalog_size, k=k,
    )
    collect("popularity", popularity, metrics, summary)
    model = load_item_knn(model_path)
    scores = score_item_knn_users(
        model, graph, user_ids, neighbors=int(selection["selected_run"]["neighbors"]),
    )
    recommendations, metrics, summary = evaluate_item_knn(
        pairs, user_ids, scores, graph.movie_ids, context, k=k,
    )
    collect("item_knn_average", recommendations, metrics, {"conflict_weight": 0.0, **summary})
    als_selection = json.loads((als_root / "selection.json").read_text(encoding="utf-8"))
    artifacts = {
        "als_average": load_artifact(als_root / als_selection["selected_run"]["artifact"]),
        "bpr_mf_average": load_artifact(base_root / base["selected_runs"]["bpr_mf"]["artifact"]),
        "lightgcn_average": load_artifact(base_root / base["selected_runs"]["lightgcn"]["artifact"]),
    }
    for method, artifact in artifacts.items():
        recommendations, metrics, summary = evaluate_embeddings(
            artifact, context, weight=0.0, k=k,
        )
        collect(method, recommendations, metrics, {"conflict_weight": 0.0, **summary})
    combined = pd.concat(metric_frames, ignore_index=True)
    comparisons = [("item_knn_average", method) for method in
                   ("popularity", "als_average", "bpr_mf_average", "lightgcn_average")]
    bootstrap = [paired_bootstrap_mean_difference(
        combined, method_a=a, method_b=b, metric=metric,
        n_resamples=int(selection["settings"]["bootstrap_resamples"]),
        random_seed=int(selection["settings"]["bootstrap_seed"]),
    ) for metric in (f"minimumNDCG@{k}", f"averageNDCG@{k}") for a, b in comparisons]
    report = {
        "purpose": "Exploratory ItemKNN comparison after test was already consumed",
        "selected_item_knn": selection["selected_run"],
        "protocol": context.diagnostics, "methods": summaries,
        "paired_bootstrap": bootstrap, "limitations": selection["limitations"],
    }
    write_csv_gzip(pd.concat(frames, ignore_index=True), output_root / "test/recommendations.csv.gz")
    write_csv_gzip(combined, output_root / "test/pair_metrics.csv.gz")
    pd.DataFrame(summaries).to_csv(output_root / "test/comparison.csv", index=False)
    write_json(report, report_path)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    return report
