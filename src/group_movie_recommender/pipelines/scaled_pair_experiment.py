"""Large disjoint-pair robustness comparison for three fixed recommenders."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from ..algorithms.embedding_inference import (
    recommend_pairs_from_embeddings,
    recommend_pairs_from_score_matrix,
)
from ..algorithms.graph_data import build_bipartite_graph
from ..algorithms.item_knn import (
    fit_item_knn,
    load_item_knn,
    save_item_knn,
    score_item_knn_users,
)
from ..algorithms.popularity import recommend_pairs_by_popularity
from ..evaluation.metrics import evaluate_shared_rankings
from ..evaluation.temporal_protocol import build_temporal_context, load_focus_ratings
from ..evaluation.uncertainty import paired_bootstrap_mean_difference
from ..preprocessing.config import PreprocessingConfig
from ..preprocessing.graph_subset import build_controlled_graph_subset
from ..shared.io import write_csv_gzip, write_json
from .lightgcn_training import LightGCNTrainingConfig, train_lightgcn
from .validation_experiment import LIMITATIONS, model_artifact, selection_key, sha256_file


METHODS = ("popularity", "lightgcn_average", "item_knn_average")


def select_disjoint_pairs(pairs: pd.DataFrame, *, pair_limit: int) -> pd.DataFrame:
    """Greedily retain training-selected pairs without repeated members."""

    if pair_limit <= 0:
        raise ValueError("pair_limit must be positive")
    required = {"userA", "userB"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"Pairs are missing columns: {sorted(missing)}")
    used: set[int] = set()
    selected = []
    for row in pairs.itertuples(index=False):
        user_a, user_b = int(row.userA), int(row.userB)
        if user_a in used or user_b in used:
            continue
        selected.append(row)
        used.update((user_a, user_b))
        if len(selected) == pair_limit:
            break
    if len(selected) < pair_limit:
        raise ValueError(
            f"Only {len(selected)} disjoint pairs are available; requested {pair_limit}"
        )
    result = pd.DataFrame.from_records(selected, columns=pairs.columns)
    result = result.reset_index(drop=True)
    result.insert(0, "pairId", np.arange(len(result), dtype=np.int32))
    return result


def _validate_settings(settings: dict) -> None:
    subset = settings["subset"]
    pair_limit = int(subset["pair_limit"])
    counts = [int(value) for value in subset["nested_pair_counts"]]
    if counts != sorted(set(counts)) or not counts or counts[-1] != pair_limit:
        raise ValueError("Nested pair counts must be unique, sorted, and end at pair_limit")
    if counts[0] <= 0 or int(subset["total_user_limit"]) < 2 * pair_limit:
        raise ValueError("The graph must retain every member of every disjoint pair")


def _context_for_count(pairs, ratings, movie_ids, preprocessing, *, period, count):
    return build_temporal_context(
        pairs.head(count), ratings, movie_ids, preprocessing, period=period,
    )


def _evaluate_three_methods(
    *,
    context,
    catalogue,
    lightgcn_artifact,
    item_scores,
    item_user_ids,
    movie_ids,
    k,
):
    recommendations, metric_frames, summaries = [], [], []

    def collect(method, recs, metrics, summary):
        recs.insert(0, "method", method)
        metrics.insert(0, "method", method)
        recommendations.append(recs)
        metric_frames.append(metrics)
        summaries.append({"method": method, **summary})

    popularity = recommend_pairs_by_popularity(
        context.pairs, catalogue, context.seen, k=k,
    )
    metrics, summary = evaluate_shared_rankings(
        popularity, context.relevance, catalog_size=context.catalog_size, k=k,
    )
    collect("popularity", popularity, metrics, summary)
    lightgcn = recommend_pairs_from_embeddings(
        context.pairs,
        lightgcn_artifact["user_ids"],
        lightgcn_artifact["user_embeddings"],
        lightgcn_artifact["movie_ids"],
        lightgcn_artifact["movie_embeddings"],
        context.seen,
        conflict_weight=0.0,
        k=k,
    )
    metrics, summary = evaluate_shared_rankings(
        lightgcn, context.relevance, catalog_size=context.catalog_size, k=k,
    )
    collect("lightgcn_average", lightgcn, metrics, {"conflict_weight": 0.0, **summary})
    item_knn = recommend_pairs_from_score_matrix(
        context.pairs,
        item_user_ids,
        item_scores,
        movie_ids,
        context.seen,
        conflict_weight=0.0,
        k=k,
    )
    metrics, summary = evaluate_shared_rankings(
        item_knn, context.relevance, catalog_size=context.catalog_size, k=k,
    )
    collect("item_knn_average", item_knn, metrics, {"conflict_weight": 0.0, **summary})
    return (
        pd.concat(recommendations, ignore_index=True),
        pd.concat(metric_frames, ignore_index=True),
        summaries,
    )


def _bootstrap_comparisons(metrics, settings, *, k):
    comparisons = (
        ("lightgcn_average", "popularity"),
        ("item_knn_average", "popularity"),
        ("item_knn_average", "lightgcn_average"),
    )
    return [
        paired_bootstrap_mean_difference(
            metrics,
            method_a=method_a,
            method_b=method_b,
            metric=metric,
            n_resamples=int(settings["bootstrap_resamples"]),
            random_seed=int(settings["bootstrap_seed"]),
        )
        for metric in (f"minimumNDCG@{k}", f"averageNDCG@{k}")
        for method_a, method_b in comparisons
    ]


def _evaluate_nested(
    *,
    pairs,
    ratings,
    graph,
    preprocessing,
    catalogue,
    lightgcn_artifact,
    item_model,
    settings,
    period,
):
    focus = np.unique(pairs[["userA", "userB"]].to_numpy().ravel())
    item_scores = score_item_knn_users(
        item_model,
        graph,
        focus,
        neighbors=int(settings["item_knn"]["neighbors"]),
    )
    summaries, recommendations, metric_frames, bootstraps = [], [], [], []
    k = int(settings["k"])
    for count in settings["subset"]["nested_pair_counts"]:
        count = int(count)
        context = _context_for_count(
            pairs,
            ratings,
            graph.movie_ids,
            preprocessing,
            period=period,
            count=count,
        )
        recs, metrics, rows = _evaluate_three_methods(
            context=context,
            catalogue=catalogue,
            lightgcn_artifact=lightgcn_artifact,
            item_scores=item_scores,
            item_user_ids=focus,
            movie_ids=graph.movie_ids,
            k=k,
        )
        recs.insert(0, "pairCount", count)
        metrics.insert(0, "pairCount", count)
        for row in rows:
            summaries.append({"pairCount": count, **row})
        recommendations.append(recs)
        metric_frames.append(metrics)
        for comparison in _bootstrap_comparisons(metrics, settings, k=k):
            bootstraps.append({"pairCount": count, **comparison})
    return {
        "recommendations": pd.concat(recommendations, ignore_index=True),
        "pair_metrics": pd.concat(metric_frames, ignore_index=True),
        "comparison": pd.DataFrame(summaries),
        "bootstrap": bootstraps,
    }


def run_scaled_pair_selection(
    *,
    data_dir: Path,
    processed_dir: Path,
    output_root: Path,
    config_path: Path,
    preprocessing_path: Path,
) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("Use a new scaled-pair output directory")
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_settings(settings)
    preprocessing = PreprocessingConfig.from_json(preprocessing_path)
    torch.set_num_threads(int(settings["torch_threads"]))
    torch.use_deterministic_algorithms(True)
    edges_path = processed_dir / "train_positive_edges.csv.gz"
    pair_path = processed_dir / "dissimilar_pairs.csv.gz"
    edges = pd.read_csv(edges_path, dtype={"userId": "int32", "movieId": "int32"})
    pair_pool = pd.read_csv(pair_path).drop(columns=["jointTestPositiveCount"], errors="ignore")
    disjoint = select_disjoint_pairs(
        pair_pool, pair_limit=int(settings["subset"]["pair_limit"]),
    )
    subset = build_controlled_graph_subset(
        edges,
        disjoint.drop(columns="pairId"),
        pair_limit=len(disjoint),
        total_user_limit=int(settings["subset"]["total_user_limit"]),
        random_seed=int(settings["subset"]["random_seed"]),
    )
    del edges
    pairs = disjoint.copy()
    graph = subset.graph
    catalogue = (
        subset.positive_edges.groupby("movieId").size()
        .rename("trainPositiveCount")
        .reset_index()
    )
    focus = np.unique(pairs[["userA", "userB"]].to_numpy().ravel())
    validation_ratings = load_focus_ratings(
        data_dir / "ratings.csv",
        focus,
        before_timestamp=preprocessing.validation_end_timestamp,
    )
    full_context = build_temporal_context(
        pairs, validation_ratings, graph.movie_ids, preprocessing, period="validation",
    )
    output_root.mkdir(parents=True)
    write_json(settings, output_root / "resolved_config.json")
    write_json(preprocessing.to_dict(), output_root / "preprocessing_config.json")
    write_csv_gzip(pairs, output_root / "focus_pairs.csv.gz")
    write_csv_gzip(subset.positive_edges, output_root / "train_positive_edges.csv.gz")
    write_csv_gzip(catalogue, output_root / "training_catalog.csv.gz")
    training = LightGCNTrainingConfig(**{
        key: settings["lightgcn"][key]
        for key in (
            "embedding_dim", "num_layers", "steps", "batch_size",
            "learning_rate", "l2_weight", "random_seed",
        )
    })

    def validate(model, step):
        artifact = model_artifact(model, graph)
        recs = recommend_pairs_from_embeddings(
            full_context.pairs,
            artifact["user_ids"],
            artifact["user_embeddings"],
            artifact["movie_ids"],
            artifact["movie_embeddings"],
            full_context.seen,
            conflict_weight=0.0,
            k=int(settings["k"]),
        )
        _, summary = evaluate_shared_rankings(
            recs,
            full_context.relevance,
            catalog_size=full_context.catalog_size,
            k=int(settings["k"]),
        )
        key = selection_key(summary, int(settings["k"]))
        print(
            f"LightGCN step={step}: val min-NDCG={key[0]:.6f}, "
            f"avg-NDCG={key[1]:.6f}",
            flush=True,
        )
        return key

    started = time.perf_counter()
    lightgcn, history = train_lightgcn(
        graph,
        training,
        validation_callback=validate,
        validation_interval=int(settings["lightgcn"]["validation_interval"]),
        patience=int(settings["lightgcn"]["patience"]),
    )
    elapsed_lightgcn = time.perf_counter() - started
    lightgcn_artifact = model_artifact(lightgcn, graph)
    lightgcn_path = output_root / "lightgcn_embeddings.npz"
    np.savez_compressed(lightgcn_path, **lightgcn_artifact)
    write_csv_gzip(history, output_root / "lightgcn_training_history.csv.gz")
    print("Fitting ItemKNN on the same graph...", flush=True)
    started = time.perf_counter()
    item_model = fit_item_knn(
        graph,
        max_neighbors=int(settings["item_knn"]["neighbors"]),
        shrinkage=float(settings["item_knn"]["shrinkage"]),
    )
    elapsed_item_knn = time.perf_counter() - started
    item_path = output_root / "item_knn.npz"
    save_item_knn(item_model, item_path)
    evaluation = _evaluate_nested(
        pairs=pairs,
        ratings=validation_ratings,
        graph=graph,
        preprocessing=preprocessing,
        catalogue=catalogue,
        lightgcn_artifact=lightgcn_artifact,
        item_model=item_model,
        settings=settings,
        period="validation",
    )
    evaluation["comparison"].to_csv(
        output_root / "validation_comparison.csv", index=False,
    )
    write_csv_gzip(
        evaluation["recommendations"], output_root / "validation_recommendations.csv.gz",
    )
    write_csv_gzip(
        evaluation["pair_metrics"], output_root / "validation_pair_metrics.csv.gz",
    )
    write_json({"comparisons": evaluation["bootstrap"]}, output_root / "validation_bootstrap.json")
    protected = (
        "resolved_config.json",
        "preprocessing_config.json",
        "focus_pairs.csv.gz",
        "train_positive_edges.csv.gz",
        "training_catalog.csv.gz",
        "lightgcn_embeddings.npz",
        "item_knn.npz",
    )
    scaled_limitations = [
        limitation for limitation in LIMITATIONS
        if not limitation.startswith("Only 2,500 training users")
        and not limitation.startswith("Pairs may share users")
    ]
    scaled_limitations.extend([
        "The controlled graph contains 5,000 users and remains a subset of MovieLens 32M.",
        "The 500 evaluation pairs are member-disjoint, but they come from the same "
        "training-selected synthetic-pair pool and do not represent real joint-viewing groups.",
    ])
    manifest = {
        "status": "frozen_before_exploratory_test",
        "warning": "Test was already consumed; scaled test results are exploratory.",
        "methods": list(METHODS),
        "aggregation": "average percentile score for both personalized models",
        "selection_rule": (
            "Reuse prior hyperparameters; restore the LightGCN checkpoint maximizing "
            "500-pair validation minimum NDCG@10, then average NDCG@10."
        ),
        "lightgcn": {
            "config": asdict(training),
            **dict(history.attrs),
            "elapsed_seconds": elapsed_lightgcn,
        },
        "item_knn": {
            **settings["item_knn"],
            "elapsed_seconds": elapsed_item_knn,
        },
        "graph": {
            "users": graph.num_users,
            "movies": graph.num_movies,
            "edges": graph.num_positive_edges,
            "focus_pairs": len(pairs),
            "focus_users": len(focus),
        },
        "validation_protocol": full_context.diagnostics,
        "file_sha256": {
            relative: sha256_file(output_root / relative) for relative in protected
        },
        "source_sha256": {
            "train_positive_edges.csv.gz": sha256_file(edges_path),
            "dissimilar_pairs.csv.gz": sha256_file(pair_path),
        },
        "limitations": scaled_limitations,
    }
    write_json(manifest, output_root / "selection.json")
    print(
        f"Scaled validation frozen: LightGCN step {history.attrs['best_step']}, "
        f"500 disjoint pairs",
        flush=True,
    )
    return manifest


def run_scaled_pair_exploratory_test(
    *,
    data_dir: Path,
    output_root: Path,
) -> dict:
    report_path = output_root / "test/test_report.json"
    if report_path.exists():
        raise FileExistsError("Scaled exploratory test already exists")
    manifest = json.loads((output_root / "selection.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["file_sha256"].items():
        if sha256_file(output_root / relative) != expected:
            raise ValueError(f"Frozen scaled-pair artifact changed: {relative}")
    settings = json.loads((output_root / "resolved_config.json").read_text(encoding="utf-8"))
    preprocessing = PreprocessingConfig.from_json(output_root / "preprocessing_config.json")
    pairs = pd.read_csv(output_root / "focus_pairs.csv.gz")
    edges = pd.read_csv(output_root / "train_positive_edges.csv.gz")
    graph = build_bipartite_graph(edges)
    catalogue = pd.read_csv(output_root / "training_catalog.csv.gz")
    with np.load(output_root / "lightgcn_embeddings.npz", allow_pickle=False) as archive:
        lightgcn_artifact = {key: archive[key] for key in archive.files}
    item_model = load_item_knn(output_root / "item_knn.npz")
    focus = np.unique(pairs[["userA", "userB"]].to_numpy().ravel())
    ratings = load_focus_ratings(data_dir / "ratings.csv", focus)
    evaluation = _evaluate_nested(
        pairs=pairs,
        ratings=ratings,
        graph=graph,
        preprocessing=preprocessing,
        catalogue=catalogue,
        lightgcn_artifact=lightgcn_artifact,
        item_model=item_model,
        settings=settings,
        period="test",
    )
    test_root = output_root / "test"
    test_root.mkdir(parents=True, exist_ok=True)
    evaluation["comparison"].to_csv(test_root / "comparison.csv", index=False)
    write_csv_gzip(evaluation["recommendations"], test_root / "recommendations.csv.gz")
    write_csv_gzip(evaluation["pair_metrics"], test_root / "pair_metrics.csv.gz")
    report = {
        "purpose": "Exploratory robustness comparison on up to 500 disjoint pairs",
        "methods": list(METHODS),
        "nested_pair_counts": settings["subset"]["nested_pair_counts"],
        "comparisons": evaluation["comparison"].to_dict(orient="records"),
        "paired_bootstrap": evaluation["bootstrap"],
        "limitations": manifest["limitations"],
    }
    write_json(report, report_path)
    print(evaluation["comparison"].to_string(index=False), flush=True)
    return report
