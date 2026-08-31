"""Validation-selected explicit MF with group-aggregation and conflict analysis."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from ..algorithms.embedding_inference import recommend_pairs_from_score_matrix
from ..algorithms.explicit_mf import (
    ExplicitMFConfig,
    build_explicit_rating_data,
    score_explicit_mf_users,
    train_explicit_mf,
)
from ..algorithms.item_knn import load_item_knn, score_item_knn_users
from ..algorithms.popularity import recommend_pairs_by_popularity
from ..evaluation.metrics import evaluate_shared_rankings
from ..evaluation.temporal_protocol import build_temporal_context, load_focus_ratings
from ..evaluation.uncertainty import paired_bootstrap_mean_difference
from ..preprocessing.config import PreprocessingConfig
from ..shared.io import write_csv_gzip, write_json
from .als_experiment import _base_inputs
from .item_knn_experiment import evaluate_item_knn
from .validation_experiment import (
    LIMITATIONS,
    evaluate_embeddings,
    load_artifact,
    selection_key,
    sha256_file,
)


def evaluate_explicit_scores(
    scores: np.ndarray,
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
    context,
    specification: dict,
    *,
    k: int,
):
    recommendations = recommend_pairs_from_score_matrix(
        context.pairs,
        user_ids,
        scores,
        movie_ids,
        context.seen,
        conflict_weight=float(specification["conflict_weight"]),
        aggregation=str(specification["kind"]),
        k=k,
    )
    metrics, summary = evaluate_shared_rankings(
        recommendations, context.relevance, catalog_size=context.catalog_size, k=k,
    )
    return recommendations, metrics, summary


def add_conflict_bands(pairs: pd.DataFrame, pair_features: pd.DataFrame) -> pd.DataFrame:
    """Create equal-sized conflict bands from training-only pair similarities."""

    features = pair_features.loc[
        :, ["userA", "userB", "coRatedTrain", "genreSimilarity", "ratingCorrelation"]
    ]
    result = pairs.merge(
        features, on=["userA", "userB"], how="left", validate="one_to_one"
    )
    if result[["genreSimilarity", "ratingCorrelation"]].isna().any().any():
        raise ValueError("Training-only conflict features are missing for focus pairs")
    genre_conflict = (-result["genreSimilarity"]).rank(method="average", pct=True)
    rating_conflict = (-result["ratingCorrelation"]).rank(method="average", pct=True)
    result["conflictScore"] = (genre_conflict + rating_conflict) / 2.0
    result["conflictBand"] = pd.qcut(
        result["conflictScore"],
        q=3,
        labels=["low", "medium", "high"],
    ).astype("string")
    return result


def summarize_conflict_bands(
    metric_frames: list[pd.DataFrame],
    pair_bands: pd.DataFrame,
    *,
    k: int,
) -> pd.DataFrame:
    """Summarize relevance and hit balance within each training conflict band."""

    metrics = pd.concat(metric_frames, ignore_index=True).merge(
        pair_bands[["pairId", "conflictScore", "conflictBand"]],
        on="pairId",
        how="left",
        validate="many_to_one",
    )
    rows = []
    for (method, band), group in metrics.groupby(
        ["method", "conflictBand"], observed=True, sort=False
    ):
        hit_a = group[f"ndcgA@{k}"].gt(0)
        hit_b = group[f"ndcgB@{k}"].gt(0)
        rows.append({
            "method": method,
            "conflictBand": band,
            "evaluatedPairs": len(group),
            "meanConflictScore": group["conflictScore"].mean(),
            f"meanAverageNDCG@{k}": group[f"averageNDCG@{k}"].mean(),
            f"meanMinimumNDCG@{k}": group[f"minimumNDCG@{k}"].mean(),
            f"meanNDCGGap@{k}": group[f"ndcgGap@{k}"].mean(),
            f"meanAverageRecall@{k}": group[f"averageRecall@{k}"].mean(),
            f"meanMinimumRecall@{k}": group[f"minimumRecall@{k}"].mean(),
            "twoSidedHitPairRate": (hit_a & hit_b).mean(),
            "noHitPairRate": (~hit_a & ~hit_b).mean(),
        })
    return pd.DataFrame(rows)


def _load_training_data(data_dir, graph, preprocessing):
    ratings = load_focus_ratings(
        data_dir / "ratings.csv",
        graph.user_ids,
        before_timestamp=preprocessing.train_end_timestamp,
    )
    ratings = ratings.loc[ratings["movieId"].isin(graph.movie_ids)]
    if (ratings["timestamp"] >= preprocessing.train_end_timestamp).any():
        raise AssertionError("Post-cutoff ratings entered explicit-MF training")
    return build_explicit_rating_data(ratings, graph.user_ids, graph.movie_ids)


def run_explicit_mf_selection(
    *,
    data_dir: Path,
    processed_dir: Path,
    base_root: Path,
    output_root: Path,
    config_path: Path,
) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("Use a new explicit-MF output directory")
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    names = [run["name"] for run in settings["runs"]]
    if len(names) != len(set(names)):
        raise ValueError("Explicit-MF run names must be unique")
    aggregation_names = [row["name"] for row in settings["aggregations"]]
    if len(aggregation_names) != len(set(aggregation_names)):
        raise ValueError("Aggregation names must be unique")
    if "average" not in aggregation_names or "nash" not in aggregation_names:
        raise ValueError("Aggregation search must contain average and Nash")
    torch.set_num_threads(int(settings["torch_threads"]))
    torch.use_deterministic_algorithms(True)
    base, pairs, graph = _base_inputs(base_root, processed_dir)
    preprocessing = PreprocessingConfig(**base["preprocessing"])
    focus = np.unique(pairs[["userA", "userB"]].to_numpy().ravel())
    validation_ratings = load_focus_ratings(
        data_dir / "ratings.csv",
        focus,
        before_timestamp=preprocessing.validation_end_timestamp,
    )
    context = build_temporal_context(
        pairs, validation_ratings, graph.movie_ids, preprocessing, period="validation"
    )
    training = _load_training_data(data_dir, graph, preprocessing)
    pair_features = pd.read_csv(processed_dir / "dissimilar_pairs.csv.gz")
    pair_bands = add_conflict_bands(pairs, pair_features)
    output_root.mkdir(parents=True)
    write_json(settings, output_root / "resolved_config.json")
    write_csv_gzip(pair_bands, output_root / "pair_conflict_bands.csv.gz")
    summaries = []
    k = int(settings["k"])
    average = next(row for row in settings["aggregations"] if row["name"] == "average")
    for specification in settings["runs"]:
        run_dir = output_root / "runs" / specification["name"]
        run_dir.mkdir(parents=True)
        config = ExplicitMFConfig(
            factors=int(specification["factors"]),
            epochs=int(settings["epochs"]),
            learning_rate=float(specification["learning_rate"]),
            regularization=float(specification["regularization"]),
            batch_size=int(settings["batch_size"]),
            random_seed=int(settings["random_seed"]),
            patience=int(settings["patience"]),
            validation_interval=int(settings["validation_interval"]),
        )

        def validate(artifact, epoch):
            scores = score_explicit_mf_users(artifact, focus)
            _, _, summary = evaluate_explicit_scores(
                scores, focus, graph.movie_ids, context, average, k=k,
            )
            key = selection_key(summary, k)
            print(
                f"{specification['name']} epoch={epoch}: "
                f"val min-NDCG={key[0]:.6f}, avg-NDCG={key[1]:.6f}",
                flush=True,
            )
            return key

        started = time.perf_counter()
        result = train_explicit_mf(training, config, validation_callback=validate)
        artifact_path = run_dir / "explicit_mf.npz"
        np.savez_compressed(artifact_path, **result.artifact)
        write_csv_gzip(result.history, run_dir / "training_history.csv.gz")
        scores = score_explicit_mf_users(result.artifact, focus)
        _, _, validation = evaluate_explicit_scores(
            scores, focus, graph.movie_ids, context, average, k=k,
        )
        summary = {
            "name": specification["name"],
            "config": asdict(config),
            "best_epoch": result.best_epoch,
            "stopped_epoch": result.stopped_epoch,
            "early_stopped": result.early_stopped,
            "training_ratings": int(len(training.ratings)),
            "training_rating_mean": float(training.ratings.mean()),
            "validation": validation,
            "elapsed_seconds": time.perf_counter() - started,
            "artifact": f"runs/{specification['name']}/explicit_mf.npz",
        }
        write_json(summary, run_dir / "summary.json")
        summaries.append(summary)
    selected = max(summaries, key=lambda row: selection_key(row["validation"], k))
    artifact = load_artifact(output_root / selected["artifact"])
    scores = score_explicit_mf_users(artifact, focus)
    aggregation_rows = []
    aggregation_metrics = []
    aggregation_recommendations = []
    for priority, specification in enumerate(settings["aggregations"]):
        recommendations, metrics, summary = evaluate_explicit_scores(
            scores, focus, graph.movie_ids, context, specification, k=k,
        )
        method = f"explicit_mf_{specification['name']}"
        recommendations.insert(0, "method", method)
        metrics.insert(0, "method", method)
        aggregation_recommendations.append(recommendations)
        aggregation_metrics.append(metrics)
        aggregation_rows.append({
            **specification,
            "selection_priority": priority,
            **summary,
        })
    selected_aggregation = max(
        aggregation_rows,
        key=lambda row: (*selection_key(row, k), -row["selection_priority"]),
    )
    comparison = pd.DataFrame([
        {"run": row["name"], "best_epoch": row["best_epoch"],
         "stopped_epoch": row["stopped_epoch"], **row["validation"]}
        for row in summaries
    ])
    comparison.to_csv(output_root / "validation_model_comparison.csv", index=False)
    pd.DataFrame(aggregation_rows).to_csv(
        output_root / "validation_aggregation_comparison.csv", index=False
    )
    write_csv_gzip(
        pd.concat(aggregation_recommendations, ignore_index=True),
        output_root / "validation_aggregation_recommendations.csv.gz",
    )
    write_csv_gzip(
        pd.concat(aggregation_metrics, ignore_index=True),
        output_root / "validation_aggregation_pair_metrics.csv.gz",
    )
    subgroup = summarize_conflict_bands(aggregation_metrics, pair_bands, k=k)
    subgroup.to_csv(output_root / "validation_conflict_subgroups.csv", index=False)
    manifest = {
        "status": "frozen_before_exploratory_test",
        "warning": "Test was already consumed; explicit-MF test results are exploratory.",
        "settings": settings,
        "selected_run": selected,
        "selected_aggregation": selected_aggregation,
        "selection_rule": (
            "Select model with average aggregation, then aggregation; maximize validation "
            "minimum NDCG@10, then average NDCG@10, with declared order breaking ties."
        ),
        "base_selection_sha256": sha256_file(base_root / "selection.json"),
        "artifact_sha256": sha256_file(output_root / selected["artifact"]),
        "training_feedback": {
            "type": "explicit ratings 0.5-5.0",
            "ratings": int(len(training.ratings)),
            "ratings_below_positive_threshold": int(
                (training.ratings < preprocessing.positive_threshold).sum()
            ),
        },
        "limitations": LIMITATIONS,
    }
    write_json(manifest, output_root / "selection.json")
    print(
        f"Explicit MF selection frozen: {selected['name']}, "
        f"aggregation={selected_aggregation['name']}",
        flush=True,
    )
    return manifest


def run_explicit_mf_exploratory_test(
    *,
    data_dir: Path,
    processed_dir: Path,
    base_root: Path,
    als_root: Path,
    item_knn_root: Path,
    output_root: Path,
) -> dict:
    report_path = output_root / "test/test_report.json"
    if report_path.exists():
        raise FileExistsError("Explicit-MF exploratory test already exists")
    selection = json.loads((output_root / "selection.json").read_text(encoding="utf-8"))
    if sha256_file(base_root / "selection.json") != selection["base_selection_sha256"]:
        raise ValueError("Base selection changed")
    artifact_path = output_root / selection["selected_run"]["artifact"]
    if sha256_file(artifact_path) != selection["artifact_sha256"]:
        raise ValueError("Frozen explicit-MF artifact changed")
    base, pairs, graph = _base_inputs(base_root, processed_dir)
    preprocessing = PreprocessingConfig(**base["preprocessing"])
    focus = np.unique(pairs[["userA", "userB"]].to_numpy().ravel())
    ratings = load_focus_ratings(data_dir / "ratings.csv", focus)
    context = build_temporal_context(
        pairs, ratings, graph.movie_ids, preprocessing, period="test"
    )
    pair_bands = pd.read_csv(output_root / "pair_conflict_bands.csv.gz")
    artifact = load_artifact(artifact_path)
    scores = score_explicit_mf_users(artifact, focus)
    k = int(selection["settings"]["k"])
    frames, metric_frames, summaries = [], [], []

    def collect(method, recommendations, metrics, summary):
        recommendations.insert(0, "method", method)
        metrics.insert(0, "method", method)
        frames.append(recommendations)
        metric_frames.append(metrics)
        summaries.append({"method": method, **summary})

    catalogue = pd.read_csv(base_root / "training_catalog.csv.gz")
    popularity = recommend_pairs_by_popularity(pairs, catalogue, context.seen, k=k)
    metrics, summary = evaluate_shared_rankings(
        popularity, context.relevance, catalog_size=context.catalog_size, k=k,
    )
    collect("popularity", popularity, metrics, summary)
    chosen = selection["selected_aggregation"]
    recommendations, metrics, summary = evaluate_explicit_scores(
        scores, focus, graph.movie_ids, context, chosen, k=k,
    )
    selected_method = f"explicit_mf_{chosen['name']}"
    collect(selected_method, recommendations, metrics, {**chosen, **summary})
    if chosen["name"] != "average":
        average = next(
            row for row in selection["settings"]["aggregations"]
            if row["name"] == "average"
        )
        recommendations, metrics, summary = evaluate_explicit_scores(
            scores, focus, graph.movie_ids, context, average, k=k,
        )
        collect("explicit_mf_average", recommendations, metrics, {**average, **summary})
    als_selection = json.loads((als_root / "selection.json").read_text(encoding="utf-8"))
    item_selection = json.loads(
        (item_knn_root / "selection.json").read_text(encoding="utf-8")
    )
    comparison_artifacts = {
        "als_average": load_artifact(
            als_root / als_selection["selected_run"]["artifact"]
        ),
        "bpr_mf_average": load_artifact(
            base_root / base["selected_runs"]["bpr_mf"]["artifact"]
        ),
        "lightgcn_average": load_artifact(
            base_root / base["selected_runs"]["lightgcn"]["artifact"]
        ),
    }
    for method, other_artifact in comparison_artifacts.items():
        recommendations, metrics, summary = evaluate_embeddings(
            other_artifact, context, weight=0.0, k=k,
        )
        collect(method, recommendations, metrics, {"conflict_weight": 0.0, **summary})
    item_model = load_item_knn(item_knn_root / item_selection["model"])
    item_scores = score_item_knn_users(
        item_model,
        graph,
        focus,
        neighbors=int(item_selection["selected_run"]["neighbors"]),
    )
    recommendations, metrics, summary = evaluate_item_knn(
        pairs, focus, item_scores, graph.movie_ids, context, k=k,
    )
    collect("item_knn_average", recommendations, metrics, {"conflict_weight": 0.0, **summary})
    combined = pd.concat(metric_frames, ignore_index=True)
    comparison_methods = [
        method for method in combined["method"].unique()
        if method != selected_method
    ]
    bootstrap = [
        paired_bootstrap_mean_difference(
            combined,
            method_a=selected_method,
            method_b=other,
            metric=metric,
            n_resamples=int(selection["settings"]["bootstrap_resamples"]),
            random_seed=int(selection["settings"]["bootstrap_seed"]),
        )
        for metric in (f"minimumNDCG@{k}", f"averageNDCG@{k}")
        for other in comparison_methods
    ]
    subgroup = summarize_conflict_bands(metric_frames, pair_bands, k=k)
    report = {
        "purpose": "Exploratory explicit-rating MF comparison after test was consumed",
        "selected_run": selection["selected_run"],
        "selected_aggregation": chosen,
        "protocol": context.diagnostics,
        "methods": summaries,
        "paired_bootstrap": bootstrap,
        "limitations": selection["limitations"],
    }
    write_csv_gzip(
        pd.concat(frames, ignore_index=True),
        output_root / "test/recommendations.csv.gz",
    )
    write_csv_gzip(combined, output_root / "test/pair_metrics.csv.gz")
    pd.DataFrame(summaries).to_csv(output_root / "test/comparison.csv", index=False)
    subgroup.to_csv(output_root / "test/conflict_subgroups.csv", index=False)
    write_json(report, report_path)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    return report
