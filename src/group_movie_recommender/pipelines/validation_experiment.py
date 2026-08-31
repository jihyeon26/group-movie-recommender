"""Validation-selected training followed by a separately invoked frozen test run."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .lightgcn_training import LightGCNTrainingConfig, train_lightgcn
from ..algorithms.embedding_inference import recommend_pairs_from_embeddings
from ..algorithms.popularity import recommend_pairs_by_popularity
from ..evaluation.metrics import evaluate_shared_rankings
from ..evaluation.temporal_protocol import (
    TemporalEvaluationContext, build_temporal_context, load_focus_ratings,
)
from ..evaluation.uncertainty import paired_bootstrap_mean_difference
from ..preprocessing.config import PreprocessingConfig
from ..preprocessing.graph_subset import build_controlled_graph_subset
from ..shared.io import write_csv_gzip, write_json


LIMITATIONS = [
    "Exploratory test: the existing cohort was filtered using validation/test activity, "
    "and test data were previously inspected. This is not an untouched confirmatory test.",
    "Only 2,500 training users and the model's warm catalogue; not all MovieLens 32M.",
    "One seed and a small, declared search budget; validation selection can overfit.",
    "Synthetic pairs and observed individual ratings are not joint-viewing satisfaction.",
    "Missing ratings are unknown, not confirmed dislikes; metrics use observed positives.",
    "Pairs may share users. Pair-bootstrap intervals are descriptive and ignore this dependence.",
    "Test is a static recommendation from the 2020 cutoff with a train-only model; "
    "no refit on validation and no updates during the long test period.",
    "This evaluation is for trained graph users, not the separate real-user fold-in demo.",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selection_key(summary: dict, k: int) -> tuple[float, float]:
    """Protect the worse-off member first, then use average relevance to break ties."""
    scores = (float(summary[f"meanMinimumNDCG@{k}"]), float(summary[f"meanAverageNDCG@{k}"]))
    if not np.isfinite(scores).all():
        raise ValueError("Selection metrics must be finite")
    return scores


def evaluate_embeddings(
    artifact: dict,
    context: TemporalEvaluationContext,
    *,
    weight: float,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    recommendations = recommend_pairs_from_embeddings(
        context.pairs, artifact["user_ids"], artifact["user_embeddings"],
        artifact["movie_ids"], artifact["movie_embeddings"], context.seen,
        conflict_weight=weight, k=k,
    )
    metrics, summary = evaluate_shared_rankings(
        recommendations, context.relevance, catalog_size=context.catalog_size, k=k,
    )
    return recommendations, metrics, summary


def model_artifact(model, graph) -> dict:
    with torch.no_grad():
        users, movies = model.propagate()
    return {
        "user_ids": graph.user_ids, "user_embeddings": users.cpu().numpy(),
        "movie_ids": graph.movie_ids, "movie_embeddings": movies.cpu().numpy(),
    }


def load_artifact(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def run_selection(
    *, data_dir: Path, processed_dir: Path, output_root: Path,
    config_path: Path, preprocessing_path: Path,
) -> dict:
    """Never load test-period labels into model or checkpoint selection."""

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("Use a new output directory; prior experiments are preserved")
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    preprocessing = PreprocessingConfig.from_json(preprocessing_path)
    names = [run["name"] for run in settings["runs"]]
    if len(names) != len(set(names)) or any(Path(name).name != name for name in names):
        raise ValueError("Run names must be unique simple directory names")
    if {run["family"] for run in settings["runs"]} != {"lightgcn", "bpr_mf"}:
        raise ValueError("Declare LightGCN and BPR MF runs for matched comparison")
    weights = settings["conflict_weights"]
    if 0.0 not in weights or any(not 0 <= float(weight) <= 1 for weight in weights):
        raise ValueError("Conflict grid must contain average (0) and weights in [0, 1]")
    k = int(settings["k"])
    torch.set_num_threads(int(settings["torch_threads"]))
    torch.use_deterministic_algorithms(True)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(settings, output_root / "resolved_config.json")
    write_json(preprocessing.to_dict(), output_root / "preprocessing_config.json")

    print("Building the fixed training-only graph...", flush=True)
    train_path = processed_dir / "train_positive_edges.csv.gz"
    pair_path = processed_dir / "dissimilar_pairs.csv.gz"
    edges = pd.read_csv(train_path, dtype={"userId": "int32", "movieId": "int32"})
    pairs = pd.read_csv(pair_path, usecols=["userA", "userB"])
    subset = build_controlled_graph_subset(edges, pairs, **settings["subset"])
    del edges
    graph = subset.graph
    write_csv_gzip(subset.pairs, output_root / "focus_pairs.csv.gz")
    catalogue = subset.positive_edges.groupby("movieId").size().rename("trainPositiveCount").reset_index()
    write_csv_gzip(catalogue, output_root / "training_catalog.csv.gz")

    ratings_path = data_dir / "ratings.csv"
    validation_ratings = load_focus_ratings(
        ratings_path, subset.focus_user_ids,
        before_timestamp=preprocessing.validation_end_timestamp,
    )
    if (validation_ratings["timestamp"] >= preprocessing.validation_end_timestamp).any():
        raise AssertionError("Test-period events entered validation selection")
    context = build_temporal_context(
        subset.pairs, validation_ratings, graph.movie_ids, preprocessing, period="validation",
    )
    write_json(context.diagnostics, output_root / "validation_context.json")
    run_summaries = []
    for run in settings["runs"]:
        run_dir = output_root / "runs" / run["name"]
        run_dir.mkdir(parents=True)
        training = LightGCNTrainingConfig(**(settings["training"] | run["overrides"]))
        if (training.num_layers == 0) != (run["family"] == "bpr_mf"):
            raise ValueError("Model family and number of graph layers disagree")

        def validate(model, step):
            _, _, summary = evaluate_embeddings(
                model_artifact(model, graph), context, weight=0.0, k=k,
            )
            primary, secondary = selection_key(summary, k)
            print(f"{run['name']} step={step}: val min-NDCG={primary:.6f}, "
                  f"avg-NDCG={secondary:.6f}", flush=True)
            return primary, secondary

        started = time.perf_counter()
        model, history = train_lightgcn(
            graph, training, validation_callback=validate,
            validation_interval=int(settings["validation_interval"]),
            patience=int(settings["patience"]),
        )
        metadata = dict(history.attrs)
        artifact = model_artifact(model, graph)
        np.savez_compressed(run_dir / "final_embeddings.npz", **artifact)
        torch.save({"model_state_dict": model.state_dict(), "training_config": asdict(training),
                    "selection": metadata}, run_dir / "best_model.pt")
        write_csv_gzip(history, run_dir / "training_history.csv.gz")
        checkpoints = history.loc[history["validation_primary"].notna()]
        write_csv_gzip(checkpoints, run_dir / "validation_history.csv.gz")
        _, _, summary = evaluate_embeddings(artifact, context, weight=0.0, k=k)
        run_summary = {
            "name": run["name"], "family": run["family"], **metadata,
            "training_config": asdict(training), "validation": summary,
            "elapsed_seconds": time.perf_counter() - started,
            "artifact": f"runs/{run['name']}/final_embeddings.npz",
        }
        write_json(run_summary, run_dir / "summary.json")
        run_summaries.append(run_summary)
        print(f"{run['name']}: restored step {metadata['best_step']} "
              f"(stopped at {metadata['stopped_step']})", flush=True)

    selected = {
        family: max(
            (run for run in run_summaries if run["family"] == family),
            key=lambda run: selection_key(run["validation"], k),
        ) for family in ("lightgcn", "bpr_mf")
    }
    chosen_artifact = load_artifact(output_root / selected["lightgcn"]["artifact"])
    aggregation_rows = []
    for weight in weights:
        _, _, summary = evaluate_embeddings(chosen_artifact, context, weight=float(weight), k=k)
        aggregation_rows.append({"conflict_weight": float(weight), **summary})
    # Exact metric ties favor the simpler/smaller penalty, including average.
    selected_ranking = max(aggregation_rows, key=lambda row: (*selection_key(row, k), -row["conflict_weight"]))
    write_json({"methods": aggregation_rows}, output_root / "validation_aggregation.json")
    pd.DataFrame([
        {"run": run["name"], "family": run["family"], "best_step": run["best_step"],
         "stopped_step": run["stopped_step"], **run["validation"]}
        for run in run_summaries
    ]).to_csv(output_root / "validation_comparison.csv", index=False)

    protected_paths = ["resolved_config.json", "preprocessing_config.json", "focus_pairs.csv.gz",
                       "training_catalog.csv.gz", *[run["artifact"] for run in selected.values()]]
    manifest = {
        "status": "frozen_before_test", "settings": settings,
        "preprocessing": preprocessing.to_dict(),
        "selection_rule": "Checkpoint/run: average aggregation, maximize minimum NDCG then average NDCG; "
        "exact ties retain earlier run/step. Tune group weight afterwards on selected LightGCN.",
        "selected_runs": selected,
        "selected_conflict_weight": selected_ranking["conflict_weight"],
        "validation_selected_ranking": selected_ranking,
        "graph": {"users": graph.num_users, "movies": graph.num_movies, "edges": graph.num_positive_edges},
        "file_sha256": {str(path): sha256_file(output_root / path) for path in protected_paths},
        "source_sha256": {"ratings.csv": sha256_file(ratings_path),
                          "train_positive_edges.csv.gz": sha256_file(train_path),
                          "dissimilar_pairs.csv.gz": sha256_file(pair_path)},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__},
        "limitations": LIMITATIONS,
    }
    write_json(manifest, output_root / "selection.json")
    print(f"Selection frozen: {selected['lightgcn']['name']}, "
          f"step {selected['lightgcn']['best_step']}, weight {selected_ranking['conflict_weight']}", flush=True)
    return manifest


def verify_frozen_files(output_root: Path, manifest: dict) -> None:
    for relative, expected in manifest["file_sha256"].items():
        if sha256_file(output_root / relative) != expected:
            raise ValueError(f"Frozen artifact changed: {relative}")


def run_frozen_test(*, data_dir: Path, output_root: Path) -> dict:
    """Evaluate only declared frozen methods; do not select or train on test."""

    report_path = output_root / "test/test_report.json"
    if report_path.exists():
        raise FileExistsError("Final test report already exists; it will not be overwritten")
    manifest_path = output_root / "selection.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "frozen_before_test":
        raise ValueError("A frozen validation selection is required")
    verify_frozen_files(output_root, manifest)
    if sha256_file(data_dir / "ratings.csv") != manifest["source_sha256"]["ratings.csv"]:
        raise ValueError("Ratings file changed after selection")
    settings = manifest["settings"]
    k = int(settings["k"])
    preprocessing = PreprocessingConfig(**manifest["preprocessing"])
    pairs = pd.read_csv(output_root / "focus_pairs.csv.gz")
    catalogue = pd.read_csv(output_root / "training_catalog.csv.gz")
    focus_users = np.unique(pairs[["userA", "userB"]].to_numpy().ravel())
    ratings = load_focus_ratings(data_dir / "ratings.csv", focus_users)
    context = build_temporal_context(
        pairs, ratings, catalogue["movieId"].to_numpy(), preprocessing, period="test",
    )
    method_summaries, all_recommendations, all_metrics = [], [], []

    def collect(method, recommendations, metrics, summary):
        recommendations.insert(0, "method", method)
        metrics.insert(0, "method", method)
        all_recommendations.append(recommendations)
        all_metrics.append(metrics)
        method_summaries.append({"method": method, **summary})

    recommendations = recommend_pairs_by_popularity(pairs, catalogue, context.seen, k=k)
    metrics, summary = evaluate_shared_rankings(
        recommendations, context.relevance, catalog_size=context.catalog_size, k=k,
    )
    collect("popularity", recommendations, metrics, summary)
    selected_weight = float(manifest["selected_conflict_weight"])
    specifications = [
        ("bpr_mf_average", "bpr_mf", 0.0),
        ("lightgcn_average", "lightgcn", 0.0),
    ]
    if selected_weight != 0.0:
        specifications.append(("lightgcn_conflict_selected", "lightgcn", selected_weight))
    for method, family, weight in specifications:
        artifact = load_artifact(output_root / manifest["selected_runs"][family]["artifact"])
        if set(artifact["movie_ids"].astype(int)) != set(catalogue["movieId"].astype(int)):
            raise ValueError("All methods must use exactly the same movie catalogue")
        recommendations, metrics, summary = evaluate_embeddings(artifact, context, weight=weight, k=k)
        collect(method, recommendations, metrics, {"conflict_weight": weight, **summary})
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    final_method = "lightgcn_conflict_selected" if selected_weight else "lightgcn_average"
    comparisons = [
        (final_method, "popularity"),
        ("lightgcn_average", "bpr_mf_average"),
    ]
    if final_method != "lightgcn_average":
        comparisons.insert(1, (final_method, "lightgcn_average"))
    uncertainty = [
        paired_bootstrap_mean_difference(
            combined_metrics, method_a=a, method_b=b, metric=f"minimumNDCG@{k}",
            n_resamples=int(settings["bootstrap_resamples"]),
            random_seed=int(settings["bootstrap_seed"]),
        ) for a, b in comparisons
    ]
    report = {
        "purpose": "Frozen exploratory temporal test; no test-based model selection",
        "selection_manifest_sha256": sha256_file(manifest_path),
        "protocol": context.diagnostics,
        "selected_lightgcn": manifest["selected_runs"]["lightgcn"]["name"],
        "selected_step": manifest["selected_runs"]["lightgcn"]["best_step"],
        "selected_conflict_weight": manifest["selected_conflict_weight"],
        "selected_ranking_method": final_method,
        "methods": method_summaries, "paired_bootstrap": uncertainty,
        "limitations": manifest["limitations"],
    }
    write_csv_gzip(pd.concat(all_recommendations, ignore_index=True), output_root / "test/recommendations.csv.gz")
    write_csv_gzip(combined_metrics, output_root / "test/pair_metrics.csv.gz")
    pd.DataFrame(method_summaries).to_csv(output_root / "test/comparison.csv", index=False)
    write_json(report, report_path)
    print(pd.DataFrame(method_summaries).to_string(index=False), flush=True)
    return report
