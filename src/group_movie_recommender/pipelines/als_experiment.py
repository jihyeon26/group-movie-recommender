"""Validation-selected implicit ALS and an explicitly exploratory frozen comparison."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from ..algorithms.embedding_inference import recommend_pairs_from_embeddings
from ..algorithms.graph_data import build_bipartite_graph
from ..algorithms.implicit_als import ImplicitALSConfig, train_implicit_als
from ..algorithms.popularity import recommend_pairs_by_popularity
from ..evaluation.metrics import evaluate_shared_rankings
from ..evaluation.temporal_protocol import build_temporal_context, load_focus_ratings
from ..evaluation.uncertainty import paired_bootstrap_mean_difference
from ..preprocessing.config import PreprocessingConfig
from ..shared.io import write_csv_gzip, write_json
from .validation_experiment import (
    LIMITATIONS, evaluate_embeddings, load_artifact, selection_key, sha256_file,
    verify_frozen_files,
)


def _base_inputs(base_root: Path, processed_dir: Path):
    manifest = json.loads((base_root / "selection.json").read_text(encoding="utf-8"))
    verify_frozen_files(base_root, manifest)
    pairs = pd.read_csv(base_root / "focus_pairs.csv.gz")
    focus = np.unique(pairs[["userA", "userB"]].to_numpy().ravel())
    edges = pd.read_csv(
        processed_dir / "train_positive_edges.csv.gz",
        dtype={"userId": "int32", "movieId": "int32"},
    )
    graph = build_bipartite_graph(edges.loc[edges["userId"].isin(
        np.concatenate([focus, _context_users(manifest, edges, focus)])
    )])
    expected_movies = pd.read_csv(base_root / "training_catalog.csv.gz")["movieId"].to_numpy()
    if not np.array_equal(graph.movie_ids, np.sort(expected_movies)):
        raise ValueError("ALS and frozen LightGCN catalogues differ")
    if graph.num_users != manifest["graph"]["users"] or graph.num_positive_edges != manifest["graph"]["edges"]:
        raise ValueError("ALS graph does not match the frozen experiment graph")
    return manifest, pairs, graph


def _context_users(manifest: dict, edges: pd.DataFrame, focus: np.ndarray) -> np.ndarray:
    """Reproduce the controlled subset's seeded context-user sample exactly."""
    settings = manifest["settings"]["subset"]
    graph_users = np.sort(edges["userId"].unique()).astype(np.int64)
    available = np.setdiff1d(graph_users, focus, assume_unique=True)
    count = min(int(settings["total_user_limit"]) - len(focus), len(available))
    rng = np.random.default_rng(int(settings["random_seed"]))
    return np.sort(rng.choice(available, size=count, replace=False)).astype(np.int64)


def run_als_selection(
    *, data_dir: Path, processed_dir: Path, base_root: Path,
    output_root: Path, config_path: Path,
) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("Use a new ALS output directory")
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    names = [row["name"] for row in settings["runs"]]
    if len(names) != len(set(names)):
        raise ValueError("ALS run names must be unique")
    torch.set_num_threads(int(settings["torch_threads"]))
    torch.use_deterministic_algorithms(True)
    base, pairs, graph = _base_inputs(base_root, processed_dir)
    preprocessing = PreprocessingConfig(**base["preprocessing"])
    focus = np.unique(pairs[["userA", "userB"]].to_numpy().ravel())
    ratings = load_focus_ratings(
        data_dir / "ratings.csv", focus,
        before_timestamp=preprocessing.validation_end_timestamp,
    )
    context = build_temporal_context(
        pairs, ratings, graph.movie_ids, preprocessing, period="validation",
    )
    output_root.mkdir(parents=True)
    write_json(settings, output_root / "resolved_config.json")
    summaries = []
    for specification in settings["runs"]:
        run_dir = output_root / "runs" / specification["name"]
        run_dir.mkdir(parents=True)
        config = ImplicitALSConfig(
            factors=int(specification["factors"]),
            alpha=float(specification["alpha"]),
            regularization=float(specification["regularization"]),
            iterations=int(settings["iterations"]),
            batch_size=int(settings["batch_size"]),
            patience=int(settings["patience"]),
            validation_interval=int(settings["validation_interval"]),
            random_seed=int(settings["random_seed"]),
        )

        def validate(users, movies, iteration):
            artifact = {"user_ids": graph.user_ids, "user_embeddings": users,
                        "movie_ids": graph.movie_ids, "movie_embeddings": movies}
            _, _, summary = evaluate_embeddings(artifact, context, weight=0.0, k=int(settings["k"]))
            scores = selection_key(summary, int(settings["k"]))
            print(f"{specification['name']} iteration={iteration}: "
                  f"val min-NDCG={scores[0]:.6f}, avg-NDCG={scores[1]:.6f}", flush=True)
            return scores

        started = time.perf_counter()
        result = train_implicit_als(graph, config, validation_callback=validate)
        artifact = {
            "user_ids": graph.user_ids, "user_embeddings": result.user_factors,
            "movie_ids": graph.movie_ids, "movie_embeddings": result.movie_factors,
        }
        artifact_path = run_dir / "final_embeddings.npz"
        np.savez_compressed(artifact_path, **artifact)
        write_csv_gzip(result.history, run_dir / "validation_history.csv.gz")
        _, _, validation = evaluate_embeddings(
            artifact, context, weight=0.0, k=int(settings["k"]),
        )
        summary = {
            "name": specification["name"], "config": asdict(config),
            "best_iteration": result.best_iteration,
            "stopped_iteration": result.stopped_iteration,
            "early_stopped": result.early_stopped,
            "validation": validation,
            "elapsed_seconds": time.perf_counter() - started,
            "artifact": f"runs/{specification['name']}/final_embeddings.npz",
        }
        write_json(summary, run_dir / "summary.json")
        summaries.append(summary)
    k = int(settings["k"])
    selected = max(summaries, key=lambda row: selection_key(row["validation"], k))
    pd.DataFrame([{"run": row["name"], "best_iteration": row["best_iteration"],
                   "stopped_iteration": row["stopped_iteration"], **row["validation"]}
                  for row in summaries]).to_csv(output_root / "validation_comparison.csv", index=False)
    manifest = {
        "status": "frozen_before_exploratory_test",
        "warning": "Test was already consumed by prior experiments; any ALS test result is exploratory.",
        "settings": settings, "selected_run": selected,
        "selection_rule": "Average aggregation; maximize validation minimum NDCG@10, then average NDCG@10.",
        "base_selection_sha256": sha256_file(base_root / "selection.json"),
        "artifact_sha256": sha256_file(output_root / selected["artifact"]),
        "limitations": LIMITATIONS,
    }
    write_json(manifest, output_root / "selection.json")
    print(f"ALS selection frozen: {selected['name']} at iteration {selected['best_iteration']}", flush=True)
    return manifest


def run_als_exploratory_test(
    *, data_dir: Path, processed_dir: Path, base_root: Path, output_root: Path,
) -> dict:
    report_path = output_root / "test/test_report.json"
    if report_path.exists():
        raise FileExistsError("ALS exploratory test already exists")
    selection = json.loads((output_root / "selection.json").read_text(encoding="utf-8"))
    if sha256_file(base_root / "selection.json") != selection["base_selection_sha256"]:
        raise ValueError("Base frozen selection changed")
    als_path = output_root / selection["selected_run"]["artifact"]
    if sha256_file(als_path) != selection["artifact_sha256"]:
        raise ValueError("Frozen ALS artifact changed")
    base, pairs, graph = _base_inputs(base_root, processed_dir)
    preprocessing = PreprocessingConfig(**base["preprocessing"])
    focus = np.unique(pairs[["userA", "userB"]].to_numpy().ravel())
    ratings = load_focus_ratings(data_dir / "ratings.csv", focus)
    context = build_temporal_context(pairs, ratings, graph.movie_ids, preprocessing, period="test")
    k = int(selection["settings"]["k"])
    catalogue = pd.read_csv(base_root / "training_catalog.csv.gz")
    frames, metric_frames, summaries = [], [], []

    def collect(method, recs, metrics, summary):
        recs.insert(0, "method", method); metrics.insert(0, "method", method)
        frames.append(recs); metric_frames.append(metrics); summaries.append({"method": method, **summary})

    popularity = recommend_pairs_by_popularity(pairs, catalogue, context.seen, k=k)
    metrics, summary = evaluate_shared_rankings(
        popularity, context.relevance, catalog_size=context.catalog_size, k=k,
    )
    collect("popularity", popularity, metrics, summary)
    artifacts = {
        "als_average": load_artifact(als_path),
        "bpr_mf_average": load_artifact(base_root / base["selected_runs"]["bpr_mf"]["artifact"]),
        "lightgcn_average": load_artifact(base_root / base["selected_runs"]["lightgcn"]["artifact"]),
    }
    for method, artifact in artifacts.items():
        recs, metrics, summary = evaluate_embeddings(artifact, context, weight=0.0, k=k)
        collect(method, recs, metrics, {"conflict_weight": 0.0, **summary})
    combined = pd.concat(metric_frames, ignore_index=True)
    comparisons = [("als_average", other) for other in
                   ("popularity", "bpr_mf_average", "lightgcn_average")]
    bootstrap = [paired_bootstrap_mean_difference(
        combined, method_a=a, method_b=b, metric=metric,
        n_resamples=int(selection["settings"]["bootstrap_resamples"]),
        random_seed=int(selection["settings"]["bootstrap_seed"]),
    ) for metric in (f"minimumNDCG@{k}", f"averageNDCG@{k}") for a, b in comparisons]
    report = {
        "purpose": "Exploratory frozen comparison after test had already been consumed",
        "selected_als": selection["selected_run"], "protocol": context.diagnostics,
        "methods": summaries, "paired_bootstrap": bootstrap,
        "limitations": selection["limitations"],
    }
    write_csv_gzip(pd.concat(frames, ignore_index=True), output_root / "test/recommendations.csv.gz")
    write_csv_gzip(combined, output_root / "test/pair_metrics.csv.gz")
    pd.DataFrame(summaries).to_csv(output_root / "test/comparison.csv", index=False)
    write_json(report, report_path)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    return report
