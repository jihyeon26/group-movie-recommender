"""Pre-test-selected user holdout for the final three-model report table."""

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
from ..algorithms.item_knn import (
    fit_item_knn,
    load_item_knn,
    save_item_knn,
    score_item_knn_users,
)
from ..evaluation.metrics import evaluate_shared_rankings
from ..evaluation.temporal_protocol import build_temporal_context, load_focus_ratings
from ..preprocessing.config import PreprocessingConfig
from ..preprocessing.graph_subset import build_controlled_graph_subset
from ..preprocessing.pairing import build_centered_genre_profiles
from ..shared.io import load_movies, write_csv_gzip, write_json
from .lightgcn_training import LightGCNTrainingConfig, train_lightgcn
from .scaled_pair_experiment import (
    _bootstrap_comparisons,
    _evaluate_three_methods,
    select_disjoint_pairs,
)
from .validation_experiment import model_artifact, selection_key, sha256_file


def pretest_eligible_user_ids(
    statistics: pd.DataFrame,
    *,
    min_train_ratings: int,
    min_train_positives: int,
    min_validation_positives: int,
    excluded_user_ids: np.ndarray,
) -> np.ndarray:
    """Select active users without reading any post-validation activity field."""

    required = {
        "userId", "train_interactions", "train_positives", "validation_positives",
    }
    missing = required - set(statistics.columns)
    if missing:
        raise ValueError(f"User statistics are missing columns: {sorted(missing)}")
    mask = (
        statistics["train_interactions"].ge(min_train_ratings)
        & statistics["train_positives"].ge(min_train_positives)
        & statistics["validation_positives"].ge(min_validation_positives)
        & ~statistics["userId"].isin(np.asarray(excluded_user_ids, dtype=np.int64))
    )
    return np.sort(statistics.loc[mask, "userId"].to_numpy(dtype=np.int64))


def random_disjoint_pairs(
    user_ids: np.ndarray,
    *,
    pair_count: int,
    random_seed: int,
) -> pd.DataFrame:
    """Shuffle eligible users once and pair adjacent users without replacement."""

    users = np.unique(np.asarray(user_ids, dtype=np.int64))
    if pair_count <= 0 or len(users) < 2 * pair_count:
        raise ValueError("At least two unique users per requested pair are required")
    rng = np.random.default_rng(random_seed)
    selected = rng.choice(users, size=2 * pair_count, replace=False).reshape(-1, 2)
    selected.sort(axis=1)
    result = pd.DataFrame(selected, columns=["userA", "userB"])
    result.insert(0, "pairId", np.arange(pair_count, dtype=np.int32))
    return result


def add_training_pair_features(
    pairs: pd.DataFrame,
    train_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    preprocessing: PreprocessingConfig,
) -> pd.DataFrame:
    """Describe random pairs using similarities that never affect their selection."""

    users = np.sort(np.unique(pairs[["userA", "userB"]].to_numpy().ravel()))
    profiles = build_centered_genre_profiles(
        train_ratings, movies, users, preprocessing,
    )
    user_position = {int(user): index for index, user in enumerate(users)}
    rating_maps = {
        int(user): dict(zip(group["movieId"].astype(int), group["rating"].astype(float)))
        for user, group in train_ratings.groupby("userId", sort=False)
    }
    rows = []
    for pair in pairs.itertuples(index=False):
        user_a, user_b = int(pair.userA), int(pair.userB)
        genre_similarity = float(
            profiles[user_position[user_a]] @ profiles[user_position[user_b]]
        )
        ratings_a, ratings_b = rating_maps[user_a], rating_maps[user_b]
        common = sorted(ratings_a.keys() & ratings_b.keys())
        correlation = float("nan")
        if len(common) >= preprocessing.pair_min_co_rated:
            values_a = np.fromiter((ratings_a[movie] for movie in common), dtype=float)
            values_b = np.fromiter((ratings_b[movie] for movie in common), dtype=float)
            if values_a.std() > 0 and values_b.std() > 0:
                correlation = float(np.corrcoef(values_a, values_b)[0, 1])
        rows.append({
            "pairId": int(pair.pairId),
            "coRatedTrain": len(common),
            "genreSimilarity": genre_similarity,
            "ratingCorrelation": correlation,
        })
    return pairs.merge(
        pd.DataFrame(rows), on="pairId", how="left", validate="one_to_one",
    )


def similarity_band_rule(development_pairs: pd.DataFrame) -> dict:
    """Freeze descriptive similarity bands from development pairs only."""

    similarities = development_pairs["genreSimilarity"].dropna()
    if similarities.empty:
        raise ValueError("Development pairs need genre similarities")
    lower, upper = similarities.quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
    return {
        "feature": "pre-2019 centered genre-profile cosine similarity",
        "cutoffs_source": "development pairs only",
        "lower_tertile": float(lower),
        "upper_tertile": float(upper),
        "labels": {
            "high_conflict": f"similarity <= {lower:.6f}",
            "mixed": f"{lower:.6f} < similarity <= {upper:.6f}",
            "similar": f"similarity > {upper:.6f}",
        },
        "role": "post-hoc descriptive analysis; never used to select pairs or models",
    }


def assign_similarity_bands(pairs: pd.DataFrame, rule: dict) -> pd.Series:
    """Apply development-frozen genre-similarity cutoffs to any pair cohort."""

    similarities = pairs["genreSimilarity"]
    lower = float(rule["lower_tertile"])
    upper = float(rule["upper_tertile"])
    labels = np.select(
        [similarities.le(lower), similarities.le(upper)],
        ["high_conflict", "mixed"],
        default="similar",
    )
    return pd.Series(labels, index=pairs.index, name="similarityBand")


def similarity_subgroup_summary(
    metrics: pd.DataFrame,
    pairs: pd.DataFrame,
    rule: dict,
    *,
    k: int,
) -> pd.DataFrame:
    """Summarize pair-level relevance by pre-frozen descriptive subgroup."""

    pair_bands = pairs[["pairId"]].copy()
    pair_bands["similarityBand"] = assign_similarity_bands(pairs, rule)
    joined = metrics.merge(pair_bands, on="pairId", how="left", validate="many_to_one")
    if joined["similarityBand"].isna().any():
        raise ValueError("Metrics contain pair IDs absent from the frozen pair cohort")
    return (
        joined.groupby(["similarityBand", "method"], sort=False)
        .agg(
            evaluatedPairs=("pairId", "nunique"),
            meanMinimumNDCG=(f"minimumNDCG@{k}", "mean"),
            meanAverageNDCG=(f"averageNDCG@{k}", "mean"),
            meanMinimumRecall=(f"minimumRecall@{k}", "mean"),
            meanAverageRecall=(f"averageRecall@{k}", "mean"),
        )
        .reset_index()
    )


def _prior_evaluated_users(pair_path: Path) -> np.ndarray:
    """Reproduce and exclude every user from the earlier 100/500-pair cohorts."""

    pairs = pd.read_csv(pair_path, usecols=["userA", "userB"])
    original = pairs.head(100)
    scaled = select_disjoint_pairs(pairs, pair_limit=500)
    return np.unique(pd.concat([
        original["userA"], original["userB"], scaled["userA"], scaled["userB"],
    ]).to_numpy(dtype=np.int64))


def _verify_hashes(root: Path, manifest: dict) -> None:
    for relative, expected in manifest["file_sha256"].items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"Frozen final-holdout file changed: {relative}")


def prepare_final_holdout(
    *,
    data_dir: Path,
    processed_dir: Path,
    output_root: Path,
    config_path: Path,
    preprocessing_path: Path,
) -> dict:
    """Freeze users, pairs, and graph without loading any 2020+ feedback."""

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("Use a new final-holdout output directory")
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    preprocessing = PreprocessingConfig.from_json(preprocessing_path)
    cohort = settings["cohort"]
    total_pairs = int(cohort["development_pairs"]) + int(cohort["holdout_pairs"])
    if int(cohort["total_user_limit"]) < 2 * total_pairs:
        raise ValueError("The graph cannot drop any development or holdout pair member")
    pair_path = processed_dir / "dissimilar_pairs.csv.gz"
    excluded = _prior_evaluated_users(pair_path)
    statistics_path = processed_dir / "user_split_statistics.csv.gz"
    statistics = pd.read_csv(
        statistics_path,
        usecols=[
            "userId", "train_interactions", "train_positives", "validation_positives",
        ],
    )
    eligible = pretest_eligible_user_ids(
        statistics,
        min_train_ratings=int(preprocessing.evaluation_min_train_ratings),
        min_train_positives=int(preprocessing.evaluation_min_train_positives),
        min_validation_positives=int(cohort["validation_min_positives"]),
        excluded_user_ids=excluded,
    )
    random_pairs = random_disjoint_pairs(
        eligible,
        pair_count=total_pairs,
        random_seed=int(cohort["random_seed"]),
    )
    selected_users = np.unique(
        random_pairs[["userA", "userB"]].to_numpy().ravel()
    )
    print(
        f"Loading pre-2019 ratings for {len(selected_users)} randomly paired users...",
        flush=True,
    )
    train_ratings = load_focus_ratings(
        data_dir / "ratings.csv",
        selected_users,
        before_timestamp=preprocessing.train_end_timestamp,
    )
    if (train_ratings["timestamp"] >= preprocessing.train_end_timestamp).any():
        raise AssertionError("Future feedback entered final pair construction")
    print("Describing random pairs from training ratings only...", flush=True)
    disjoint = add_training_pair_features(
        random_pairs,
        train_ratings,
        load_movies(data_dir),
        preprocessing,
    )
    development_count = int(cohort["development_pairs"])
    development = disjoint.head(development_count).copy().reset_index(drop=True)
    holdout = disjoint.iloc[development_count:].copy().reset_index(drop=True)
    development["pairId"] = np.arange(len(development), dtype=np.int32)
    holdout["pairId"] = np.arange(len(holdout), dtype=np.int32)
    all_members = pd.concat([
        development["userA"], development["userB"],
        holdout["userA"], holdout["userB"],
    ])
    if all_members.duplicated().any():
        raise AssertionError("Development and holdout users must be globally disjoint")
    edge_path = processed_dir / "train_positive_edges.csv.gz"
    edges = pd.read_csv(edge_path, dtype={"userId": "int32", "movieId": "int32"})
    subset = build_controlled_graph_subset(
        edges,
        disjoint.drop(columns="pairId"),
        pair_limit=len(disjoint),
        total_user_limit=int(cohort["total_user_limit"]),
        random_seed=int(cohort["random_seed"]),
    )
    catalogue = (
        subset.positive_edges.groupby("movieId").size()
        .rename("trainPositiveCount")
        .reset_index()
    )
    output_root.mkdir(parents=True)
    write_json(settings, output_root / "resolved_config.json")
    write_json(preprocessing.to_dict(), output_root / "preprocessing_config.json")
    write_csv_gzip(development, output_root / "development_pairs.csv.gz")
    write_csv_gzip(holdout, output_root / "holdout_pairs.csv.gz")
    write_csv_gzip(subset.positive_edges, output_root / "train_positive_edges.csv.gz")
    write_csv_gzip(catalogue, output_root / "training_catalog.csv.gz")
    protected = (
        "resolved_config.json", "preprocessing_config.json",
        "development_pairs.csv.gz", "holdout_pairs.csv.gz",
        "train_positive_edges.csv.gz", "training_catalog.csv.gz",
    )
    protocol = {
        "status": "prepared_without_loading_2020_plus_feedback",
        "cohort_rule": (
            "Random member-disjoint pairing among users eligible from pre-test activity; "
            "pair similarities use pre-2019 ratings for description only; all users from "
            "earlier evaluated pair cohorts are excluded."
        ),
        "eligible_users_before_pairing": len(eligible),
        "excluded_prior_users": len(excluded),
        "pair_sampling": {
            "method": "uniform random users without replacement, paired adjacently",
            "random_seed": int(cohort["random_seed"]),
            "similarity_not_used_for_selection": True,
            "pairs_with_reliable_rating_correlation": int(
                disjoint["ratingCorrelation"].notna().sum()
            ),
            "genre_similarity_summary": {
                key: float(value) for key, value in
                disjoint["genreSimilarity"].describe().to_dict().items()
            },
        },
        "development_pairs": len(development),
        "holdout_pairs": len(holdout),
        "globally_disjoint_pair_users": int(all_members.nunique()),
        "graph": {
            "users": subset.graph.num_users,
            "movies": subset.graph.num_movies,
            "edges": subset.graph.num_positive_edges,
        },
        "file_sha256": {
            relative: sha256_file(output_root / relative) for relative in protected
        },
        "source_sha256": {
            "train_positive_edges.csv.gz": sha256_file(edge_path),
            "user_split_statistics.csv.gz": sha256_file(statistics_path),
        },
    }
    write_json(protocol, output_root / "protocol.json")
    print(
        f"Final cohorts frozen: {len(development)} development and "
        f"{len(holdout)} holdout pairs",
        flush=True,
    )
    return protocol


def train_final_holdout_models(
    *,
    data_dir: Path,
    output_root: Path,
) -> dict:
    """Use development users only for checkpoint selection and freeze artifacts."""

    selection_path = output_root / "selection.json"
    if selection_path.exists():
        raise FileExistsError("Final-holdout models are already trained")
    protocol = json.loads((output_root / "protocol.json").read_text(encoding="utf-8"))
    _verify_hashes(output_root, protocol)
    settings = json.loads((output_root / "resolved_config.json").read_text(encoding="utf-8"))
    preprocessing = PreprocessingConfig.from_json(output_root / "preprocessing_config.json")
    development = pd.read_csv(output_root / "development_pairs.csv.gz")
    edges = pd.read_csv(output_root / "train_positive_edges.csv.gz")
    graph = build_bipartite_graph(edges)
    catalogue = pd.read_csv(output_root / "training_catalog.csv.gz")
    users = np.unique(development[["userA", "userB"]].to_numpy().ravel())
    validation_ratings = load_focus_ratings(
        data_dir / "ratings.csv",
        users,
        before_timestamp=preprocessing.validation_end_timestamp,
    )
    context = build_temporal_context(
        development,
        validation_ratings,
        graph.movie_ids,
        preprocessing,
        period="validation",
    )
    torch.set_num_threads(int(settings["torch_threads"]))
    torch.use_deterministic_algorithms(True)
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
            context.pairs,
            artifact["user_ids"],
            artifact["user_embeddings"],
            artifact["movie_ids"],
            artifact["movie_embeddings"],
            context.seen,
            conflict_weight=0.0,
            k=int(settings["k"]),
        )
        _, summary = evaluate_shared_rankings(
            recs,
            context.relevance,
            catalog_size=context.catalog_size,
            k=int(settings["k"]),
        )
        key = selection_key(summary, int(settings["k"]))
        print(
            f"Final LightGCN step={step}: val min-NDCG={key[0]:.6f}, "
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
        selection_smoothing=int(settings["lightgcn"].get("selection_smoothing", 1)),
    )
    lightgcn_elapsed = time.perf_counter() - started
    artifact = model_artifact(lightgcn, graph)
    lightgcn_path = output_root / "lightgcn_embeddings.npz"
    np.savez_compressed(lightgcn_path, **artifact)
    write_csv_gzip(history, output_root / "lightgcn_training_history.csv.gz")
    print("Fitting final ItemKNN...", flush=True)
    started = time.perf_counter()
    item_model = fit_item_knn(
        graph,
        max_neighbors=int(settings["item_knn"]["neighbors"]),
        shrinkage=float(settings["item_knn"]["shrinkage"]),
    )
    item_elapsed = time.perf_counter() - started
    item_path = output_root / "item_knn.npz"
    save_item_knn(item_model, item_path)
    item_scores = score_item_knn_users(
        item_model,
        graph,
        users,
        neighbors=int(settings["item_knn"]["neighbors"]),
    )
    recommendations, metrics, summaries = _evaluate_three_methods(
        context=context,
        catalogue=catalogue,
        lightgcn_artifact=artifact,
        item_scores=item_scores,
        item_user_ids=users,
        movie_ids=graph.movie_ids,
        k=int(settings["k"]),
    )
    write_csv_gzip(recommendations, output_root / "development_recommendations.csv.gz")
    write_csv_gzip(metrics, output_root / "development_pair_metrics.csv.gz")
    pd.DataFrame(summaries).to_csv(output_root / "development_comparison.csv", index=False)
    boundary = int(history.attrs["best_step"]) == int(training.steps)
    protected = ("lightgcn_embeddings.npz", "item_knn.npz")
    selection = {
        "status": (
            "training_budget_boundary" if boundary else "frozen_before_holdout_test"
        ),
        "holdout_labels_loaded": False,
        "selection_rule": (
            "Fixed prior hyperparameters; select LightGCN checkpoint on development-user "
            "2019 minimum NDCG@10, then average NDCG@10."
        ),
        "lightgcn": {
            "config": asdict(training),
            **dict(history.attrs),
            "elapsed_seconds": lightgcn_elapsed,
        },
        "item_knn": {
            **settings["item_knn"],
            "elapsed_seconds": item_elapsed,
        },
        "development_protocol": context.diagnostics,
        "development_methods": summaries,
        "similarity_subgroup_rule": similarity_band_rule(development),
        "file_sha256": {
            relative: sha256_file(output_root / relative) for relative in protected
        },
        "boundary_action": (
            "Increase the training budget and retrain before loading holdout labels."
            if boundary else "The selected checkpoint is interior; holdout evaluation is allowed."
        ),
    }
    write_json(selection, selection_path)
    print(
        f"Final training status: {selection['status']}; best step "
        f"{history.attrs['best_step']} of {training.steps}",
        flush=True,
    )
    return selection


def evaluate_final_holdout_once(
    *,
    data_dir: Path,
    output_root: Path,
) -> dict:
    """Load 2020+ holdout feedback once after every method is frozen."""

    report_path = output_root / "holdout/holdout_report.json"
    if report_path.exists():
        raise FileExistsError("Final holdout was already evaluated and cannot be overwritten")
    protocol = json.loads((output_root / "protocol.json").read_text(encoding="utf-8"))
    selection = json.loads((output_root / "selection.json").read_text(encoding="utf-8"))
    _verify_hashes(output_root, protocol)
    _verify_hashes(output_root, selection)
    if selection["status"] != "frozen_before_holdout_test":
        raise RuntimeError(selection["boundary_action"])
    settings = json.loads((output_root / "resolved_config.json").read_text(encoding="utf-8"))
    preprocessing = PreprocessingConfig.from_json(output_root / "preprocessing_config.json")
    holdout = pd.read_csv(output_root / "holdout_pairs.csv.gz")
    edges = pd.read_csv(output_root / "train_positive_edges.csv.gz")
    graph = build_bipartite_graph(edges)
    catalogue = pd.read_csv(output_root / "training_catalog.csv.gz")
    users = np.unique(holdout[["userA", "userB"]].to_numpy().ravel())
    ratings = load_focus_ratings(data_dir / "ratings.csv", users)
    context = build_temporal_context(
        holdout, ratings, graph.movie_ids, preprocessing, period="test",
    )
    with np.load(output_root / "lightgcn_embeddings.npz", allow_pickle=False) as archive:
        artifact = {key: archive[key] for key in archive.files}
    item_model = load_item_knn(output_root / "item_knn.npz")
    item_scores = score_item_knn_users(
        item_model,
        graph,
        users,
        neighbors=int(settings["item_knn"]["neighbors"]),
    )
    recommendations, metrics, summaries = _evaluate_three_methods(
        context=context,
        catalogue=catalogue,
        lightgcn_artifact=artifact,
        item_scores=item_scores,
        item_user_ids=users,
        movie_ids=graph.movie_ids,
        k=int(settings["k"]),
    )
    bootstrap = _bootstrap_comparisons(metrics, settings, k=int(settings["k"]))
    subgroup = similarity_subgroup_summary(
        metrics,
        holdout,
        selection["similarity_subgroup_rule"],
        k=int(settings["k"]),
    )
    holdout_root = output_root / "holdout"
    holdout_root.mkdir(parents=True)
    write_csv_gzip(recommendations, holdout_root / "recommendations.csv.gz")
    write_csv_gzip(metrics, holdout_root / "pair_metrics.csv.gz")
    pd.DataFrame(summaries).to_csv(holdout_root / "comparison.csv", index=False)
    subgroup.to_csv(holdout_root / "similarity_subgroups.csv", index=False)
    report = {
        "purpose": "Frozen user-disjoint holdout result for the final report",
        "cohort_selection": protocol["cohort_rule"],
        "methods_frozen_before_holdout": [
            "popularity", "lightgcn_average", "item_knn_average",
        ],
        "protocol": context.diagnostics,
        "methods": summaries,
        "paired_bootstrap": bootstrap,
        "similarity_subgroup_rule": selection["similarity_subgroup_rule"],
        "similarity_subgroups": subgroup.to_dict(orient="records"),
        "limitations": [
            "Synthetic pairs and individual future ratings approximate joint satisfaction.",
            "The graph contains 5,000 users rather than all MovieLens 32M users.",
            "The same 2020+ calendar period was examined for earlier, excluded users; "
            "this holdout cohort itself is user-disjoint and was frozen without test activity.",
            "Missing ratings remain unknown rather than confirmed dislikes.",
            "One pairing seed and one model-training seed were used.",
        ],
    }
    write_json(report, report_path)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    return report
