# Group Movie Recommender

This repository contains the reproducible preprocessing foundation for a conflict-aware movie recommender for two users with different tastes. The project uses MovieLens 32M and focuses on group fairness: improving the less-satisfied member without collapsing recommendations to a small set of safe, popular movies.

## Repository structure

```text
.
├── configs/
│   └── preprocessing.json       # Reproducible split and filtering thresholds
├── scripts/
│   ├── check_data.py            # Schema, missing-data, duplicate, and integrity checks
│   └── prepare_data.py          # Split, graph, catalogue, pair, and export pipeline
├── src/group_movie_recommender/
│   ├── preprocessing/           # Validation, splitting, sampling, and pair construction
│   │   ├── config.py            # Typed preprocessing configuration
│   │   ├── data_check.py        # Source data validation and cleaning
│   │   ├── pairing.py           # Training-only synthetic pair construction
│   │   ├── sampling.py          # Sparse-profile interaction sampling
│   │   └── splitting.py         # Temporal split and user cohorts
│   ├── filtering/               # Warm catalogue and pair candidate filtering
│   │   ├── candidates.py        # Shared warm, unseen candidate sets
│   │   └── catalog.py           # Positive graph and warm-item filtering
│   ├── algorithms/              # Recommendation and scoring algorithms
│   │   ├── graph_data.py        # Graph indexing and BPR negative sampling
│   │   ├── group_ranking.py     # Average and conflict-aware aggregation
│   │   ├── lightgcn_math.py     # Propagation, scoring, and BPR loss mathematics
│   │   └── popularity.py        # Non-personalized popularity baseline
│   ├── evaluation/              # Offline group evaluation
│   │   ├── diagnostics.py       # Held-out joint-relevance diagnostics
│   │   └── metrics.py           # Member and shared-list ranking metrics
│   ├── pipelines/
│   │   └── preprocessing.py     # End-to-end preprocessing orchestration
│   └── shared/
│       └── io.py                # Memory-aware MovieLens I/O
├── tests/                       # Small deterministic unit tests
├── outputs/                     # Generated files; ignored except for .gitkeep
└── exploration/                 # Local exploratory notebooks; fully ignored
```

The local `dataset/`, `exploration/`, `outputs/`, `.venv/`, and `tmp/` directories are intentionally excluded from Git.

## Data setup

Download and extract MovieLens 32M so that the local directory contains:

```text
dataset/movie_lens32m/
├── ratings.csv
├── movies.csv
├── links.csv
└── tags.csv
```

The dataset itself must not be committed to this repository.

## Environment

Using `uv`:

```powershell
uv venv
uv pip install -e ".[dev]"
```

Activate the environment or select `.venv/Scripts/python.exe` as the notebook kernel.

## Run the pipeline

Validate the raw files first:

```powershell
python scripts/check_data.py
```

Then create the temporal split, positive graph, warm catalogue, and dissimilar pairs:

```powershell
python scripts/prepare_data.py
```

Generated artifacts are written under `outputs/processed_movielens32m/` and include:

- `train_positive_edges.csv.gz`
- `user_split_statistics.csv.gz`
- `evaluation_users.csv.gz`
- `warm_movies.csv.gz`
- `dissimilar_pairs.csv.gz`
- `manifest.json`

## Run the popularity baseline

After preprocessing, generate one shared Top-10 list per pair and evaluate it:

```powershell
python scripts/evaluate_popularity.py
```

The ignored `outputs/popularity_baseline/` directory will contain:

- `recommendations.csv.gz`
- `pair_metrics.csv.gz`
- `metrics.json`

The baseline ranks warm movies by positive training-interaction count and removes
movies already observed by either group member. The same list is evaluated against
each member's eligible test positives. See
`notebooks/01_popularity_and_metrics_walkthrough.ipynb` for a small worked example.

## Understand group ranking

`notebooks/02_group_ranking_walkthrough.ipynb` compares average aggregation with
conflict-aware aggregation on three inspectable candidate movies. It shows how the
conflict weight changes the shared ranking and the minimum-member metric before a
personalized model is introduced.

## Understand the graph input

`notebooks/03_graph_data_walkthrough.ipynb` converts original MovieLens IDs into
contiguous user and movie indices, constructs the undirected bipartite `edge_index`,
and samples reproducible BPR triples whose negative movies are unseen by the user.

## Understand LightGCN mathematics

`notebooks/04_lightgcn_math_walkthrough.ipynb` applies symmetric degree
normalization, neighbor propagation, layer averaging, dot-product scoring, and BPR
loss with NumPy. It exposes the model mathematics before adding automatic
differentiation and a full training loop.

## Preprocessing definition

- Ratings of at least 4.0 are positive interactions.
- Training contains ratings before 2019.
- Validation contains ratings from 2019.
- Test contains ratings from 2020 onward.
- Lower-activity users can contribute to graph training.
- The stricter main evaluation cohort is selected separately.
- Warm movies are defined only from positive training interactions.
- Synthetic pair features use training ratings only.
- Dissimilar pairs fall in the bottom quartile for both centered genre similarity and co-rating correlation.
- Test feedback is used only after pair selection to measure observable joint relevance.

All thresholds are stored in `configs/preprocessing.json` rather than hard-coded in scripts.

## Tests

```powershell
python -m pytest
```

The tests use small synthetic tables and do not require the MovieLens files.
