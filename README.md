# Group Movie Recommender

This repository contains the reproducible preprocessing foundation for a conflict-aware movie recommender for two users with different tastes. The project uses MovieLens 32M and focuses on group fairness: improving the less-satisfied member without collapsing recommendations to a small set of safe, popular movies.

## Repository structure

```text
.
├── configs/
│   └── preprocessing.json       # Reproducible split and filtering thresholds
├── scripts/
│   ├── check_data.py            # Schema, missing-data, duplicate, and integrity checks
│   ├── prepare_data.py          # Split, graph, catalogue, pair, and export pipeline
│   ├── evaluate_popularity.py   # Baseline recommendations and evaluation
│   ├── evaluate_lightgcn_subset.py # Validation ranking integration check
│   ├── run_lightgcn_scale_experiment.py # Staged scale diagnostics
│   ├── train_lightgcn_subset.py # Controlled MovieLens integration run
│   └── train_lightgcn_toy.py    # CPU learning sanity check
├── src/group_movie_recommender/
│   ├── preprocessing/           # Validation, splitting, sampling, and pair construction
│   │   ├── config.py            # Typed preprocessing configuration
│   │   ├── data_check.py        # Source data validation and cleaning
│   │   ├── pairing.py           # Training-only synthetic pair construction
│   │   ├── graph_subset.py      # Reproducible focus/context user subsets
│   │   ├── sampling.py          # Sparse-profile interaction sampling
│   │   └── splitting.py         # Temporal split and user cohorts
│   ├── filtering/               # Warm catalogue and pair candidate filtering
│   │   ├── candidates.py        # Shared warm, unseen candidate sets
│   │   └── catalog.py           # Positive graph and warm-item filtering
│   ├── algorithms/              # Recommendation and scoring algorithms
│   │   ├── embedding_inference.py # Full-catalogue scores from saved embeddings
│   │   ├── graph_data.py        # Graph indexing and BPR negative sampling
│   │   ├── group_ranking.py     # Average and conflict-aware aggregation
│   │   ├── lightgcn.py          # Trainable PyTorch LightGCN
│   │   ├── lightgcn_math.py     # Propagation, scoring, and BPR loss mathematics
│   │   └── popularity.py        # Non-personalized popularity baseline
│   ├── evaluation/              # Offline group evaluation
│   │   ├── diagnostics.py       # Held-out joint-relevance diagnostics
│   │   └── metrics.py           # Member and shared-list ranking metrics
│   ├── pipelines/
│   │   ├── preprocessing.py     # End-to-end preprocessing orchestration
│   │   └── lightgcn_training.py # Small-graph mini-batch optimization
│   └── shared/
│       └── io.py                # Memory-aware MovieLens I/O
├── tests/                       # Small deterministic unit tests
├── outputs/                     # Generated files; ignored except for .gitkeep
├── app.py                       # Single-person movie-rating data collector
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

Install the optional PyTorch training dependency when working on LightGCN:

```powershell
uv pip install -e ".[training,dev]"
```

Activate the environment or select `.venv/Scripts/python.exe` as the notebook kernel.

## Collect ratings from one participant

The Streamlit app lets one participant rate a genre-diverse set drawn from popular
MovieLens 32M titles. Install the app dependency after preparing the data:

```powershell
uv venv tmp/app-venv
uv pip install --python tmp/app-venv/Scripts/python.exe -e ".[app]"
./tmp/app-venv/Scripts/streamlit.exe run app.py
```

The separate ignored environment avoids conflicts with notebooks or training jobs
that may have the main `.venv` open.

The participant can skip unseen titles, continue through unique 12-movie batches
for as long as desired, go back to earlier batches, or search the complete
MovieLens catalogue. Movie posters are shown using each title's MovieLens-to-TMDB
link so that similar titles are easier to recognize.

- Assign a different positive participant ID to every person.
- Only explicit half-star ratings from 0.5 to 5.0 are saved.
- "Not seen" choices remain missing and are never converted into dislikes.
- Every saved rating keeps the time at which it was entered.
- The sidebar download creates a CSV as soon as at least one movie is rated.

The download button exports `userId,movieId,rating,timestamp`, matching the
MovieLens ratings schema. The default participant ID is above the MovieLens 32M
user range and can be changed in the sidebar. Keep exported live ratings separate
from the historical offline benchmark until a retraining and time-split policy is
defined.

Poster images come from TMDB. The app contains the attribution required by TMDB
and uses a checked-in URL cache for the starter movies; searched titles are looked
up when selected and fall back to a local placeholder if no poster is available.

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
and samples reproducible BPR triples whose negatives are absent from the user's
positive training graph. Such movies can be unrated or rated below four; they are
not confirmed dislikes. Validation/test feedback is never used to sample them.

## Understand LightGCN mathematics

`notebooks/04_lightgcn_math_walkthrough.ipynb` applies symmetric degree
normalization, neighbor propagation, layer averaging, dot-product scoring, and BPR
loss with NumPy. It exposes the model mathematics before adding automatic
differentiation and a full training loop.

## Run the toy LightGCN trainer

The PyTorch learning check uses a synthetic graph and is not a validation result:

```powershell
python scripts/train_lightgcn_toy.py
```

Its ignored output contains the fixed training-batch loss before and after 100
updates. `notebooks/05_lightgcn_training_walkthrough.ipynb` explains embedding
parameters, automatic differentiation, optimization, and full-catalog scoring.

For the local verification, the broken existing `.venv` was preserved and a separate
CPU environment was created at `tmp/lightgcn-venv`. Select its
`Scripts/python.exe` as the notebook interpreter, or run:

```powershell
& .\tmp\lightgcn-venv\Scripts\python.exe scripts/train_lightgcn_toy.py
```

The prototype performs full-graph propagation at every optimization step. It has
not yet been profiled or tuned for MovieLens 32M, and does not perform validation
selection or test evaluation. See the official
[PyTorch sparse matrix multiplication documentation](https://docs.pytorch.org/docs/stable/generated/torch.sparse.mm.html)
for the differentiable propagation operation used here.

## Run the controlled MovieLens subset

After preprocessing, run a bounded real-data integration check:

```powershell
python scripts/train_lightgcn_subset.py
```

The default configuration retains the users in 25 selected pairs and samples
context users to a total of 500. It exports original IDs with final embeddings so
future scoring does not confuse MovieLens IDs with matrix row indices. The run is
explained in `notebooks/06_movielens_subset_walkthrough.ipynb`. Its training loss is
not used as a model-selection metric and its subset results are not comparable to
the full-catalogue popularity baseline.

## Evaluate subset rankings

After the controlled subset trainer has exported its embeddings, compare popularity,
LightGCN average aggregation, and several conflict penalties on validation data:

```powershell
python scripts/evaluate_lightgcn_subset.py
```

Every method receives the same subset catalogue and removes the union of the two
members' training histories. Validation positives are used only as labels. Relevance
is pair-specific: if one member has already seen a movie, that movie is unavailable
to the pair and is removed from both members' relevance denominators.

The ignored `outputs/lightgcn_subset_validation/` directory contains ranked lists,
per-pair metrics, and a JSON report. The report selects among the personalized
methods by minimum-member NDCG@10 and uses average NDCG@10 only as a tie-breaker.
`notebooks/07_validation_ranking_walkthrough.ipynb` explains the comparison and its
limitations. In particular, a small integration run with all methods tied at zero
on the primary metric cannot support a claim that one aggregation method is better.

## Scale the validation experiment deliberately

The scale runner separates three possible causes of sparse Top-10 results: catalogue
coverage, number of evaluated pairs, and training duration. Its default runs only
the inexpensive smoke stage:

```powershell
python scripts/run_lightgcn_scale_experiment.py --stages smoke
```

Run later stages explicitly when the smoke stage succeeds:

```powershell
python scripts/run_lightgcn_scale_experiment.py --stages coverage stability trained
```

Run the zero-propagation BPR matrix-factorization ablation on the same controlled
graph, pairs, embedding size, training steps, and candidates:

```powershell
python scripts/run_lightgcn_scale_experiment.py --stages bpr_mf
```

The ablation labels its methods `bpr_mf_average` and `bpr_mf_conflict_*`. When
both `trained` and `bpr_mf` outputs exist, the runner also writes paired bootstrap
comparisons to `outputs/lightgcn_scale_experiment/model_ablation_bootstrap.json`.

The stages are defined in `configs/lightgcn_scale_experiment.json`. Each stage gets
separate model and validation directories, and completed stages are combined in
`outputs/lightgcn_scale_experiment/scale_comparison.csv.gz`. The stages change one
main factor at a time: graph context, pair count, and training steps. They remain
validation experiments; test data must not be used to choose the scale or conflict
weight. See `notebooks/08_scale_experiment_walkthrough.ipynb` for the comparison.
Use `--summarize-only` to rebuild the comparison table without retraining.

## Preprocessing definition

- Ratings of at least 4.0 are positive interactions.
- Training contains ratings before 2019.
- Validation contains ratings from 2019.
- Test contains ratings from 2020 onward.
- Lower-activity users can contribute to graph training.
- The current stricter evaluation cohort uses activity thresholds from train,
  validation, and test. It is therefore suitable for exploratory analysis but is
  not an untouched confirmatory test cohort.
- Warm movies are defined only from positive training interactions.
- Synthetic pair features use training ratings only.
- Dissimilar pairs fall in the bottom quartile for both centered genre similarity and co-rating correlation.
- Test feedback has already been inspected for cohort filtering, pair diagnostics,
  and the standalone popularity result. Final reporting must disclose this rather
  than describe the current test period as sealed.

All thresholds are stored in `configs/preprocessing.json` rather than hard-coded in scripts.

## Tests

```powershell
python -m pytest
```

The tests use small synthetic tables and do not require the MovieLens files.
