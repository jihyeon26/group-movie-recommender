# Group Movie Recommender

A reproducible MovieLens 32M experiment for recommending one shared Top-10
movie list to two users. The final comparison evaluates global popularity,
ItemKNN, and LightGCN on randomly sampled, member-disjoint user pairs.

## Research question

Under a temporal warm-catalogue evaluation on randomly paired MovieLens users,
do LightGCN or ItemKNN improve minimum-member NDCG@10 over global popularity,
and what trade-offs do they create in average NDCG@10 and catalogue coverage?

## Final result

The frozen holdout contains 500 random pairs. Pair-level relevance metrics are
reported on the 195 pairs with at least one eligible positive future item for
both members.

| Method | Min NDCG@10 | Avg NDCG@10 | Min Recall@10 | Avg Recall@10 | Coverage@10 |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.0260 | 0.0959 | 0.0067 | 0.0376 | 0.0173 |
| LightGCN | 0.0226 | 0.0848 | 0.0053 | 0.0310 | **0.0254** |
| ItemKNN | **0.0335** | **0.1100** | **0.0074** | **0.0419** | 0.0202 |

Compared with popularity, ItemKNN improves:

- minimum-member NDCG@10 by `+0.00750` (95% paired-bootstrap CI
  `[+0.00055, +0.01484]`);
- average NDCG@10 by `+0.01414` (95% CI `[+0.00398, +0.02457]`).

LightGCN recommends from a broader portion of the catalogue but does not improve
holdout relevance. The result therefore supports ItemKNN, not additional model
complexity, for this controlled task.

## Evaluation protocol

- Dataset: MovieLens 32M explicit ratings.
- Positive interaction: rating of at least 4.0.
- Train: ratings before 2019.
- Development: ratings from the 2019 calendar year.
- Holdout: ratings from 2020 onward.
- Pairs: 300 development pairs and 500 holdout pairs sampled uniformly without
  replacement; no user appears in more than one pair.
- Pair similarity: descriptive only and never used to select pair members.
- Graph: 5,000 users, 16,923 warm movies, and 706,501 positive train edges.
- Candidates: warm graph movies not previously rated by either member.
- Primary metric: minimum-member NDCG@10.
- Secondary metrics: average NDCG@10 and minimum/average Recall@10.
- Beyond-accuracy metric: catalogue coverage@10.
- Uncertainty: paired bootstrap over common evaluable pairs with 10,000
  resamples.

LightGCN was allowed a maximum of 2,500 steps with validation every 100 steps
and patience 5. Training stopped at step 1,300 and restored the interior best
checkpoint from step 800.

## Setup

Python 3.11 or newer is required. Using `uv`:

```powershell
uv venv
uv pip install -e ".[training,dev]"
```

Place MovieLens 32M under the ignored local data directory:

```text
dataset/movie_lens32m/
├── ratings.csv
├── movies.csv
├── links.csv
└── tags.csv
```

Download the dataset separately; do not commit it to the repository.

## Reproduce the final experiment

Validate and preprocess MovieLens:

```powershell
python scripts/check_data.py
python scripts/prepare_data.py
```

Prepare the frozen cohorts, train the models, and evaluate the holdout once:

```powershell
python scripts/run_final_holdout_experiment.py prepare
python scripts/run_final_holdout_experiment.py train
python scripts/run_final_holdout_experiment.py test
```

Configuration is stored in `configs/final_holdout_experiment.json`. Generated
data, model artifacts, recommendations, and reports are written under
`outputs/final_holdout_experiment/` and excluded from Git.

The test phase verifies hashes for the frozen cohort and model artifacts. It
refuses evaluation when LightGCN's selected checkpoint is at the training-budget
boundary and refuses to overwrite an existing holdout report.

## Repository structure

```text
configs/                         Experiment and preprocessing configuration
scripts/                         Command-line entry points
src/group_movie_recommender/
├── algorithms/                  Popularity, ItemKNN, LightGCN, and ranking
├── evaluation/                  Temporal protocol, metrics, and uncertainty
├── filtering/                   Warm catalogue and candidate filtering
├── pipelines/                   Reproducible experiment orchestration
├── preprocessing/               Validation, splitting, and pair construction
└── shared/                      Memory-aware I/O utilities
tests/                           Deterministic unit tests
```

Local datasets, generated outputs, exploratory notebooks, temporary files, and
report documents are intentionally ignored.

## Tests

```powershell
python -m pytest
```

The test suite uses small synthetic inputs and does not require MovieLens files.

## Limitations

- Pairs are synthetic and do not represent observed households or joint choices.
- Only 195 of 500 holdout pairs have observable two-sided relevance.
- The controlled graph is smaller than the full MovieLens 32M graph.
- Missing ratings are unknown rather than confirmed dislikes.
- One pair-sampling seed and one LightGCN training seed are used.
- The same calendar test period had been inspected for earlier, excluded users;
  the final cohort is user-disjoint and frozen without test-activity filtering,
  but the test period is not globally untouched.
