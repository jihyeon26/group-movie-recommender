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
| LightGCN | 0.0306 | 0.1088 | **0.0084** | **0.0468** | **0.0358** |
| ItemKNN | **0.0335** | **0.1100** | 0.0074 | 0.0419 | 0.0202 |

Both personalized methods improve average NDCG@10 over popularity:

- ItemKNN by `+0.01414` (95% paired-bootstrap CI `[+0.00398, +0.02456]`);
- LightGCN by `+0.01294` (95% CI `[+0.00035, +0.02559]`).

The two are statistically indistinguishable from each other on both NDCG
outcomes (`+0.00120`, CI `[-0.01060, +0.01326]` on average NDCG@10). ItemKNN
holds a small lead on ranking quality; LightGCN attains the highest recall, the
broadest catalogue coverage (606 against 341 distinct items), and the lowest
share of pairs served nothing relevant (0.292 against 0.349).

Evidence on the primary minimum-member metric is weak for every comparison. It
is exactly zero for 83 to 87 per cent of evaluable pairs, only 48 of the 195
pairs return a non-zero value under any method, and of the six reported
comparisons only the ItemKNN advantage over popularity in average NDCG@10
survives a Bonferroni correction.

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

LightGCN was allowed a maximum of 2,500 steps at batch size 8,192, with
validation every 100 steps, patience 8, and checkpoint selection on a trailing
mean over three validation estimates. Training stopped at step 1,500 and
restored the interior best checkpoint from step 700, which is 8.12 epochs over
the 706,501 training edges.

An earlier configuration compared single validation estimates of
minimum-member NDCG@10, whose coefficient of variation across checkpoints is
0.14 against 0.04 for average NDCG@10. It selected a sampling spike and stopped
after 0.58 epochs. The defect was identified from the training and development
curves and corrected before the results above; see
`configs/final_holdout_experiment.json` for the superseded budget.

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
python scripts/run_final_holdout_experiment.py prepare --config configs/final_holdout_experiment_converged.json --output-root outputs/final_holdout_converged
python scripts/run_final_holdout_experiment.py train --output-root outputs/final_holdout_converged
python scripts/run_final_holdout_experiment.py test --output-root outputs/final_holdout_converged
```

Configuration is stored in `configs/final_holdout_experiment_converged.json`.
Generated data, model artifacts, recommendations, and reports are written under
`outputs/final_holdout_converged/` and excluded from Git. The cohort block is
identical to the superseded `configs/final_holdout_experiment.json`, so both
configurations freeze the same users, pairs, and graph.

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
- The primary metric is informative on 48 of 195 pairs, and only one of the
  six reported comparisons survives a Bonferroni correction.
- The same calendar test period had been inspected for earlier, excluded users;
  the final cohort is user-disjoint and frozen without test-activity filtering,
  but the test period is not globally untouched.
