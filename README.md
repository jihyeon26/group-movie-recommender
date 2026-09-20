# Group Movie Recommender

A reproducible MovieLens 32M experiment that recommends one shared Top-10 list to
two users, comparing global popularity, ItemKNN, and LightGCN on randomly
sampled, member-disjoint pairs.

**Research question.** Do LightGCN and ItemKNN improve NDCG@10 over a popularity baseline for two-user group recommendation, particularly for the less-satisfied member, and what trade-offs do they create in recall and catalogue coverage?

## Result

Frozen holdout of 500 random pairs, scored on the 195 pairs where both members
have an eligible positive future item.

| Method | Min NDCG@10 | Avg NDCG@10 | Min Recall@10 | Avg Recall@10 | Coverage@10 |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.0260 | 0.0959 | 0.0067 | 0.0376 | 0.0173 |
| LightGCN | 0.0306 | 0.1088 | **0.0084** | **0.0468** | **0.0358** |
| ItemKNN | **0.0335** | **0.1100** | 0.0074 | 0.0419 | 0.0202 |

Both personalized methods beat popularity on average NDCG@10 — ItemKNN by
`+0.01414` (95% CI `[+0.00398, +0.02456]`), LightGCN by `+0.01294`
(`[+0.00035, +0.02559]`) — and are statistically indistinguishable from each
other. ItemKNN leads on ranking quality, LightGCN on recall and coverage.

Minimum-member NDCG@10 is zero for 82 to 85 per cent of evaluable pairs, so the
evidence about the worse-off member is weak: of the six reported comparisons,
only ItemKNN over popularity on average NDCG@10 survives a Bonferroni
correction.

## Protocol

| | |
|---|---|
| Data | MovieLens 32M; a rating of at least 4.0 is a positive interaction |
| Split | train before 2019, development 2019, holdout 2020 onward |
| Pairs | 300 development and 500 holdout, uniform without replacement, member-disjoint |
| Graph | 5,000 users, 16,923 warm movies, 706,501 positive train edges |
| Candidates | warm graph movies unrated by either member |
| Metrics | min (primary) and average NDCG@10, min/average Recall@10, coverage@10 |
| Uncertainty | paired bootstrap, 10,000 resamples |

Pair similarity is descriptive only and never selects pair members.

LightGCN uses 32 dimensions, 2 layers, batch size 8,192, and learning rate 0.02,
with the checkpoint selected on a trailing mean of three validation scores —
step 700 of 2,500, or 8.12 epochs. `configs/final_holdout_experiment.json` keeps
the superseded budget, whose single-estimate stopping rule ended training after
0.58 epochs.

## Setup

Python 3.11 or newer, using `uv`:

```powershell
uv venv
uv pip install -e ".[training,dev]"
```

Download MovieLens 32M separately into the ignored `dataset/movie_lens32m/`
(`ratings.csv`, `movies.csv`, `links.csv`, `tags.csv`).

## Reproduce

```powershell
python scripts/check_data.py
python scripts/prepare_data.py

python scripts/run_final_holdout_experiment.py prepare --config configs/final_holdout_experiment_converged.json --output-root outputs/final_holdout_reproduction
python scripts/run_final_holdout_experiment.py train --output-root outputs/final_holdout_reproduction
python scripts/run_final_holdout_experiment.py test --output-root outputs/final_holdout_reproduction
```

Use a new, empty output root for each reproduction. The committed
`outputs/final_holdout_converged/` directory is the reference result snapshot
and is intentionally not overwritten.

The test phase verifies the frozen cohort and model hashes, refuses a checkpoint
selected at the training-budget boundary, and refuses to overwrite an existing
holdout report.

Every number in the tables above comes from
`outputs/final_holdout_converged/holdout/holdout_report.json`, which is
committed along with the pair-level metrics, the subgroup table, the training
history, and the frozen protocol and selection records. Model artifacts, graph
edges, and recommendation dumps are regenerable and stay out of Git.

## Layout

```text
configs/     Experiment and preprocessing configuration
scripts/     Command-line entry points
src/group_movie_recommender/
├── algorithms/     Popularity, ItemKNN, LightGCN, and group ranking
├── evaluation/     Temporal protocol, metrics, and uncertainty
├── filtering/      Warm catalogue and candidate filtering
├── pipelines/      Experiment orchestration
├── preprocessing/  Validation, splitting, and pair construction
└── shared/         Memory-aware I/O
tests/       Deterministic unit tests (`python -m pytest`)
```

Tests use small synthetic inputs and do not require MovieLens files.

## Limitations

- Pairs are synthetic; separate future ratings stand in for joint satisfaction.
- Only 195 of 500 holdout pairs are evaluable, which selects for active users.
- The 5,000-user graph is smaller than the full MovieLens 32M graph.
- Missing ratings are unknown, not confirmed dislikes.
- One pair-sampling seed and one training seed.

