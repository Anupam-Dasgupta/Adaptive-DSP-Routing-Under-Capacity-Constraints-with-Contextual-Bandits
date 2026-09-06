# Implementation Plan

## Objective

Study a contextual semi-bandit-style routing problem with partial feedback and
per-DSP capacity constraints. Each Avazu row is an incoming request. A policy may
forward that request independently to multiple synthetic DSPs, pays a DSP-specific
serving cost, and observes only the outcomes of selected DSPs.

## Data inspection (2026-09-06)

The repository contained two Parquet files produced by `load_data.ipynb` from the
Hugging Face dataset `EvgeniaKyriazi/ctr-prediction-dataset`:

| File | Rows | Columns | Size |
|---|---:|---:|---:|
| `avazu_train.parquet` | 800,000 | 27 | 31.5 MB |
| `avazu_test.parquet` | 200,000 | 27 | 8.4 MB |

Both partitions have the same schema, no missing values, a click rate of 0.17488
when combined, and timestamps from 2014-10-21 00:00 through 2014-10-30 05:00.
There are 56 distinct hourly timestamps across 10 dates.

The supplied partitions are **not chronological**: each spans the entire time range.
`__index_level_0__` is unique across their union and covers 0 through 999,999, but
that source index is not chronological either. Therefore the loader concatenates the
partitions and performs a stable sort by `(datetime, __index_level_0__)`. Every
chronological model split is made only after that reconstruction; rows are never
randomly shuffled.

### Observed schema and cardinality

Counts below are exact unique counts in the supplied 800k / 200k partitions.

| Column | Parquet dtype | Train / test unique | Decision |
|---|---|---:|---|
| `id` | float64 | 800,000 / 200,000 | Exclude: unique identifier; conversion to float also loses original string precision |
| `click` | int64 | 2 / 2 | Exclude: post-event label and not a DSP reward |
| `hour` | int64 | 56 / 56 | Validate against `datetime`, then replace with cyclic time features |
| `datetime` | timestamp[ns] | 56 / 56 | Ordering plus cyclic features; do not encode raw timestamp |
| `date` | date32 | 10 / 10 | Redundant with `datetime`; exclude |
| `__index_level_0__` | int64 | 800,000 / 200,000 | Stable tie-break only; exclude from features |
| `C1` | int64 | 7 / 7 | Categorical feature |
| `banner_pos` | int64 | 7 / 7 | Categorical feature |
| `site_id` | string | 2,414 / 1,734 | Initially exclude to control dimension |
| `site_domain` | string | 2,600 / 1,635 | Initially exclude to control dimension |
| `site_category` | string | 20 / 20 | Categorical, capped to frequent levels |
| `app_id` | string | 2,617 / 1,645 | Initially exclude to control dimension |
| `app_domain` | string | 171 / 110 | Initially exclude to control dimension |
| `app_category` | string | 25 / 23 | Categorical, capped to frequent levels |
| `device_id` | string | 109,718 / 31,982 | Exclude: near-identifier and too sparse |
| `device_ip` | string | 398,229 / 133,257 | Exclude: near-identifier and too sparse |
| `device_model` | string | 4,849 / 3,723 | Initially exclude to control dimension |
| `device_type` | int64 | 4 / 4 | Categorical feature |
| `device_conn_type` | int64 | 4 / 4 | Categorical feature |
| `C14` | int64 | 1,770 / 1,487 | Initially exclude to control dimension |
| `C15` | int64 | 8 / 8 | Log-scaled width feature |
| `C16` | int64 | 9 / 9 | Log-scaled height and aspect-ratio features |
| `C17` | int64 | 354 / 335 | Initially exclude to control dimension |
| `C18` | int64 | 4 / 4 | Categorical feature |
| `C19` | int64 | 62 / 59 | Initially exclude |
| `C20` | int64 | 162 / 157 | Initially exclude (`-1` also appears) |
| `C21` | int64 | 54 / 52 | Initially exclude |

The feature set uses an intercept; sine/cosine encodings for hour-of-day and
day-of-week; log width, log height, and aspect ratio; plus bounded one-hot encoding
of `C1`, `banner_pos`, `site_category`, `app_category`, `device_type`,
`device_conn_type`, and `C18`. Each categorical field contributes at most eight
columns, so total dimension is at most 64 and is about 50 on this data. The encoder
is fit on the chronological training prefix only. Unseen evaluation levels map to
the infrequent bucket or an all-zero block.

## Implementation phases

### Phase 1 — data and contracts (implemented)

- Validate required files and schema before reading.
- Recombine partitions, validate `hour`/`datetime`, stable-sort, and take a
  configurable chronological prefix.
- Create train-only-fitted sparse features with a strict leakage deny-list.
- Add deterministic chronological splitting and data/feature tests.

### Phase 2 — controlled environment (implemented)

- Generate paired synthetic DSP rewards from a fixed seed.
- Keep full counterfactual expected/realized matrices inside the environment only.
- Add heterogeneous costs, abrupt drift, and per-window hard capacity.
- Add a zero-inflated two-stage buy/value model for robustness experiments.

### Phase 3 — policies and evaluation (implemented)

- Baselines: ForwardAll and seeded Random.
- Contextual Greedy and LinUCB with efficient Sherman–Morrison updates.
- Batched Discounted LinUCB for non-stationary rewards.
- Interpretable per-DSP shadow-price pacing, always backed by the hard limiter.
- Evaluation loop reveals only selected DSP outcomes.
- Capacity-constrained, non-causal expected-reward oracle for pseudo-regret.

### Phase 4 — experiments (implemented)

- Stationary paired comparison.
- Exploration-strength, capacity, and routing-cost sweeps.
- Abrupt-drift comparison and discount-factor sensitivity.
- Aggregate paired seeds using mean and standard deviation.

The final laptop-sized suite uses 100,000 order-preserving samples across the full
ten-day source period, an 80,000-event evaluation suffix, and three paired seeds. It
includes capacity ratios 0.2/0.4/0.6 and discount factors
0.98/0.995/0.999/0.9995.

### Phase 5 — reporting (implemented)

- Cumulative realized profit and pseudo-regret.
- Gross value, serving cost, utilization, routing efficiency, and zero violations.
- Rolling post-drift performance with an explicit recovery definition.
- Interview-facing explanation and resume bullets only after measured results exist.

## Scientific invariants

1. Context order is chronological and never randomly shuffled.
2. `click` and downstream outcomes never enter the feature matrix.
3. Policy updates receive only rewards for routed DSPs.
4. Hard capacity limits cannot be overridden by a policy or pacing heuristic.
5. All policies compared under one seed share the same generated world.
6. Pseudo-regret uses true expected net reward; reported profit uses sampled realized
   reward.
7. Avazu supplies contexts only; DSP behavior and counterfactual outcomes are openly
   documented as synthetic.

## Intentional deviations

- The downloaded files were already Parquet, so they are moved under `data/raw/`
  without rewriting or duplicating them.
- Their `train`/`test` names are not honored as an experimental split because doing
  so would leak future timestamps into both sides. They are source partitions only.
- High-cardinality identifiers are omitted from the first pipeline rather than
  densely one-hot encoded. Controlled hashing can be tested later as an ablation.
- Development runs should use 2k–50k events. The planned 300k–500k runs wait until
  correctness and runtime are established.
