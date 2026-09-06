# Methodology

## Problem

At event `t`, the exchange receives an Avazu context `x_t` and independently decides
whether to forward it to each DSP `j`. A request may go to several DSPs. Routing to a
DSP costs `c_j`; not routing reveals no outcome for that DSP.

This is a contextual semi-bandit-style routing problem with partial feedback and
per-DSP capacity constraints. It is not a single-action multi-armed bandit.

## Context data

Avazu supplies realistic request contexts. Its `click` label is never a model input
or a DSP reward. The baseline ignores it, while the optional CTR extension uses it
only as a supervised target and evaluation label. Avazu cannot provide counterfactual
responses from several DSPs for the same request, so those responses are generated
by the simulator.

The downloaded train and test Parquet files are random source partitions. The loader
recombines them, checks that the integer and timestamp hour fields agree, then uses a
stable `(datetime, source_index)` sort. Train/evaluation splits are chronological
prefix/suffix splits. The prefix fits preprocessing; reported policy metrics use only
the later suffix.

The final experiment suite systematically thins the ordered stream to a configurable
number of rows spread across the full source period. It always retains the first and
last event and never shuffles the sequence.

The reported run uses 100,000 contexts across 2014-10-21 through 2014-10-30. The
first 20% fit preprocessing; the remaining 80,000 events form the policy stream.

The encoder is fit on the training prefix only. It produces:

- an intercept;
- sine and cosine for hour of day and day of week;
- log ad width, log ad height, and aspect ratio;
- bounded one-hot encodings for seven low-cardinality context fields.

Each categorical field contributes at most eight columns, keeping the total at 64 or
fewer. `click`, `id`, raw time fields, and the source index are explicitly excluded.

## CTR-anchored extension

The optional extension fits logistic regression on the feature matrix from the
chronological training prefix, with Avazu `click` used only as its target. It reports
ROC-AUC, log loss, Brier score, and a quantile-binned calibration curve on the later
evaluation suffix.

The resulting out-of-sample probability is appended as one routing feature and used
as the shared relevance prior in a new DSP world. Each DSP still has separate buying
and conditional-value preferences, serving cost, and sampled feedback. This makes
the simulated world respond to an observed data signal without pretending that one
Avazu click is a counterfactual outcome from every DSP.

The feature ablation holds the generated world, costs, capacities, seeds, policy
hyperparameters, and event order fixed. Greedy and LinUCB are each evaluated with the
original feature matrix and with the appended CTR score. Paired profit gain and
pseudo-regret reduction therefore isolate the value of exposing that score to the
router; they do not measure the separate effect of changing the reward simulator.

## Synthetic DSP worlds

The linear world computes one latent score per DSP and converts it to a positive
expected gross value with `softplus`. Realized gross value adds seeded noise.

The two-stage world separates buying and value:

```text
p(t,j) = sigmoid(beta_j · x_t)
buy(t,j) ~ Bernoulli(p(t,j))
value(t,j) = softplus(phi_j · x_t + noise)
gross(t,j) = buy(t,j) * value(t,j)
net(t,j) = gross(t,j) - cost_j
```

For abrupt drift, half of the DSP coefficient vectors change halfway through the
stream. The policy is not told when this happens.

One seed generates the complete paired world before policies are compared. Every
policy under that seed therefore faces the same contexts, latent DSPs, drift, and
sampled outcomes.

## Policies

ForwardAll requests every DSP and relies on the hard limiter. Random requests each
DSP independently with a seeded probability.

Greedy and LinUCB maintain a separate ridge model per DSP. LinUCB routes on the score

```text
theta_hat_j · x + alpha * sqrt(x^T A_j^-1 x) - shadow_price_j
```

Greedy uses `alpha = 0`. Ordinary updates use the Sherman–Morrison identity, avoiding
a fresh matrix inverse after every observed reward.

Discounted LinUCB downweights older sufficient statistics by `gamma`. It batches
observations and refreshes the matrices periodically, keeping matrix inversions off
the per-request path. `gamma = 1` removes forgetting; smaller values adapt faster but
retain less history.

## Capacity and pacing

Each DSP has a hard route limit in every fixed event window. No policy can bypass
that limiter.

The optional pacing controller maintains a non-negative shadow price:

```text
lambda_j = max(0, lambda_j + eta * (action_j - target_rate_j))
```

Routing faster than the capacity target raises the threshold; routing slower lowers
it. This is an experimental heuristic, not a claim about any production exchange.

## Partial feedback

The environment stores full counterfactual matrices for evaluation, but its public
`reveal` method returns only selected DSP indices and their outcomes. The runner sends
that small feedback object to the policy. Tests verify that an unrouted DSP is not
updated.

## Oracle and metrics

For each DSP and capacity window, the non-causal oracle selects at most the top
`C_j` positive expected net rewards. It is an upper-bound benchmark and never trains
a policy.

Reported metrics distinguish:

- realized gross value, serving cost, and net profit;
- expected pseudo-regret against the constrained oracle;
- routes used versus incoming requests forwarded at least once;
- capacity utilization and violations;
- net profit per route.

Realized profit uses sampled outcomes. Pseudo-regret uses true expected net reward.
Because the oracle allocates over a complete capacity window, its within-window
cumulative trace can temporarily trail a causal policy; comparisons are meaningful
at completed window boundaries and at the full horizon.

## Current limitations

- DSP behavior is synthetic and deliberately simple.
- Quick runs use too few events and seeds for scientific conclusions.
- Shadow-price pacing is a heuristic rather than a solved constrained optimization
  problem.
- Discounted LinUCB uses batched model refreshes, so decisions between refreshes use
  the latest cached parameters.
- Recovery-time and multi-seed aggregation are planned but not yet reported.
