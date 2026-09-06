# Seven-Day Learning Guide

This project is learnable in one hard-working week because it uses a small number of
ideas repeatedly. Aim for four to six focused hours per day. Do not try to memorize
the code line by line; understand the data flow, equations, and experimental choices.

## Day 1 — data, chronology, and leakage

Read `data.py`, `PLAN.md`, and the feature tests.

Be able to explain:

- why the supplied train/test files had to be recombined;
- why a chronological split is different from a random split;
- why `click`, `id`, timestamps, and source indices are excluded;
- why preprocessing is fit only on the earlier prefix;
- why the final sample is systematic and order-preserving.

Exercise: print five raw rows, five ordered rows, and all feature names. Explain every
feature without looking at the code.

## Day 2 — the semi-synthetic environment

Read `simulator.py` and the methodology's DSP-world section.

Understand the two-stage model:

```text
buy_probability = sigmoid(beta · x)
buy ~ Bernoulli(buy_probability)
value_if_bought = softplus(phi · x + noise)
net_reward = buy * value_if_bought - routing_cost
```

Be able to explain why real Avazu contexts are useful even though DSP rewards are
synthetic, and why this is more honest than treating `click` as every DSP's outcome.

Exercise: change one seed and one DSP cost, rerun a smoke experiment, and predict the
direction of the metric changes before viewing them.

## Day 3 — partial feedback, capacity, and the oracle

Read `evaluation.py`, `CapacityLimiter`, and the capacity/feedback tests.

Be able to explain:

- why one request can be routed to several DSPs;
- why an unrouted DSP cannot update;
- why the hard limiter and pacing controller are separate;
- why the oracle is allowed to be non-causal;
- why pseudo-regret uses expected reward while profit uses sampled reward.

Exercise: work through a four-event, two-DSP example by hand and reproduce the oracle
selection under a capacity of one route per DSP.

## Day 4 — ridge regression, Greedy, and LinUCB

Read the first half of `policies.py`.

Know these expressions:

```text
theta_hat = A^-1 b
greedy_score = theta_hat · x
uncertainty = sqrt(x^T A^-1 x)
linucb_score = greedy_score + alpha * uncertainty
```

The uncertainty term is large for contexts the model has seen less information about.
`alpha` controls how much the policy values that uncertainty. The Sherman–Morrison
update changes `A^-1` after one observation without solving a new matrix system.

Exercise: explain why `alpha = 0` is Greedy and why a large `alpha` can waste scarce
capacity.

## Day 5 — pacing, drift, and discounting

Read `ShadowPricePacer`, `DiscountedLinUCBPolicy`, and `run_drift.py`.

Understand:

```text
lambda = max(0, lambda + eta * (action - target_rate))
discounted_weight(age) = gamma ^ age
```

The shadow price represents the opportunity cost of consuming DSP capacity now.
Discounting forgets old evidence: smaller `gamma` adapts faster but increases variance.

Exercise: explain the actual negative result. In this simulator, ordinary LinUCB beat
all discounted variants; among discounted variants, performance improved as `gamma`
approached 1. Do not claim that discounting automatically handles drift better.

## Day 6 — paired experiments and interpretation

Read `run_suite.py` and `results/metrics/experiment_report.md`.

Be able to explain:

- why policies under one seed share the same generated world;
- why paired profit differences are clearer than comparing unpaired means;
- why mean ± standard deviation over three seeds is useful but not a significance
  claim;
- why capacity scarcity increased LinUCB's relative advantage over Random;
- why three seeds and a semi-synthetic world limit generalization.

Exercise: reproduce the 17.3% improvement and 78.8% regret reduction from the table.

## Day 7 — rerun and defend

Run the tests, one smoke experiment, and the full suite help command. Then answer these
without notes:

1. Why is this not supervised CTR prediction?
2. Why is this not a standard single-action bandit?
3. What information reaches a policy update?
4. What does `x^T A^-1 x` mean?
5. Why can capacity make a positive-reward route undesirable now?
6. What does the shadow price mean economically?
7. Why can aggressive forgetting hurt?
8. Why is the oracle clairvoyant, and why is that acceptable?
9. What exactly is real and synthetic in the experiment?
10. Which result surprised you, and what experiment would you run next?

If you can answer those questions and make a small code change without assistance,
you understand enough to discuss the project honestly in an interview.

## Optional CTR-anchored extension

After the core seven days, read `ctr.py` and `run_ctr_anchored.py`. Be able to explain
why `click` is valid as the supervised model's target but invalid as an input feature
or as the reward for every DSP. Also explain why chronological out-of-sample
predictions avoid future-label leakage and why calibration matters when a predicted
probability becomes an input to a downstream decision system.
