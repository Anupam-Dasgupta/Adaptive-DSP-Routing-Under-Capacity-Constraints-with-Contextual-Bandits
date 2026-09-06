# CTR-Anchored Experiment

The logistic CTR model is fit only on the chronological training prefix.
Its out-of-sample score is a shared relevance prior, not a DSP reward.

## CTR model

- Source events: 100,000
- Evaluation events: 80,000
- Routing features: 50
- ROC-AUC: 0.6416
- Log loss: 0.4492
- Brier score: 0.1404
- Observed evaluation CTR: 0.1766
- Mean predicted CTR: 0.1959

## Routing results

Capacity ratio: 0.40; paired seeds: 42, 43, 44

| Policy | Net profit | Pseudo-regret | Utilization | Violations |
|---|---:|---:|---:|---:|
| Greedy | 35330.8 ± 2685.8 | 8047.7 ± 6916.5 | 0.838 ± 0.221 | 0 |
| LinUCB | 41985.9 ± 9069.1 | 1586.4 ± 225.0 | 0.995 ± 0.007 | 0 |
| Random | 23291.9 ± 6542.8 | 20192.0 ± 2538.3 | 0.915 ± 0.002 | 0 |

## Paired differences

- LinUCB minus Random: 18694.0 ± 2619.4 (80.3% of Random mean).
- LinUCB minus Greedy: 6655.1 ± 6892.1.

## CTR routing-feature ablation

The DSP world, capacities, seeds, and policies are identical. Only the
out-of-sample CTR score is hidden from or shown to the router. Positive
paired differences mean that exposing the CTR feature helped.

| Policy | Profit without CTR | Profit with CTR | Paired profit gain | Regret without CTR | Regret with CTR | Paired regret reduction |
|---|---:|---:|---:|---:|---:|---:|
| Greedy | 35273.6 | 35330.8 | 57.2 ± 699.2 | 8113.0 | 8047.7 | 65.3 ± 689.3 |
| LinUCB | 41986.7 | 41985.9 | -0.8 ± 106.8 | 1508.6 | 1586.4 | -77.8 ± 26.8 |

The added CTR score produced no stable profit improvement: Greedy's small mean gain
was dominated by seed variation, while LinUCB was effectively unchanged. The
original features already contain the inputs from which the CTR score is computed.

## Interpretation boundary

The CTR metrics use real held-out Avazu labels. DSP outcomes remain
synthetic, and three simulator seeds do not support a production-uplift
claim. This experiment is a robustness check on the routing method.
