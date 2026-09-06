# Experiment Results

- Chronological source events: 100,000
- Evaluation events per run: 80,000
- Source time span: 2014-10-21 00:00:00 through 2014-10-30 05:00:00
- Feature dimension: 49
- Paired simulator seeds: 42, 43, 44
- Values below are mean ± sample standard deviation.

## Capacity sweep

| Capacity | Policy | Net profit | Pseudo-regret | Utilization |
|---:|---|---:|---:|---:|
| 0.20 | Greedy | 12063.6 ± 3590.1 | 1102.4 ± 270.4 | 0.866 ± 0.223 |
| 0.20 | LinUCB | 12296.9 ± 3486.0 | 890.2 ± 162.7 | 0.962 ± 0.066 |
| 0.20 | Random | 9987.5 ± 3224.4 | 3131.8 ± 353.1 | 0.913 ± 0.003 |
| 0.40 | Greedy | 22175.7 ± 8031.2 | 2169.0 ± 879.6 | 0.795 ± 0.321 |
| 0.40 | LinUCB | 23505.1 ± 7107.3 | 925.9 ± 122.0 | 0.957 ± 0.075 |
| 0.40 | Random | 20044.4 ± 6491.0 | 4367.4 ± 600.1 | 0.915 ± 0.002 |
| 0.60 | Greedy | 32848.0 ± 10488.8 | 1908.2 ± 317.2 | 0.817 ± 0.293 |
| 0.60 | LinUCB | 33959.5 ± 10464.2 | 875.0 ± 63.1 | 0.935 ± 0.112 |
| 0.60 | Random | 30049.5 ± 9718.5 | 4674.3 ± 539.4 | 0.916 ± 0.000 |

## Abrupt drift

| Policy | Net profit | Post-drift profit | Post-drift regret |
|---|---:|---:|---:|
| Discounted γ=0.98 | 20925.5 ± 5754.8 | 10574.5 ± 2812.0 | 1343.0 ± 211.7 |
| Discounted γ=0.995 | 21251.3 ± 5778.2 | 10752.7 ± 2911.4 | 1188.3 ± 175.2 |
| Discounted γ=0.999 | 21840.7 ± 5857.5 | 11019.8 ± 2957.4 | 897.7 ± 80.0 |
| Discounted γ=0.9995 | 22098.7 ± 5964.8 | 11125.4 ± 3056.6 | 756.6 ± 51.8 |
| Greedy | 21696.1 ± 5298.4 | 10954.2 ± 3070.0 | 938.3 ± 383.5 |
| LinUCB | 22835.8 ± 5825.2 | 11366.4 ± 2899.1 | 537.6 ± 123.2 |

## Paired differences at 40% capacity

- LinUCB minus Random profit: 3460.7 ± 617.8 (17.3% of Random mean).
- LinUCB minus Greedy profit: 1329.3 ± 973.2.

The oracle is a non-causal expected-reward upper bound. Net profit uses
sampled realized outcomes; pseudo-regret uses expected net reward.
