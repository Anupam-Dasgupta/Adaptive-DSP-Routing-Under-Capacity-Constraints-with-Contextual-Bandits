"""Evaluate routing in a DSP world anchored to chronological Avazu CTR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_dsp_routing.ctr import append_ctr_feature, fit_ctr_model  # noqa: E402
from adaptive_dsp_routing.data import (  # noqa: E402
    AvazuFeaturePipeline,
    chronological_sample,
    chronological_split,
    load_avazu,
)
from adaptive_dsp_routing.evaluation import EpisodeResult, run_policy  # noqa: E402
from adaptive_dsp_routing.plotting import (  # noqa: E402
    plot_comparison,
    plot_ctr_calibration,
)
from adaptive_dsp_routing.policies import (  # noqa: E402
    GreedyLinearPolicy,
    LinearUCBPolicy,
    RandomPolicy,
    ShadowPricePacer,
)
from adaptive_dsp_routing.simulator import generate_ctr_anchored_world  # noqa: E402


def capacity_limits(ratio: float, n_dsps: int, window_size: int) -> np.ndarray:
    multipliers = np.linspace(0.75, 1.25, n_dsps)
    values = np.rint(ratio * window_size * multipliers)
    return np.clip(values, 1, window_size).astype("int64")


def summarize_ablation(raw: pd.DataFrame) -> pd.DataFrame:
    """Compare paired runs with and without the CTR routing feature."""

    rows: list[dict[str, float | str]] = []
    for policy in ("Greedy", "LinUCB"):
        policy_rows = raw[raw["policy"] == policy]
        profit = policy_rows.pivot(
            index="seed", columns="feature_set", values="net_profit"
        )
        regret = policy_rows.pivot(
            index="seed", columns="feature_set", values="pseudo_regret"
        )
        required = {"without_ctr", "with_ctr"}
        if not required.issubset(profit.columns) or not required.issubset(
            regret.columns
        ):
            raise ValueError(f"Incomplete CTR ablation runs for {policy}.")
        profit_gain = profit["with_ctr"] - profit["without_ctr"]
        regret_reduction = regret["without_ctr"] - regret["with_ctr"]
        rows.append(
            {
                "policy": policy,
                "without_ctr_profit_mean": profit["without_ctr"].mean(),
                "with_ctr_profit_mean": profit["with_ctr"].mean(),
                "profit_gain_mean": profit_gain.mean(),
                "profit_gain_std": profit_gain.std(ddof=1),
                "without_ctr_regret_mean": regret["without_ctr"].mean(),
                "with_ctr_regret_mean": regret["with_ctr"].mean(),
                "regret_reduction_mean": regret_reduction.mean(),
                "regret_reduction_std": regret_reduction.std(ddof=1),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    ctr_metrics: dict[str, float],
    summary: pd.DataFrame,
    ablation: pd.DataFrame,
    raw: pd.DataFrame,
    output_path: Path,
    *,
    total_events: int,
    evaluation_events: int,
    feature_count: int,
    seeds: list[int],
    capacity_ratio: float,
) -> None:
    lines = [
        "# CTR-Anchored Experiment",
        "",
        "The logistic CTR model is fit only on the chronological training prefix.",
        "Its out-of-sample score is a shared relevance prior, not a DSP reward.",
        "",
        "## CTR model",
        "",
        f"- Source events: {total_events:,}",
        f"- Evaluation events: {evaluation_events:,}",
        f"- Routing features: {feature_count}",
        f"- ROC-AUC: {ctr_metrics['roc_auc']:.4f}",
        f"- Log loss: {ctr_metrics['log_loss']:.4f}",
        f"- Brier score: {ctr_metrics['brier_score']:.4f}",
        f"- Observed evaluation CTR: {ctr_metrics['observed_ctr']:.4f}",
        f"- Mean predicted CTR: {ctr_metrics['mean_predicted_ctr']:.4f}",
        "",
        "## Routing results",
        "",
        f"Capacity ratio: {capacity_ratio:.2f}; paired seeds: "
        + ", ".join(map(str, seeds)),
        "",
        "| Policy | Net profit | Pseudo-regret | Utilization | Violations |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values("policy").itertuples():
        lines.append(
            f"| {row.policy} | {row.net_profit_mean:.1f} ± "
            f"{row.net_profit_std:.1f} | {row.pseudo_regret_mean:.1f} ± "
            f"{row.pseudo_regret_std:.1f} | {row.capacity_utilization_mean:.3f} ± "
            f"{row.capacity_utilization_std:.3f} | "
            f"{row.capacity_violations_max:.0f} |"
        )

    primary = raw[raw["feature_set"] == "with_ctr"]
    paired = primary.pivot(index="seed", columns="policy", values="net_profit")
    if {"LinUCB", "Random", "Greedy"}.issubset(paired.columns):
        random_difference = paired["LinUCB"] - paired["Random"]
        greedy_difference = paired["LinUCB"] - paired["Greedy"]
        relative = 100.0 * random_difference.mean() / paired["Random"].mean()
        lines.extend(
            [
                "",
                "## Paired differences",
                "",
                f"- LinUCB minus Random: {random_difference.mean():.1f} ± "
                f"{random_difference.std(ddof=1):.1f} ({relative:.1f}% of Random mean).",
                f"- LinUCB minus Greedy: {greedy_difference.mean():.1f} ± "
                f"{greedy_difference.std(ddof=1):.1f}.",
                "",
            ]
        )
    lines.extend(
        [
            "## CTR routing-feature ablation",
            "",
            "The DSP world, capacities, seeds, and policies are identical. Only the",
            "out-of-sample CTR score is hidden from or shown to the router. Positive",
            "paired differences mean that exposing the CTR feature helped.",
            "",
            "| Policy | Profit without CTR | Profit with CTR | Paired profit gain | "
            "Regret without CTR | Regret with CTR | Paired regret reduction |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ablation.itertuples():
        lines.append(
            f"| {row.policy} | {row.without_ctr_profit_mean:.1f} | "
            f"{row.with_ctr_profit_mean:.1f} | {row.profit_gain_mean:.1f} ± "
            f"{row.profit_gain_std:.1f} | {row.without_ctr_regret_mean:.1f} | "
            f"{row.with_ctr_regret_mean:.1f} | "
            f"{row.regret_reduction_mean:.1f} ± {row.regret_reduction_std:.1f} |"
        )
    lines.extend(
        [
            "",
            "The added CTR score produced no stable profit improvement: Greedy's",
            "small mean gain was dominated by seed variation, while LinUCB was",
            "effectively unchanged. The original features already contain the inputs",
            "from which the CTR score is computed.",
            "",
        ]
    )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "The CTR metrics use real held-out Avazu labels. DSP outcomes remain",
            "synthetic, and three simulator seeds do not support a production-uplift",
            "claim. This experiment is a robustness check on the routing method.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--n-events", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--n-dsps", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=1_000)
    parser.add_argument("--capacity-ratio", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--pacing-eta", type=float, default=0.02)
    args = parser.parse_args()

    source = load_avazu(args.data_dir, n_events=None)
    events = chronological_sample(source, args.n_events)
    training, evaluation = chronological_split(events, train_fraction=0.2)
    pipeline = AvazuFeaturePipeline().fit(training)
    training_features = pipeline.transform(training)
    base_features = pipeline.transform(evaluation)
    ctr_result = fit_ctr_model(
        training_features,
        training["click"],
        base_features,
        evaluation["click"],
    )
    routing_features = append_ctr_feature(base_features, ctr_result.probabilities)
    capacities = capacity_limits(
        args.capacity_ratio, args.n_dsps, args.window_size
    )

    def pacer() -> ShadowPricePacer:
        return ShadowPricePacer(
            capacities, args.window_size, eta=args.pacing_eta
        )

    rows: list[dict[str, float | int | str]] = []
    first_results: dict[str, EpisodeResult] = {}
    for seed in args.seeds:
        world = generate_ctr_anchored_world(
            base_features,
            ctr_result.probabilities,
            n_dsps=args.n_dsps,
            seed=seed,
        )
        random_result = run_policy(
            routing_features,
            world,
            RandomPolicy(args.n_dsps, args.capacity_ratio, seed=seed + 10_000),
            capacities=capacities,
            window_size=args.window_size,
        )
        rows.append(
            {
                "policy": "Random",
                "feature_set": "with_ctr",
                "seed": seed,
                **random_result.summary(),
            }
        )
        if seed == args.seeds[0]:
            first_results["Random"] = random_result
        print(f"seed={seed} policy=Random feature_set=with_ctr done")

        feature_sets = {
            "without_ctr": base_features,
            "with_ctr": routing_features,
        }
        for feature_set, policy_features in feature_sets.items():
            policies = {
                "Greedy": GreedyLinearPolicy(
                    args.n_dsps, policy_features.shape[1], pacer=pacer()
                ),
                "LinUCB": LinearUCBPolicy(
                    args.n_dsps,
                    policy_features.shape[1],
                    alpha=args.alpha,
                    pacer=pacer(),
                ),
            }
            for name, policy in policies.items():
                result = run_policy(
                    policy_features,
                    world,
                    policy,
                    capacities=capacities,
                    window_size=args.window_size,
                )
                rows.append(
                    {
                        "policy": name,
                        "feature_set": feature_set,
                        "seed": seed,
                        **result.summary(),
                    }
                )
                if seed == args.seeds[0] and feature_set == "with_ctr":
                    first_results[name] = result
                print(
                    f"seed={seed} policy={name} feature_set={feature_set} done"
                )

    raw = pd.DataFrame(rows)
    primary = raw[raw["feature_set"] == "with_ctr"]
    summary = primary.groupby("policy", as_index=False).agg(
        net_profit_mean=("net_profit", "mean"),
        net_profit_std=("net_profit", "std"),
        pseudo_regret_mean=("pseudo_regret", "mean"),
        pseudo_regret_std=("pseudo_regret", "std"),
        capacity_utilization_mean=("capacity_utilization", "mean"),
        capacity_utilization_std=("capacity_utilization", "std"),
        capacity_violations_max=("capacity_violations", "max"),
    )
    ablation = summarize_ablation(raw)

    metrics_dir = ROOT / "results" / "metrics"
    figures_dir = ROOT / "results" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    ctr_metrics = ctr_result.summary()
    (metrics_dir / "ctr_model_metrics.json").write_text(
        json.dumps(ctr_metrics, indent=2), encoding="utf-8"
    )
    raw.to_csv(metrics_dir / "ctr_anchored_raw.csv", index=False)
    summary.to_csv(metrics_dir / "ctr_anchored_summary.csv", index=False)
    ablation.to_csv(metrics_dir / "ctr_feature_ablation.csv", index=False)
    write_report(
        ctr_metrics,
        summary,
        ablation,
        raw,
        metrics_dir / "ctr_anchored_report.md",
        total_events=len(events),
        evaluation_events=len(evaluation),
        feature_count=routing_features.shape[1],
        seeds=args.seeds,
        capacity_ratio=args.capacity_ratio,
    )
    plot_ctr_calibration(ctr_result, figures_dir / "ctr_calibration.png")
    plot_comparison(
        first_results, figures_dir / "ctr_anchored_comparison.png"
    )
    print(json.dumps(ctr_metrics, indent=2))
    print(summary.to_string(index=False))
    print(ablation.to_string(index=False))


if __name__ == "__main__":
    main()
