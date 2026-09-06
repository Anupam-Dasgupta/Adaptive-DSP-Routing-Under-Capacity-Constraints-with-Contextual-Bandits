"""Run the small multi-seed experiment set used for the project summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_dsp_routing.data import (  # noqa: E402
    AvazuFeaturePipeline,
    chronological_sample,
    chronological_split,
    load_avazu,
)
from adaptive_dsp_routing.evaluation import EpisodeResult, run_policy  # noqa: E402
from adaptive_dsp_routing.plotting import (  # noqa: E402
    plot_capacity_sweep,
    plot_seed_traces,
)
from adaptive_dsp_routing.policies import (  # noqa: E402
    DiscountedLinUCBPolicy,
    GreedyLinearPolicy,
    LinearUCBPolicy,
    RandomPolicy,
    ShadowPricePacer,
)
from adaptive_dsp_routing.simulator import generate_two_stage_world  # noqa: E402

BASE_METRICS = (
    "net_profit",
    "pseudo_regret",
    "capacity_utilization",
    "routing_efficiency",
    "routes_used",
)
DRIFT_METRICS = BASE_METRICS + (
    "post_drift_net_profit",
    "post_drift_pseudo_regret",
)


def append_ctr_results(lines: list[str], metrics_dir: Path) -> None:
    """Add the optional CTR-anchored experiment when its outputs are present."""

    metrics_path = metrics_dir / "ctr_model_metrics.json"
    summary_path = metrics_dir / "ctr_anchored_summary.csv"
    if not metrics_path.is_file() or not summary_path.is_file():
        return

    ctr = json.loads(metrics_path.read_text(encoding="utf-8"))
    routing = pd.read_csv(summary_path)
    lines.extend(
        [
            "",
            "## Supplementary CTR-anchored robustness experiment",
            "",
            "We also fit logistic regression on the chronological training prefix",
            "and used its out-of-sample CTR prediction as a shared relevance prior",
            "for the synthetic DSP world. `click` remained a target only, never a",
            "routing feature or DSP reward.",
            "",
            "### CTR model",
            "",
            "| ROC-AUC | Log loss | Brier score | Observed CTR | Mean predicted CTR |",
            "|---:|---:|---:|---:|---:|",
            f"| {ctr['roc_auc']:.4f} | {ctr['log_loss']:.4f} | "
            f"{ctr['brier_score']:.4f} | {ctr['observed_ctr']:.4f} | "
            f"{ctr['mean_predicted_ctr']:.4f} |",
            "",
            "### Routing at 40% mean capacity",
            "",
            "| Policy | Net profit | Pseudo-regret | Utilization | Violations |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in routing.sort_values("policy").itertuples():
        lines.append(
            f"| {row.policy} | {row.net_profit_mean:.1f} ± "
            f"{row.net_profit_std:.1f} | {row.pseudo_regret_mean:.1f} ± "
            f"{row.pseudo_regret_std:.1f} | "
            f"{row.capacity_utilization_mean:.3f} ± "
            f"{row.capacity_utilization_std:.3f} | "
            f"{row.capacity_violations_max:.0f} |"
        )
    indexed = routing.set_index("policy")
    if {"LinUCB", "Random"}.issubset(indexed.index):
        relative = 100.0 * (
            indexed.loc["LinUCB", "net_profit_mean"]
            - indexed.loc["Random", "net_profit_mean"]
        ) / indexed.loc["Random", "net_profit_mean"]
        lines.extend(
            [
                "",
                f"LinUCB's mean profit was {relative:.1f}% above Random in this",
                "robustness experiment. DSP outcomes remain synthetic and three",
                "seeds do not support a production-uplift claim.",
            ]
        )

    ablation_path = metrics_dir / "ctr_feature_ablation.csv"
    if not ablation_path.is_file():
        return
    ablation = pd.read_csv(ablation_path)
    lines.extend(
        [
            "",
            "### CTR routing-feature ablation",
            "",
            "The paired runs use the same CTR-anchored world and differ only in",
            "whether the router receives the out-of-sample CTR score.",
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
            "The CTR score produced no stable incremental profit improvement. Its",
            "inputs were already present in the original routing features.",
        ]
    )


def capacity_limits(ratio: float, n_dsps: int, window_size: int) -> np.ndarray:
    """Create heterogeneous DSP limits with the requested mean ratio."""

    multipliers = np.linspace(0.75, 1.25, n_dsps)
    values = np.rint(ratio * window_size * multipliers)
    return np.clip(values, 1, window_size).astype("int64")


def make_pacer(
    capacities: np.ndarray, window_size: int, eta: float
) -> ShadowPricePacer:
    return ShadowPricePacer(capacities, window_size, eta=eta)


def aggregate(
    rows: pd.DataFrame, groups: list[str], metrics: tuple[str, ...]
) -> pd.DataFrame:
    summary = rows.groupby(groups, as_index=False)[list(metrics)].agg(["mean", "std"])
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary


def result_row(
    experiment: str,
    policy: str,
    seed: int,
    result: EpisodeResult,
    capacity_ratio: float,
    drift_time: int | None = None,
) -> dict[str, object]:
    row = {
        "experiment": experiment,
        "policy": policy,
        "seed": seed,
        "capacity_ratio": capacity_ratio,
        **result.summary(),
    }
    if drift_time is not None:
        row["post_drift_net_profit"] = float(result.realized_net[drift_time:].sum())
        row["post_drift_pseudo_regret"] = float(
            result.oracle_expected_net[drift_time:].sum()
            - result.expected_net[drift_time:].sum()
        )
    return row


def write_report(
    capacity: pd.DataFrame,
    drift: pd.DataFrame,
    raw: pd.DataFrame,
    output_path: Path,
    *,
    total_events: int,
    evaluation_events: int,
    seeds: list[int],
    feature_count: int,
    start_time: object,
    end_time: object,
) -> None:
    lines = [
        "# Experiment Results",
        "",
        f"- Chronological source events: {total_events:,}",
        f"- Evaluation events per run: {evaluation_events:,}",
        f"- Source time span: {start_time} through {end_time}",
        f"- Feature dimension: {feature_count}",
        f"- Paired simulator seeds: {', '.join(map(str, seeds))}",
        "- Values below are mean ± sample standard deviation.",
        "",
        "## Capacity sweep",
        "",
        "| Capacity | Policy | Net profit | Pseudo-regret | Utilization |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in capacity.sort_values(["capacity_ratio", "policy"]).itertuples():
        lines.append(
            f"| {row.capacity_ratio:.2f} | {row.policy} | "
            f"{row.net_profit_mean:.1f} ± {row.net_profit_std:.1f} | "
            f"{row.pseudo_regret_mean:.1f} ± {row.pseudo_regret_std:.1f} | "
            f"{row.capacity_utilization_mean:.3f} ± "
            f"{row.capacity_utilization_std:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Abrupt drift",
            "",
            "| Policy | Net profit | Post-drift profit | Post-drift regret |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in drift.sort_values("policy").itertuples():
        lines.append(
            f"| {row.policy} | {row.net_profit_mean:.1f} ± "
            f"{row.net_profit_std:.1f} | {row.post_drift_net_profit_mean:.1f} ± "
            f"{row.post_drift_net_profit_std:.1f} | "
            f"{row.post_drift_pseudo_regret_mean:.1f} ± "
            f"{row.post_drift_pseudo_regret_std:.1f} |"
        )

    main = raw[
        (raw["experiment"] == "capacity")
        & np.isclose(raw["capacity_ratio"], 0.4)
    ].pivot(index="seed", columns="policy", values="net_profit")
    if {"LinUCB", "Random", "Greedy"}.issubset(main.columns):
        versus_random = main["LinUCB"] - main["Random"]
        versus_greedy = main["LinUCB"] - main["Greedy"]
        relative = 100.0 * versus_random.mean() / main["Random"].mean()
        lines.extend(
            [
                "",
                "## Paired differences at 40% capacity",
                "",
                f"- LinUCB minus Random profit: {versus_random.mean():.1f} ± "
                f"{versus_random.std(ddof=1):.1f} ({relative:.1f}% of Random mean).",
                f"- LinUCB minus Greedy profit: {versus_greedy.mean():.1f} ± "
                f"{versus_greedy.std(ddof=1):.1f}.",
            ]
        )
    lines.extend(
        [
            "",
            "The oracle is a non-causal expected-reward upper bound. Net profit uses",
            "sampled realized outcomes; pseudo-regret uses expected net reward.",
            "",
        ]
    )
    append_ctr_results(lines, output_path.parent)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--n-events", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--n-dsps", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=1_000)
    parser.add_argument("--capacity-ratios", type=float, nargs="+", default=[0.2, 0.4, 0.6])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--gammas", type=float, nargs="+", default=[0.98, 0.995, 0.999, 0.9995]
    )
    parser.add_argument("--pacing-eta", type=float, default=0.02)
    parser.add_argument("--update-interval", type=int, default=250)
    args = parser.parse_args()

    source = load_avazu(args.data_dir, n_events=None)
    events = chronological_sample(source, args.n_events)
    training, evaluation = chronological_split(events, train_fraction=0.2)
    pipeline = AvazuFeaturePipeline().fit(training)
    features = pipeline.transform(evaluation)
    feature_count = features.shape[1]
    print(
        f"Loaded {len(events):,} chronological events; "
        f"evaluating {len(evaluation):,} with {feature_count} features from "
        f"{events['datetime'].min()} to {events['datetime'].max()}."
    )

    raw_rows: list[dict[str, object]] = []
    drift_names = ["Greedy", "LinUCB"] + [
        f"Discounted γ={gamma:g}" for gamma in args.gammas
    ]
    drift_traces: dict[str, list[EpisodeResult]] = {
        name: [] for name in drift_names
    }
    for seed in args.seeds:
        stationary_world = generate_two_stage_world(
            features, n_dsps=args.n_dsps, seed=seed
        )
        for ratio in args.capacity_ratios:
            capacities = capacity_limits(ratio, args.n_dsps, args.window_size)
            policies = {
                "Random": RandomPolicy(args.n_dsps, ratio, seed=seed + 10_000),
                "Greedy": GreedyLinearPolicy(
                    args.n_dsps,
                    feature_count,
                    pacer=make_pacer(capacities, args.window_size, args.pacing_eta),
                ),
                "LinUCB": LinearUCBPolicy(
                    args.n_dsps,
                    feature_count,
                    alpha=args.alpha,
                    pacer=make_pacer(capacities, args.window_size, args.pacing_eta),
                ),
            }
            for name, policy in policies.items():
                result = run_policy(
                    features,
                    stationary_world,
                    policy,
                    capacities=capacities,
                    window_size=args.window_size,
                )
                raw_rows.append(result_row("capacity", name, seed, result, ratio))
                print(f"capacity={ratio:.2f} seed={seed} policy={name} done")

        drift_time = len(evaluation) // 2
        drift_world = generate_two_stage_world(
            features,
            n_dsps=args.n_dsps,
            seed=seed,
            drift_time=drift_time,
        )
        capacities = capacity_limits(0.4, args.n_dsps, args.window_size)
        drift_policies = {
            "Greedy": GreedyLinearPolicy(
                args.n_dsps,
                feature_count,
                pacer=make_pacer(capacities, args.window_size, args.pacing_eta),
            ),
            "LinUCB": LinearUCBPolicy(
                args.n_dsps,
                feature_count,
                alpha=args.alpha,
                pacer=make_pacer(capacities, args.window_size, args.pacing_eta),
            ),
        }
        for gamma in args.gammas:
            drift_policies[f"Discounted γ={gamma:g}"] = DiscountedLinUCBPolicy(
                args.n_dsps,
                feature_count,
                alpha=args.alpha,
                gamma=gamma,
                update_interval=args.update_interval,
                pacer=make_pacer(capacities, args.window_size, args.pacing_eta),
            )
        for name, policy in drift_policies.items():
            result = run_policy(
                features,
                drift_world,
                policy,
                capacities=capacities,
                window_size=args.window_size,
            )
            raw_rows.append(
                result_row(
                    "drift", name, seed, result, 0.4, drift_time=drift_time
                )
            )
            drift_traces[name].append(result)
            print(f"drift seed={seed} policy={name} done")

    raw = pd.DataFrame(raw_rows)
    capacity_rows = raw[raw["experiment"] == "capacity"]
    drift_rows = raw[raw["experiment"] == "drift"]
    capacity_summary = aggregate(
        capacity_rows, ["capacity_ratio", "policy"], BASE_METRICS
    )
    drift_summary = aggregate(drift_rows, ["policy"], DRIFT_METRICS)

    metrics_dir = ROOT / "results" / "metrics"
    figures_dir = ROOT / "results" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(metrics_dir / "experiment_runs_raw.csv", index=False)
    capacity_summary.to_csv(metrics_dir / "capacity_summary.csv", index=False)
    drift_summary.to_csv(metrics_dir / "drift_summary.csv", index=False)
    write_report(
        capacity_summary,
        drift_summary,
        raw,
        metrics_dir / "experiment_report.md",
        total_events=len(events),
        evaluation_events=len(evaluation),
        seeds=args.seeds,
        feature_count=feature_count,
        start_time=events["datetime"].min(),
        end_time=events["datetime"].max(),
    )
    plot_capacity_sweep(capacity_summary, figures_dir / "capacity_sweep.png")
    plot_seed_traces(
        drift_traces,
        figures_dir / "drift_comparison.png",
        drift_time=len(evaluation) // 2,
    )
    print(f"Saved summaries under {metrics_dir}")
    print(f"Saved figures under {figures_dir}")


if __name__ == "__main__":
    main()
