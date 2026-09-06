"""Quick stationary baseline comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_dsp_routing.data import (  # noqa: E402
    AvazuFeaturePipeline,
    chronological_split,
    load_avazu,
)
from adaptive_dsp_routing.evaluation import run_policy  # noqa: E402
from adaptive_dsp_routing.plotting import plot_comparison  # noqa: E402
from adaptive_dsp_routing.policies import (  # noqa: E402
    ForwardAllPolicy,
    GreedyLinearPolicy,
    LinearUCBPolicy,
    RandomPolicy,
    ShadowPricePacer,
)
from adaptive_dsp_routing.simulator import (  # noqa: E402
    generate_linear_world,
    generate_two_stage_world,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--n-events", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-dsps", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=1_000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--pacing-eta", type=float, default=0.02)
    parser.add_argument(
        "--world", choices=("linear", "two-stage"), default="linear"
    )
    args = parser.parse_args()

    events = load_avazu(args.data_dir, n_events=args.n_events)
    training, evaluation = chronological_split(events, train_fraction=0.2)
    pipeline = AvazuFeaturePipeline().fit(training)
    features = pipeline.transform(evaluation)
    world_builder = (
        generate_linear_world if args.world == "linear" else generate_two_stage_world
    )
    world = world_builder(features, n_dsps=args.n_dsps, seed=args.seed)
    capacities = np.linspace(
        0.25, 0.55, args.n_dsps, dtype="float64"
    ) * args.window_size
    capacities = capacities.astype("int64")

    def pacer() -> ShadowPricePacer:
        return ShadowPricePacer(capacities, args.window_size, eta=args.pacing_eta)

    policies = {
        "ForwardAll": ForwardAllPolicy(args.n_dsps),
        "Random": RandomPolicy(
            args.n_dsps,
            route_probability=float(capacities.mean() / args.window_size),
            seed=args.seed,
        ),
        "Greedy": GreedyLinearPolicy(
            args.n_dsps, features.shape[1], pacer=pacer()
        ),
        "LinUCB": LinearUCBPolicy(
            args.n_dsps,
            features.shape[1],
            alpha=args.alpha,
            pacer=pacer(),
        ),
    }
    results = {
        name: run_policy(
            features,
            world,
            policy,
            capacities=capacities,
            window_size=args.window_size,
        )
        for name, policy in policies.items()
    }
    metrics_path = ROOT / "results" / "metrics" / "stationary_quick.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps({name: result.summary() for name, result in results.items()}, indent=2),
        encoding="utf-8",
    )
    figure_path = plot_comparison(
        results, ROOT / "results" / "figures" / "stationary_quick.png"
    )
    print(json.dumps({name: result.summary() for name, result in results.items()}, indent=2))
    print(f"Saved {metrics_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
