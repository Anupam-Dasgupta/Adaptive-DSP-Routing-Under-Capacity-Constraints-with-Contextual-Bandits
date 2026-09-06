from __future__ import annotations

import unittest

import numpy as np

from adaptive_dsp_routing.evaluation import run_policy
from adaptive_dsp_routing.policies import RandomPolicy
from adaptive_dsp_routing.simulator import (
    generate_linear_world,
    generate_two_stage_world,
)


class ReproducibilityTests(unittest.TestCase):
    def test_same_seed_recreates_world_and_policy_trace(self) -> None:
        features = np.arange(60, dtype="float64").reshape(20, 3) / 60.0
        first_world = generate_linear_world(features, n_dsps=3, seed=19)
        second_world = generate_linear_world(features, n_dsps=3, seed=19)
        np.testing.assert_array_equal(
            first_world.realized_gross_values, second_world.realized_gross_values
        )

        first = run_policy(
            features,
            first_world,
            RandomPolicy(3, 0.5, seed=23),
            capacities=np.array([4, 4, 4]),
            window_size=5,
        )
        second = run_policy(
            features,
            second_world,
            RandomPolicy(3, 0.5, seed=23),
            capacities=np.array([4, 4, 4]),
            window_size=5,
        )
        np.testing.assert_array_equal(first.actions, second.actions)
        np.testing.assert_array_equal(first.realized_net, second.realized_net)

    def test_two_stage_world_is_seeded(self) -> None:
        features = np.eye(8, dtype="float64")
        first = generate_two_stage_world(features, n_dsps=2, seed=31, drift_time=4)
        second = generate_two_stage_world(features, n_dsps=2, seed=31, drift_time=4)
        np.testing.assert_array_equal(
            first.expected_net_rewards, second.expected_net_rewards
        )
        np.testing.assert_array_equal(
            first.realized_gross_values, second.realized_gross_values
        )


if __name__ == "__main__":
    unittest.main()
