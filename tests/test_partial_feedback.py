from __future__ import annotations

import unittest

import numpy as np

from adaptive_dsp_routing.evaluation import run_policy
from adaptive_dsp_routing.policies import LinearUCBPolicy, RoutingPolicy
from adaptive_dsp_routing.simulator import CounterfactualWorld, ObservedFeedback


class SpyPolicy(RoutingPolicy):
    def __init__(self) -> None:
        super().__init__(n_dsps=3)
        self.feedback: list[ObservedFeedback] = []

    def select(self, context: object) -> np.ndarray:
        del context
        return np.array([True, False, True])

    def update(self, context: object, feedback: ObservedFeedback) -> None:
        del context
        self.feedback.append(feedback)


class PartialFeedbackTests(unittest.TestCase):
    def test_policy_never_receives_unrouted_outcome(self) -> None:
        features = np.ones((4, 2))
        world = CounterfactualWorld(
            expected_net_rewards=np.arange(12, dtype="float64").reshape(4, 3),
            realized_gross_values=np.arange(12, dtype="float64").reshape(4, 3) + 1,
            costs=np.array([0.1, 0.2, 0.3]),
        )
        policy = SpyPolicy()
        result = run_policy(
            features,
            world,
            policy,
            capacities=np.array([4, 4, 4]),
            window_size=4,
        )

        self.assertTrue(np.all(~result.actions[:, 1]))
        self.assertEqual(len(policy.feedback), 4)
        for feedback in policy.feedback:
            np.testing.assert_array_equal(feedback.dsp_indices, np.array([0, 2]))
            self.assertEqual(feedback.net_rewards.shape, (2,))

    def test_linear_policy_updates_only_observed_dsp(self) -> None:
        policy = LinearUCBPolicy(3, 2, warm_start=0)
        context = np.array([1.0, 0.5])
        feedback = ObservedFeedback(
            dsp_indices=np.array([1]),
            gross_values=np.array([1.0]),
            net_rewards=np.array([0.7]),
        )
        policy.update(context, feedback)
        np.testing.assert_array_equal(policy.observations, [0, 1, 0])
        np.testing.assert_array_equal(policy.reward_sums[0], np.zeros(2))
        np.testing.assert_array_equal(policy.reward_sums[2], np.zeros(2))
        self.assertFalse(np.allclose(policy.reward_sums[1], 0.0))


if __name__ == "__main__":
    unittest.main()
