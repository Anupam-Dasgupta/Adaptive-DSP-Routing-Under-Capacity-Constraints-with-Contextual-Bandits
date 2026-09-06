from __future__ import annotations

import unittest

import numpy as np

from adaptive_dsp_routing.evaluation import constrained_oracle_actions
from adaptive_dsp_routing.policies import ShadowPricePacer
from adaptive_dsp_routing.simulator import CapacityLimiter


class CapacityTests(unittest.TestCase):
    def test_hard_limit_holds_in_every_window(self) -> None:
        limiter = CapacityLimiter(np.array([2, 4]), window_size=5)
        actions = np.vstack([limiter.apply(np.ones(2, dtype=bool)) for _ in range(12)])
        for start in range(0, len(actions), 5):
            used = actions[start : start + 5].sum(axis=0)
            self.assertTrue(np.all(used <= np.array([2, 4])))

    def test_capacity_resets_at_window_boundary(self) -> None:
        limiter = CapacityLimiter(np.array([1]), window_size=2)
        allowed = [bool(limiter.apply(np.array([True]))[0]) for _ in range(3)]
        self.assertEqual(allowed, [True, False, True])

    def test_oracle_uses_only_top_positive_rewards(self) -> None:
        rewards = np.array(
            [
                [0.5, -1.0],
                [0.2, 0.4],
                [0.9, 0.3],
                [-0.1, 0.8],
            ]
        )
        actions = constrained_oracle_actions(
            rewards, capacities=np.array([1, 2]), window_size=4
        )
        np.testing.assert_array_equal(actions[:, 0], [False, False, True, False])
        np.testing.assert_array_equal(actions[:, 1], [False, True, False, True])

    def test_shadow_price_rises_after_over_routing(self) -> None:
        pacer = ShadowPricePacer(np.array([1]), window_size=4, eta=1.0)
        pacer.update(np.array([True]))
        self.assertGreater(pacer.shadow_prices[0], 0.0)
        self.assertLess(pacer.adjust(np.array([0.5]))[0], 0.0)


if __name__ == "__main__":
    unittest.main()
