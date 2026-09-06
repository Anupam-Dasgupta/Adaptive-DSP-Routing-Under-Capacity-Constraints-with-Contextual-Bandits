from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from adaptive_dsp_routing.ctr import append_ctr_feature, fit_ctr_model
from adaptive_dsp_routing.simulator import generate_ctr_anchored_world


class CTRAnchorTests(unittest.TestCase):
    def test_evaluation_labels_do_not_change_predictions(self) -> None:
        training = sparse.csr_matrix(
            np.column_stack([np.ones(20), np.linspace(-2.0, 2.0, 20)])
        )
        evaluation = sparse.csr_matrix(
            np.column_stack([np.ones(10), np.linspace(-1.5, 1.5, 10)])
        )
        training_labels = np.array([0] * 10 + [1] * 10)
        evaluation_labels = np.arange(10) % 2
        first = fit_ctr_model(
            training, training_labels, evaluation, evaluation_labels
        )
        second = fit_ctr_model(
            training, training_labels, evaluation, 1 - evaluation_labels
        )
        np.testing.assert_allclose(first.probabilities, second.probabilities)

    def test_ctr_feature_is_appended_without_changing_existing_values(self) -> None:
        features = sparse.csr_matrix(np.eye(4))
        probabilities = np.linspace(0.1, 0.4, 4)
        combined = append_ctr_feature(features, probabilities)
        self.assertEqual(combined.shape, (4, 5))
        np.testing.assert_allclose(combined[:, :4].toarray(), features.toarray())
        np.testing.assert_allclose(combined[:, 4].toarray().ravel(), probabilities)

    def test_ctr_anchored_world_is_seeded_and_responds_to_prior(self) -> None:
        features = sparse.csr_matrix(np.eye(8))
        low_prior = np.full(8, 0.05)
        high_prior = np.full(8, 0.50)
        first = generate_ctr_anchored_world(features, low_prior, n_dsps=2, seed=9)
        second = generate_ctr_anchored_world(features, low_prior, n_dsps=2, seed=9)
        high = generate_ctr_anchored_world(features, high_prior, n_dsps=2, seed=9)
        np.testing.assert_array_equal(
            first.realized_gross_values, second.realized_gross_values
        )
        self.assertGreater(
            high.expected_net_rewards.mean(), first.expected_net_rewards.mean()
        )


if __name__ == "__main__":
    unittest.main()
