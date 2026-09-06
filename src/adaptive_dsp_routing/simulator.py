"""Small reproducible DSP world and hard capacity enforcement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservedFeedback:
    """Only outcomes corresponding to routed DSP indices."""

    dsp_indices: np.ndarray
    gross_values: np.ndarray
    net_rewards: np.ndarray


@dataclass(frozen=True)
class CounterfactualWorld:
    """Paired evaluation world; policies access it only through ``reveal``."""

    expected_net_rewards: np.ndarray
    realized_gross_values: np.ndarray
    costs: np.ndarray

    def __post_init__(self) -> None:
        expected = np.asarray(self.expected_net_rewards, dtype="float64")
        gross = np.asarray(self.realized_gross_values, dtype="float64")
        costs = np.asarray(self.costs, dtype="float64")
        if expected.shape != gross.shape or expected.ndim != 2:
            raise ValueError("Expected and realized reward arrays must share shape (T, J).")
        if costs.shape != (expected.shape[1],):
            raise ValueError("costs must contain one value per DSP.")
        object.__setattr__(self, "expected_net_rewards", expected)
        object.__setattr__(self, "realized_gross_values", gross)
        object.__setattr__(self, "costs", costs)

    @property
    def n_events(self) -> int:
        return self.expected_net_rewards.shape[0]

    @property
    def n_dsps(self) -> int:
        return self.expected_net_rewards.shape[1]

    def reveal(self, event_index: int, actions: np.ndarray) -> ObservedFeedback:
        selected = np.flatnonzero(np.asarray(actions, dtype=bool))
        gross = self.realized_gross_values[event_index, selected].copy()
        return ObservedFeedback(
            dsp_indices=selected,
            gross_values=gross,
            net_rewards=gross - self.costs[selected],
        )


def generate_linear_world(
    features: object,
    *,
    n_dsps: int = 4,
    seed: int = 42,
    costs: np.ndarray | None = None,
    drift_time: int | None = None,
    drift_scale: float = 0.8,
) -> CounterfactualWorld:
    """Generate a paired linear/softplus DSP world with optional abrupt drift."""

    if n_dsps <= 0:
        raise ValueError("n_dsps must be positive.")
    n_events, n_features = features.shape
    if drift_time is not None and not 0 < drift_time < n_events:
        raise ValueError("drift_time must be inside the event stream.")
    if not 0.0 <= drift_scale <= 1.0:
        raise ValueError("drift_scale must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    scale = 0.25 / np.sqrt(max(n_features, 1))
    coefficients = rng.normal(0.0, scale, size=(n_features, n_dsps))
    logits = np.asarray(features @ coefficients, dtype="float64")

    if drift_time is not None:
        affected = np.arange(0, n_dsps, 2)
        new_direction = rng.normal(0.0, scale, size=(n_features, len(affected)))
        drifted = (
            (1.0 - drift_scale) * coefficients[:, affected]
            + drift_scale * new_direction
        )
        logits[drift_time:, affected] = np.asarray(
            features[drift_time:] @ drifted, dtype="float64"
        )

    expected_gross = np.logaddexp(0.0, logits)
    if costs is None:
        costs = np.linspace(0.20, 0.45, n_dsps, dtype="float64")
    costs = np.asarray(costs, dtype="float64")
    if costs.shape != (n_dsps,) or np.any(costs < 0):
        raise ValueError("costs must be non-negative with one value per DSP.")
    realized_gross = np.clip(
        expected_gross + rng.normal(0.0, 0.15, expected_gross.shape),
        0.0,
        None,
    )
    return CounterfactualWorld(
        expected_net_rewards=expected_gross - costs,
        realized_gross_values=realized_gross,
        costs=costs,
    )


def generate_two_stage_world(
    features: object,
    *,
    n_dsps: int = 4,
    seed: int = 42,
    costs: np.ndarray | None = None,
    drift_time: int | None = None,
    drift_scale: float = 0.8,
) -> CounterfactualWorld:
    """Generate zero-inflated DSP values from buy and conditional-value models."""

    if n_dsps <= 0:
        raise ValueError("n_dsps must be positive.")
    n_events, n_features = features.shape
    if drift_time is not None and not 0 < drift_time < n_events:
        raise ValueError("drift_time must be inside the event stream.")
    if not 0.0 <= drift_scale <= 1.0:
        raise ValueError("drift_scale must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    scale = 0.35 / np.sqrt(max(n_features, 1))
    buy_coefficients = rng.normal(0.0, scale, size=(n_features, n_dsps))
    value_coefficients = rng.normal(0.0, scale, size=(n_features, n_dsps))
    buy_logits = np.asarray(features @ buy_coefficients, dtype="float64")
    value_logits = np.asarray(features @ value_coefficients, dtype="float64")

    if drift_time is not None:
        affected = np.arange(0, n_dsps, 2)
        for coefficients, logits in (
            (buy_coefficients, buy_logits),
            (value_coefficients, value_logits),
        ):
            direction = rng.normal(0.0, scale, size=(n_features, len(affected)))
            drifted = (
                (1.0 - drift_scale) * coefficients[:, affected]
                + drift_scale * direction
            )
            logits[drift_time:, affected] = np.asarray(
                features[drift_time:] @ drifted, dtype="float64"
            )

    buy_probability = 1.0 / (1.0 + np.exp(-np.clip(buy_logits, -30.0, 30.0)))
    expected_if_bought = np.logaddexp(0.0, value_logits)
    expected_gross = buy_probability * expected_if_bought
    bought = rng.random(expected_gross.shape) < buy_probability
    realized_if_bought = np.logaddexp(
        0.0,
        value_logits + rng.normal(0.0, 0.25, value_logits.shape),
    )
    realized_gross = bought * realized_if_bought
    if costs is None:
        costs = np.linspace(0.08, 0.22, n_dsps, dtype="float64")
    costs = np.asarray(costs, dtype="float64")
    if costs.shape != (n_dsps,) or np.any(costs < 0):
        raise ValueError("costs must be non-negative with one value per DSP.")
    return CounterfactualWorld(
        expected_net_rewards=expected_gross - costs,
        realized_gross_values=realized_gross,
        costs=costs,
    )


def generate_ctr_anchored_world(
    features: object,
    ctr_probabilities: object,
    *,
    n_dsps: int = 4,
    seed: int = 42,
    costs: np.ndarray | None = None,
    drift_time: int | None = None,
    drift_scale: float = 0.8,
) -> CounterfactualWorld:
    """Generate DSP responses around an out-of-sample Avazu CTR prior.

    The CTR score represents shared request relevance. DSP-specific coefficients add
    heterogeneous buying and value preferences. The observed Avazu click itself is
    never used as a DSP reward.
    """

    if n_dsps <= 0:
        raise ValueError("n_dsps must be positive.")
    n_events, n_features = features.shape
    prior = np.asarray(ctr_probabilities, dtype="float64").ravel()
    if prior.shape != (n_events,):
        raise ValueError("ctr_probabilities must contain one value per event.")
    if not np.all(np.isfinite(prior)) or np.any((prior < 0) | (prior > 1)):
        raise ValueError("ctr_probabilities must be finite values in [0, 1].")
    if drift_time is not None and not 0 < drift_time < n_events:
        raise ValueError("drift_time must be inside the event stream.")
    if not 0.0 <= drift_scale <= 1.0:
        raise ValueError("drift_scale must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    scale = 0.25 / np.sqrt(max(n_features, 1))
    buy_coefficients = rng.normal(0.0, scale, size=(n_features, n_dsps))
    value_coefficients = rng.normal(0.0, scale, size=(n_features, n_dsps))
    context_buy = np.asarray(features @ buy_coefficients, dtype="float64")
    value_logits = np.asarray(features @ value_coefficients, dtype="float64")

    if drift_time is not None:
        affected = np.arange(0, n_dsps, 2)
        for coefficients, logits in (
            (buy_coefficients, context_buy),
            (value_coefficients, value_logits),
        ):
            direction = rng.normal(0.0, scale, size=(n_features, len(affected)))
            drifted = (
                (1.0 - drift_scale) * coefficients[:, affected]
                + drift_scale * direction
            )
            logits[drift_time:, affected] = np.asarray(
                features[drift_time:] @ drifted, dtype="float64"
            )

    clipped_prior = np.clip(prior, 1e-5, 1.0 - 1e-5)
    prior_logit = np.log(clipped_prior / (1.0 - clipped_prior))
    dsp_offsets = np.linspace(-0.5, 0.5, n_dsps, dtype="float64")
    buy_logits = prior_logit[:, None] + context_buy + dsp_offsets
    buy_probability = 1.0 / (1.0 + np.exp(-np.clip(buy_logits, -30.0, 30.0)))

    expected_if_bought = 1.0 + np.logaddexp(0.0, value_logits)
    expected_gross = buy_probability * expected_if_bought
    bought = rng.random(expected_gross.shape) < buy_probability
    realized_if_bought = 1.0 + np.logaddexp(
        0.0,
        value_logits + rng.normal(0.0, 0.25, value_logits.shape),
    )
    realized_gross = bought * realized_if_bought
    if costs is None:
        costs = np.linspace(0.08, 0.22, n_dsps, dtype="float64")
    costs = np.asarray(costs, dtype="float64")
    if costs.shape != (n_dsps,) or np.any(costs < 0):
        raise ValueError("costs must be non-negative with one value per DSP.")
    return CounterfactualWorld(
        expected_net_rewards=expected_gross - costs,
        realized_gross_values=realized_gross,
        costs=costs,
    )


class CapacityLimiter:
    """Enforce independent DSP capacities in fixed-size event windows."""

    def __init__(self, capacities: np.ndarray, window_size: int):
        values = np.asarray(capacities, dtype="int64")
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if values.ndim != 1 or np.any(values < 0) or np.any(values > window_size):
            raise ValueError("capacities must be a 1D array between 0 and window_size.")
        self.capacities = values
        self.window_size = int(window_size)
        self.total_used = np.zeros_like(values)
        self._window_used = np.zeros_like(values)
        self._position = 0

    def apply(self, requested: np.ndarray) -> np.ndarray:
        actions = np.asarray(requested, dtype=bool)
        if actions.shape != self.capacities.shape:
            raise ValueError("requested actions must contain one value per DSP.")
        if self._position > 0 and self._position % self.window_size == 0:
            self._window_used.fill(0)
        allowed = actions & (self._window_used < self.capacities)
        self._window_used += allowed.astype("int64")
        self.total_used += allowed.astype("int64")
        self._position += 1
        return allowed
