"""Routing baselines and small per-DSP linear bandit policies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse

from .simulator import ObservedFeedback


class RoutingPolicy(ABC):
    def __init__(self, n_dsps: int):
        if n_dsps <= 0:
            raise ValueError("n_dsps must be positive.")
        self.n_dsps = n_dsps

    @abstractmethod
    def select(self, context: object) -> np.ndarray:
        """Return one requested binary action per DSP."""

    def update(self, context: object, feedback: ObservedFeedback) -> None:
        """Learn only from the routed outcomes contained in ``feedback``."""

        del context, feedback

    def on_actions(self, actions: np.ndarray) -> None:
        """Observe final routed actions, but never their counterfactual outcomes."""

        del actions


class ForwardAllPolicy(RoutingPolicy):
    def select(self, context: object) -> np.ndarray:
        del context
        return np.ones(self.n_dsps, dtype=bool)


class RandomPolicy(RoutingPolicy):
    def __init__(self, n_dsps: int, route_probability: float, seed: int = 42):
        super().__init__(n_dsps)
        if not 0.0 <= route_probability <= 1.0:
            raise ValueError("route_probability must be between 0 and 1.")
        self.route_probability = route_probability
        self._rng = np.random.default_rng(seed)

    def select(self, context: object) -> np.ndarray:
        del context
        return self._rng.random(self.n_dsps) < self.route_probability


def _as_dense(context: object, n_features: int) -> np.ndarray:
    if sparse.issparse(context):
        values = context.toarray().ravel()
    else:
        values = np.asarray(context, dtype="float64").ravel()
    if values.shape != (n_features,):
        raise ValueError(f"Expected {n_features} context features, got {values.size}.")
    return values


class ShadowPricePacer:
    """Raise routing thresholds when a DSP consumes capacity too quickly."""

    def __init__(self, capacities: np.ndarray, window_size: int, eta: float = 0.02):
        capacities = np.asarray(capacities, dtype="float64")
        if window_size <= 0 or capacities.ndim != 1:
            raise ValueError("capacities must be 1D and window_size must be positive.")
        if np.any(capacities < 0) or np.any(capacities > window_size):
            raise ValueError("capacities must lie between 0 and window_size.")
        if eta < 0:
            raise ValueError("eta must be non-negative.")
        self.target_rates = capacities / window_size
        self.eta = eta
        self.shadow_prices = np.zeros_like(self.target_rates)

    def adjust(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, dtype="float64") - self.shadow_prices

    def update(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions, dtype="float64")
        if actions.shape != self.target_rates.shape:
            raise ValueError("actions must contain one value per DSP.")
        self.shadow_prices = np.maximum(
            0.0,
            self.shadow_prices + self.eta * (actions - self.target_rates),
        )


class LinearUCBPolicy(RoutingPolicy):
    """Independent ridge-regression UCB model for each DSP."""

    def __init__(
        self,
        n_dsps: int,
        n_features: int,
        *,
        alpha: float = 0.5,
        ridge: float = 1.0,
        warm_start: int = 5,
        pacer: ShadowPricePacer | None = None,
    ):
        super().__init__(n_dsps)
        if n_features <= 0 or alpha < 0 or ridge <= 0 or warm_start < 0:
            raise ValueError("Invalid linear-policy hyperparameters.")
        self.n_features = n_features
        self.alpha = alpha
        self.ridge = ridge
        self.warm_start = warm_start
        self.pacer = pacer
        identity = np.eye(n_features, dtype="float64") / ridge
        self.inverse_design = np.repeat(identity[None, :, :], n_dsps, axis=0)
        self.reward_sums = np.zeros((n_dsps, n_features), dtype="float64")
        self.estimates = np.zeros((n_dsps, n_features), dtype="float64")
        self.observations = np.zeros(n_dsps, dtype="int64")
        self._current_context: np.ndarray | None = None

    def select(self, context: object) -> np.ndarray:
        x = _as_dense(context, self.n_features)
        self._current_context = x
        means = self.estimates @ x
        uncertainty = np.sqrt(
            np.maximum(
                np.einsum("i,jik,k->j", x, self.inverse_design, x),
                0.0,
            )
        )
        scores = means + self.alpha * uncertainty
        if self.pacer is not None:
            scores = self.pacer.adjust(scores)
        return (scores > 0.0) | (self.observations < self.warm_start)

    def update(self, context: object, feedback: ObservedFeedback) -> None:
        x = (
            self._current_context
            if self._current_context is not None
            else _as_dense(context, self.n_features)
        )
        for dsp, reward in zip(feedback.dsp_indices, feedback.net_rewards):
            inverse = self.inverse_design[dsp]
            projected = inverse @ x
            denominator = 1.0 + x @ projected
            self.inverse_design[dsp] = inverse - np.outer(projected, projected) / denominator
            self.reward_sums[dsp] += float(reward) * x
            self.estimates[dsp] = self.inverse_design[dsp] @ self.reward_sums[dsp]
            self.observations[dsp] += 1
        self._current_context = None

    def on_actions(self, actions: np.ndarray) -> None:
        if self.pacer is not None:
            self.pacer.update(actions)


class GreedyLinearPolicy(LinearUCBPolicy):
    """Online ridge regression without an uncertainty bonus."""

    def __init__(
        self,
        n_dsps: int,
        n_features: int,
        *,
        ridge: float = 1.0,
        warm_start: int = 5,
        pacer: ShadowPricePacer | None = None,
    ):
        super().__init__(
            n_dsps,
            n_features,
            alpha=0.0,
            ridge=ridge,
            warm_start=warm_start,
            pacer=pacer,
        )


class DiscountedLinUCBPolicy(RoutingPolicy):
    """Batched discounted LinUCB for non-stationary DSP rewards.

    Sufficient statistics are refreshed every ``update_interval`` events. This keeps
    matrix inversions off the request path while weighting recent observations more
    heavily than old ones.
    """

    def __init__(
        self,
        n_dsps: int,
        n_features: int,
        *,
        alpha: float = 0.5,
        gamma: float = 0.995,
        ridge: float = 1.0,
        warm_start: int = 5,
        update_interval: int = 250,
        pacer: ShadowPricePacer | None = None,
    ):
        super().__init__(n_dsps)
        if n_features <= 0 or alpha < 0 or ridge <= 0 or warm_start < 0:
            raise ValueError("Invalid discounted-policy hyperparameters.")
        if not 0.0 < gamma <= 1.0 or update_interval <= 0:
            raise ValueError("gamma must be in (0, 1] and update_interval positive.")
        self.n_features = n_features
        self.alpha = alpha
        self.gamma = gamma
        self.ridge = ridge
        self.warm_start = warm_start
        self.update_interval = update_interval
        self.pacer = pacer
        identity = np.eye(n_features, dtype="float64")
        self.design = np.repeat((ridge * identity)[None, :, :], n_dsps, axis=0)
        self.inverse_design = np.repeat((identity / ridge)[None, :, :], n_dsps, axis=0)
        self.reward_sums = np.zeros((n_dsps, n_features), dtype="float64")
        self.estimates = np.zeros((n_dsps, n_features), dtype="float64")
        self.observations = np.zeros(n_dsps, dtype="int64")
        self._step = 0
        self._last_refresh = 0
        self._pending: list[tuple[int, int, np.ndarray, float]] = []
        self._current_context: np.ndarray | None = None

    def _refresh(self) -> None:
        elapsed = self._step - self._last_refresh
        if elapsed <= 0:
            return
        decay = self.gamma**elapsed
        identity = np.eye(self.n_features, dtype="float64")
        self.design = (
            decay * self.design
            + (1.0 - decay) * self.ridge * identity[None, :, :]
        )
        self.reward_sums *= decay
        for observation_step, dsp, x, reward in self._pending:
            weight = self.gamma ** (self._step - observation_step)
            self.design[dsp] += weight * np.outer(x, x)
            self.reward_sums[dsp] += weight * reward * x
        self._pending.clear()
        for dsp in range(self.n_dsps):
            self.inverse_design[dsp] = np.linalg.inv(self.design[dsp])
            self.estimates[dsp] = self.inverse_design[dsp] @ self.reward_sums[dsp]
        self._last_refresh = self._step

    def select(self, context: object) -> np.ndarray:
        self._step += 1
        if self._step - self._last_refresh >= self.update_interval:
            self._refresh()
        x = _as_dense(context, self.n_features)
        self._current_context = x
        means = self.estimates @ x
        uncertainty = np.sqrt(
            np.maximum(
                np.einsum("i,jik,k->j", x, self.inverse_design, x),
                0.0,
            )
        )
        scores = means + self.alpha * uncertainty
        if self.pacer is not None:
            scores = self.pacer.adjust(scores)
        return (scores > 0.0) | (self.observations < self.warm_start)

    def update(self, context: object, feedback: ObservedFeedback) -> None:
        x = (
            self._current_context
            if self._current_context is not None
            else _as_dense(context, self.n_features)
        )
        for dsp, reward in zip(feedback.dsp_indices, feedback.net_rewards):
            self._pending.append((self._step, int(dsp), x.copy(), float(reward)))
            self.observations[dsp] += 1
        self._current_context = None

    def on_actions(self, actions: np.ndarray) -> None:
        if self.pacer is not None:
            self.pacer.update(actions)
