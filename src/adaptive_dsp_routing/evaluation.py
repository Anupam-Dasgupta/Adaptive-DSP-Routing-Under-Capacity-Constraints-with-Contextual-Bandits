"""Evaluation loop, constrained oracle, and compact metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policies import RoutingPolicy
from .simulator import CapacityLimiter, CounterfactualWorld


def constrained_oracle_actions(
    expected_net_rewards: np.ndarray,
    capacities: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Select each DSP's top positive expected opportunities per window."""

    rewards = np.asarray(expected_net_rewards, dtype="float64")
    limits = np.asarray(capacities, dtype="int64")
    if rewards.ndim != 2 or limits.shape != (rewards.shape[1],):
        raise ValueError("Reward and capacity shapes are inconsistent.")
    actions = np.zeros(rewards.shape, dtype=bool)
    for start in range(0, rewards.shape[0], window_size):
        stop = min(start + window_size, rewards.shape[0])
        for dsp, capacity in enumerate(limits):
            positive = np.flatnonzero(rewards[start:stop, dsp] > 0.0)
            if positive.size == 0 or capacity == 0:
                continue
            order = np.argsort(rewards[start:stop, dsp][positive], kind="stable")
            chosen = positive[order[-min(int(capacity), positive.size) :]]
            actions[start + chosen, dsp] = True
    return actions


@dataclass(frozen=True)
class EpisodeResult:
    actions: np.ndarray
    realized_gross: np.ndarray
    routing_cost: np.ndarray
    realized_net: np.ndarray
    expected_net: np.ndarray
    oracle_expected_net: np.ndarray
    capacities: np.ndarray
    window_size: int

    @property
    def cumulative_net(self) -> np.ndarray:
        return np.cumsum(self.realized_net)

    @property
    def cumulative_pseudo_regret(self) -> np.ndarray:
        return np.cumsum(self.oracle_expected_net - self.expected_net)

    def summary(self) -> dict[str, float | int]:
        routes_used = int(self.actions.sum())
        requests_forwarded = int(self.actions.any(axis=1).sum())
        available = 0
        for start in range(0, len(self.actions), self.window_size):
            length = min(self.window_size, len(self.actions) - start)
            available += int(np.minimum(self.capacities, length).sum())
        violations = 0
        for start in range(0, len(self.actions), self.window_size):
            used = self.actions[start : start + self.window_size].sum(axis=0)
            violations += int(np.maximum(used - self.capacities, 0).sum())
        return {
            "events": len(self.actions),
            "routes_used": routes_used,
            "routing_fraction": routes_used / self.actions.size,
            "requests_forwarded": requests_forwarded,
            "request_forward_fraction": requests_forwarded / len(self.actions),
            "gross_value": float(self.realized_gross.sum()),
            "routing_cost": float(self.routing_cost.sum()),
            "net_profit": float(self.realized_net.sum()),
            "pseudo_regret": float(
                self.oracle_expected_net.sum() - self.expected_net.sum()
            ),
            "capacity_utilization": routes_used / available if available else 0.0,
            "capacity_violations": violations,
            "routing_efficiency": (
                float(self.realized_net.sum()) / routes_used if routes_used else 0.0
            ),
        }


def run_policy(
    features: object,
    world: CounterfactualWorld,
    policy: RoutingPolicy,
    *,
    capacities: np.ndarray,
    window_size: int,
) -> EpisodeResult:
    """Run sequentially; update the policy with routed feedback only."""

    if features.shape[0] != world.n_events:
        raise ValueError("Feature rows and world events must match.")
    if policy.n_dsps != world.n_dsps:
        raise ValueError("Policy and world must contain the same number of DSPs.")
    limits = np.asarray(capacities, dtype="int64")
    limiter = CapacityLimiter(limits, window_size)
    oracle = constrained_oracle_actions(
        world.expected_net_rewards, limits, window_size
    )

    actions = np.zeros((world.n_events, world.n_dsps), dtype=bool)
    realized_gross = np.zeros(world.n_events)
    routing_cost = np.zeros(world.n_events)
    realized_net = np.zeros(world.n_events)
    expected_net = np.zeros(world.n_events)
    oracle_expected_net = np.zeros(world.n_events)

    for event_index in range(world.n_events):
        context = features[event_index]
        allowed = limiter.apply(policy.select(context))
        policy.on_actions(allowed)
        feedback = world.reveal(event_index, allowed)
        policy.update(context, feedback)
        actions[event_index] = allowed
        realized_gross[event_index] = feedback.gross_values.sum()
        routing_cost[event_index] = world.costs[feedback.dsp_indices].sum()
        realized_net[event_index] = feedback.net_rewards.sum()
        expected_net[event_index] = world.expected_net_rewards[
            event_index, allowed
        ].sum()
        oracle_expected_net[event_index] = world.expected_net_rewards[
            event_index, oracle[event_index]
        ].sum()

    return EpisodeResult(
        actions=actions,
        realized_gross=realized_gross,
        routing_cost=routing_cost,
        realized_net=realized_net,
        expected_net=expected_net,
        oracle_expected_net=oracle_expected_net,
        capacities=limits,
        window_size=window_size,
    )
