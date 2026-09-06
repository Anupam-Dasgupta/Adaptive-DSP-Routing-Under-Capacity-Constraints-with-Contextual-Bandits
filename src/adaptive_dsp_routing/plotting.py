"""Standard Matplotlib output for experiment traces."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluation import EpisodeResult
from .ctr import CTRModelResult


def plot_comparison(
    results: Mapping[str, EpisodeResult],
    output_path: str | Path,
    *,
    drift_time: int | None = None,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for name, result in results.items():
        axes[0].plot(result.cumulative_net, label=name)
        axes[1].plot(result.cumulative_pseudo_regret, label=name)
    if drift_time is not None:
        for axis in axes:
            axis.axvline(drift_time, color="black", linestyle="--", label="drift")
    axes[0].set_title("Cumulative realized net profit")
    axes[0].set_ylabel("Net profit")
    axes[1].set_title("Cumulative pseudo-regret vs constrained oracle")
    axes[1].set_ylabel("Pseudo-regret")
    axes[1].set_xlabel("Event")
    for axis in axes:
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys())
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def plot_capacity_sweep(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot mean net profit with one-standard-deviation error bars."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    for policy in summary["policy"].unique():
        rows = summary[summary["policy"] == policy].sort_values("capacity_ratio")
        axis.errorbar(
            rows["capacity_ratio"],
            rows["net_profit_mean"],
            yerr=rows["net_profit_std"].fillna(0.0),
            marker="o",
            capsize=4,
            label=policy,
        )
    axis.set_title("Net profit under DSP capacity scarcity")
    axis.set_xlabel("Mean capacity ratio")
    axis.set_ylabel("Net profit, mean ± one standard deviation")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def plot_seed_traces(
    results: Mapping[str, Sequence[EpisodeResult]],
    output_path: str | Path,
    *,
    drift_time: int,
) -> Path:
    """Plot mean cumulative profit and regret across paired seeds."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for name, runs in results.items():
        profit = np.vstack([run.cumulative_net for run in runs])
        regret = np.vstack([run.cumulative_pseudo_regret for run in runs])
        x = np.arange(profit.shape[1])
        for axis, values in zip(axes, (profit, regret)):
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
            axis.plot(x, mean, label=name)
            axis.fill_between(x, mean - std, mean + std, alpha=0.15)
    for axis in axes:
        axis.axvline(drift_time, color="black", linestyle="--", label="drift")
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys())
    axes[0].set_title("Cumulative realized net profit across paired seeds")
    axes[0].set_ylabel("Net profit")
    axes[1].set_title("Cumulative pseudo-regret across paired seeds")
    axes[1].set_ylabel("Pseudo-regret")
    axes[1].set_xlabel("Evaluation event")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def plot_ctr_calibration(
    result: CTRModelResult, output_path: str | Path
) -> Path:
    """Plot out-of-sample predicted CTR against observed click frequency."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 6))
    upper = min(
        1.0,
        1.1
        * max(
            float(result.mean_predicted.max()),
            float(result.fraction_positive.max()),
        ),
    )
    axis.plot([0, upper], [0, upper], color="black", linestyle="--", label="ideal")
    axis.plot(
        result.mean_predicted,
        result.fraction_positive,
        marker="o",
        label="logistic CTR model",
    )
    axis.set_title("Chronological CTR calibration")
    axis.set_xlabel("Mean predicted CTR")
    axis.set_ylabel("Observed click frequency")
    axis.set_xlim(0.0, upper)
    axis.set_ylim(0.0, upper)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination
