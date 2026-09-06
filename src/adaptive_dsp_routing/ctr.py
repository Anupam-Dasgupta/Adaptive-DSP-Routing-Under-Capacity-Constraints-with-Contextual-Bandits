"""Chronological CTR baseline used as a real-data relevance prior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


@dataclass(frozen=True)
class CTRModelResult:
    """Out-of-sample predictions and standard binary-classification metrics."""

    probabilities: np.ndarray
    roc_auc: float
    log_loss: float
    brier_score: float
    observed_ctr: float
    mean_predicted: np.ndarray
    fraction_positive: np.ndarray

    def summary(self) -> dict[str, float]:
        return {
            "roc_auc": self.roc_auc,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "observed_ctr": self.observed_ctr,
            "mean_predicted_ctr": float(self.probabilities.mean()),
        }


def _binary_labels(values: object, name: str) -> np.ndarray:
    labels = np.asarray(values, dtype="int64").ravel()
    if labels.size == 0 or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(f"{name} must contain binary 0/1 labels.")
    return labels


def fit_ctr_model(
    training_features: object,
    training_labels: object,
    evaluation_features: object,
    evaluation_labels: object,
    *,
    regularization: float = 1.0,
) -> CTRModelResult:
    """Fit on an earlier prefix and score a later chronological suffix."""

    if regularization <= 0:
        raise ValueError("regularization must be positive.")
    y_train = _binary_labels(training_labels, "training_labels")
    y_evaluation = _binary_labels(evaluation_labels, "evaluation_labels")
    if training_features.shape[0] != y_train.size:
        raise ValueError("Training features and labels have different lengths.")
    if evaluation_features.shape[0] != y_evaluation.size:
        raise ValueError("Evaluation features and labels have different lengths.")
    if np.unique(y_train).size < 2 or np.unique(y_evaluation).size < 2:
        raise ValueError("Both chronological partitions must contain both classes.")

    model = LogisticRegression(
        C=regularization,
        fit_intercept=False,
        max_iter=200,
        solver="liblinear",
        random_state=0,
    )
    model.fit(training_features, y_train)
    probabilities = model.predict_proba(evaluation_features)[:, 1]
    fraction_positive, mean_predicted = calibration_curve(
        y_evaluation,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )
    return CTRModelResult(
        probabilities=probabilities,
        roc_auc=float(roc_auc_score(y_evaluation, probabilities)),
        log_loss=float(log_loss(y_evaluation, probabilities)),
        brier_score=float(brier_score_loss(y_evaluation, probabilities)),
        observed_ctr=float(y_evaluation.mean()),
        mean_predicted=mean_predicted,
        fraction_positive=fraction_positive,
    )


def append_ctr_feature(features: object, probabilities: object) -> object:
    """Append one out-of-sample CTR score to each routing context."""

    values = np.asarray(probabilities, dtype="float64").ravel()
    if features.shape[0] != values.size:
        raise ValueError("Features and CTR probabilities have different lengths.")
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("CTR probabilities must be finite values in [0, 1].")
    if sparse.issparse(features):
        return sparse.hstack(
            [features, sparse.csr_matrix(values[:, None])], format="csr"
        )
    return np.column_stack([np.asarray(features), values])
