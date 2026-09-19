"""Calibration metrics. These, not accuracy, are the success criteria."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> float:
    """Standard binned ECE over top-1 confidence vs empirical accuracy."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def brier_multiclass(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between the distribution and the one-hot label."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    return float(-np.log(probs[np.arange(len(labels)), labels] + eps).mean())


def summarize(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    """probs: (n, k) distributions; labels: (n,) true class indices."""
    top1 = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    correct = (top1 == labels).astype(float)
    return {
        "n": len(labels),
        "accuracy": float(correct.mean()),
        "ece": expected_calibration_error(confidences, correct, n_bins),
        "brier": brier_multiclass(probs, labels),
        "nll": nll(probs, labels),
        "mean_top1_prob": float(confidences.mean()),
    }


def summarize_binary(p_yes: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Binary (noul) case: p_yes: (n,) P(yes); labels: (n,) 0/1."""
    probs = np.stack([1.0 - p_yes, p_yes], axis=1)
    return summarize(probs, labels.astype(int))
