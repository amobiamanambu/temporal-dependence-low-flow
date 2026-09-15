"""Proper-score utilities used by the example and its tests."""

from __future__ import annotations

import numpy as np


def _paired_finite(observed: object, probability: object) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(observed, dtype=float)
    p = np.asarray(probability, dtype=float)
    if y.shape != p.shape:
        raise ValueError("observed and probability must have the same shape")
    valid = np.isfinite(y) & np.isfinite(p)
    if not np.any(valid):
        raise ValueError("no finite observed–forecast pairs")
    y = y[valid]
    p = p[valid]
    if np.any((y < 0) | (y > 1)):
        raise ValueError("observations must be binary values in {0, 1}")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("forecast probabilities must lie in [0, 1]")
    return y, p


def brier_score(observed: object, probability: object) -> float:
    """Return the mean squared error of a binary probabilistic forecast."""
    y, p = _paired_finite(observed, probability)
    return float(np.mean((p - y) ** 2))


def relative_skill_percent(candidate_score: float, reference_score: float) -> float:
    """Return relative score reduction; positive values favor the candidate."""
    candidate = float(candidate_score)
    reference = float(reference_score)
    if not np.isfinite(candidate) or not np.isfinite(reference):
        raise ValueError("scores must be finite")
    if reference <= 0:
        raise ValueError("reference_score must be positive")
    return 100.0 * (1.0 - candidate / reference)
