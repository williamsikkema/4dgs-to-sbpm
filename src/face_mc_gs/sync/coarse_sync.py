"""Integer-frame lag via normalized cross-correlation of 1D signals."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
def _normalize(z: NDArray[np.floating]) -> NDArray[np.floating]:
    z = np.asarray(z, dtype=np.float64)
    z = z - np.nanmean(z)
    s = np.nanstd(z)
    if s < 1e-12:
        return np.zeros_like(z)
    return z / s


def cross_correlation_lag(
    ref: NDArray[np.floating],
    other: NDArray[np.floating],
    max_lag: int,
) -> tuple[int, float]:
    """
    Find integer lag `lag` such that other aligned to ref minimizes MSE in overlap.
    Positive lag: other is shifted right (delayed) relative to ref.
    """
    ref = np.asarray(ref, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    n = min(len(ref), len(other))
    ref = ref[:n]
    other = other[:n]
    ref_n = _normalize(ref)
    other_n = _normalize(other)
    best_lag = 0
    best_corr = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            a = ref_n[lag:]
            b = other_n[:-lag]
        elif lag < 0:
            a = ref_n[:lag]
            b = other_n[-lag:]
        else:
            a = ref_n
            b = other_n
        if len(a) < 4:
            continue
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        if c > best_corr:
            best_corr = c
            best_lag = lag
    return best_lag, best_corr


def combine_lags(
    lags: list[int],
    weights: list[float] | None,
    method: Literal["median", "weighted_mean"] = "median",
) -> int:
    if weights is None:
        weights = [1.0] * len(lags)
    if method == "median":
        return int(np.round(np.median(lags)))
    w = np.asarray(weights, dtype=np.float64)
    w = w / (np.sum(w) + 1e-12)
    return int(round(float(np.dot(np.asarray(lags, dtype=np.float64), w))))
