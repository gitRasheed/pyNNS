"""Stochastic dominance routines matching NNS' discrete SD conventions.

Dominance uses strict floating-point comparisons with no tolerance, plus R's
curve equality guard: equal LPM/CDF curves are non-dominance even when samples
differ below meaningful double precision. Efficient-set output follows the R
C++ routine's LPM-at-global-maximum ordering and original-index tie break.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pynns.core import _as_1d_values, lpm


def fsd(x: NDArray[np.float64], y: NDArray[np.float64]) -> int:
    """First-order stochastic dominance."""
    x_values = _as_sd_values(x, "x")
    y_values = _as_sd_values(y, "y")
    return _sd_result(x_values, y_values, 1)


def ssd(x: NDArray[np.float64], y: NDArray[np.float64]) -> int:
    """Second-order stochastic dominance."""
    x_values = _as_sd_values(x, "x")
    y_values = _as_sd_values(y, "y")
    return _sd_result(x_values, y_values, 2)


def tsd(x: NDArray[np.float64], y: NDArray[np.float64]) -> int:
    """Third-order stochastic dominance."""
    x_values = _as_sd_values(x, "x")
    y_values = _as_sd_values(y, "y")
    return _sd_result(x_values, y_values, 3)


def sd_efficient_set(returns: NDArray[np.float64], degree: int) -> list[int]:
    """Return indices of non-dominated columns at the requested SD degree."""
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("returns must be a 2D array.")
    if values.shape[0] == 0:
        raise ValueError("returns must have at least one row.")
    if not 1 <= degree <= 3:
        raise ValueError("degree must be 1, 2, or 3.")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must contain only finite values.")

    if values.shape[1] == 0:
        return []

    tmax = float(np.max(values))
    order = sorted(
        range(values.shape[1]),
        key=lambda index: (float(lpm(degree, tmax, values[:, index])), index),
    )

    keep: list[int] = []
    for index in order:
        candidate = values[:, index]
        dominated = any(_dominates(values[:, kept], candidate, degree) for kept in keep)
        if not dominated:
            keep.append(index)
    return keep


def _sd_result(x: NDArray[np.float64], y: NDArray[np.float64], degree: int) -> int:
    if _dominates(x, y, degree):
        return 1
    if _dominates(y, x, degree):
        return -1
    return 0


def _dominates(x: NDArray[np.float64], y: NDArray[np.float64], degree: int) -> bool:
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    if np.array_equal(np.sort(x), np.sort(y)):
        return False
    if np.min(x) < np.min(y):
        return False
    if degree > 1 and np.mean(x) < np.mean(y):
        return False

    grid = np.sort(np.concatenate((x, y)))
    x_lpm = _dominance_curve(x, grid, degree)
    y_lpm = _dominance_curve(y, grid, degree)
    if np.array_equal(x_lpm, y_lpm):
        return False
    return bool(not np.any(x_lpm > y_lpm))


def _dominance_curve(
    values: NDArray[np.float64],
    grid: NDArray[np.float64],
    degree: int,
) -> NDArray[np.float64]:
    if degree == 1:
        return np.asarray(lpm(0, grid, values), dtype=np.float64)
    return np.asarray(lpm(degree - 1, grid, values), dtype=np.float64)


def _as_sd_values(x: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    values = _as_1d_values(x)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values
