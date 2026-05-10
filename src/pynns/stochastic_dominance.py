"""Stochastic dominance routines matching NNS' discrete SD conventions.

Dominance uses strict floating-point comparisons with no tolerance, plus R's
curve equality guard: equal LPM/CDF curves are non-dominance even when samples
differ below meaningful double precision. Efficient-set output follows the R
C++ routine's LPM-at-global-maximum ordering and original-index tie break.
"""

from __future__ import annotations

from collections.abc import Iterator

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

    sorted_values = np.sort(values, axis=0)
    curves = _sd_curve_table(sorted_values, degree)
    minimums = sorted_values[0, :]
    means = np.mean(values, axis=0)
    tmax = float(np.max(values))
    order_lpm = _lpm_at_target(values, tmax, degree)
    order = sorted(
        range(values.shape[1]),
        key=lambda index: (order_lpm[index], index),
    )

    keep: list[int] = []
    for index in order:
        dominated = any(
            _dominates_from_curves(
                kept,
                index,
                degree,
                curves,
                sorted_values,
                minimums,
                means,
            )
            for kept in keep
        )
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


def _dominates_from_curves(
    x_index: int,
    y_index: int,
    degree: int,
    curves: NDArray[np.float64],
    sorted_values: NDArray[np.float64],
    minimums: NDArray[np.float64],
    means: NDArray[np.float64],
) -> bool:
    if np.array_equal(sorted_values[:, x_index], sorted_values[:, y_index]):
        return False
    if minimums[x_index] < minimums[y_index]:
        return False
    if degree > 1 and means[x_index] < means[y_index]:
        return False

    x_curve = curves[:, x_index]
    y_curve = curves[:, y_index]
    if np.array_equal(x_curve, y_curve):
        return False
    return bool(not np.any(x_curve > y_curve))


def _sd_curve_table(
    sorted_values: NDArray[np.float64],
    degree: int,
) -> NDArray[np.float64]:
    grid = np.unique(sorted_values.reshape(-1))
    observations, columns = sorted_values.shape
    curves = np.empty((grid.size, columns), dtype=np.float64)

    if degree == 1:
        _fill_cdf_curves(curves, grid, sorted_values)
        return curves

    prefix1 = _prefix_sum(sorted_values)
    if degree == 2:
        _fill_lpm_degree1_curves(curves, grid, sorted_values, prefix1, observations)
        return curves

    prefix2 = _prefix_sum(sorted_values * sorted_values)
    _fill_lpm_degree2_curves(curves, grid, sorted_values, prefix1, prefix2, observations)
    return curves


def _fill_cdf_curves(
    curves: NDArray[np.float64],
    grid: NDArray[np.float64],
    sorted_values: NDArray[np.float64],
) -> None:
    observations = sorted_values.shape[0]
    for start, stop in _grid_chunks(grid.size, sorted_values.shape[1]):
        thresholds = grid[start:stop]
        for index in range(sorted_values.shape[1]):
            counts = np.searchsorted(sorted_values[:, index], thresholds, side="right")
            curves[start:stop, index] = counts / observations


def _fill_lpm_degree1_curves(
    curves: NDArray[np.float64],
    grid: NDArray[np.float64],
    sorted_values: NDArray[np.float64],
    prefix1: NDArray[np.float64],
    observations: int,
) -> None:
    for start, stop in _grid_chunks(grid.size, sorted_values.shape[1]):
        thresholds = grid[start:stop]
        for index in range(sorted_values.shape[1]):
            counts = np.searchsorted(sorted_values[:, index], thresholds, side="right")
            sums1 = prefix1[counts, index]
            curves[start:stop, index] = (counts * thresholds - sums1) / observations


def _fill_lpm_degree2_curves(
    curves: NDArray[np.float64],
    grid: NDArray[np.float64],
    sorted_values: NDArray[np.float64],
    prefix1: NDArray[np.float64],
    prefix2: NDArray[np.float64],
    observations: int,
) -> None:
    for start, stop in _grid_chunks(grid.size, sorted_values.shape[1]):
        thresholds = grid[start:stop]
        for index in range(sorted_values.shape[1]):
            counts = np.searchsorted(sorted_values[:, index], thresholds, side="right")
            sums1 = prefix1[counts, index]
            sums2 = prefix2[counts, index]
            curves[start:stop, index] = (
                counts * thresholds * thresholds - 2.0 * thresholds * sums1 + sums2
            ) / observations


def _prefix_sum(values: NDArray[np.float64]) -> NDArray[np.float64]:
    prefix = np.empty((values.shape[0] + 1, values.shape[1]), dtype=np.float64)
    prefix[0, :] = 0.0
    np.cumsum(values, axis=0, out=prefix[1:, :])
    return prefix


def _lpm_at_target(
    values: NDArray[np.float64],
    target: float,
    degree: int,
) -> NDArray[np.float64]:
    deviations = np.maximum(0.0, target - values)
    if degree > 1:
        deviations = deviations**degree
    return np.asarray(np.mean(deviations, axis=0), dtype=np.float64)


def _grid_chunks(grid_size: int, columns: int) -> Iterator[tuple[int, int]]:
    max_intermediate_bytes = 100 * 1024 * 1024
    row_bytes = columns * np.dtype(np.float64).itemsize
    chunk_size = max(1, max_intermediate_bytes // max(row_bytes, 1))
    for start in range(0, grid_size, chunk_size):
        yield start, min(start + chunk_size, grid_size)


def _as_sd_values(x: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    values = _as_1d_values(x)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values
