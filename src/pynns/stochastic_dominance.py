"""Stochastic dominance routines matching NNS' discrete SD conventions.

Dominance uses strict floating-point comparisons with no tolerance, plus R's
curve equality guard: equal LPM/CDF curves are non-dominance even when samples
differ below meaningful double precision. Efficient-set output follows the R
C++ routine's LPM-at-global-maximum ordering and original-index tie break.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pynns.core import _as_1d_values, lpm


@dataclass(frozen=True)
class _SDPrecomputed:
    values: NDArray[np.float64]
    sorted_values: NDArray[np.float64]
    curves: NDArray[np.float64]
    minimums: NDArray[np.float64]
    means: NDArray[np.float64]
    identical: NDArray[np.bool_]


def fsd(x: NDArray[np.float64], y: NDArray[np.float64]) -> int:
    """First-order stochastic dominance."""
    x_values = _as_sd_values(x, "x")
    y_values = _as_sd_values(y, "y")
    return _sd_result(x_values, y_values, 1)


def fsd_uni(x: NDArray[np.float64], y: NDArray[np.float64], type: str = "discrete") -> int:
    """Unidirectional first-order stochastic dominance: 1 if x dominates y, else 0."""
    x_values = _as_sd_values(x, "x")
    y_values = _as_sd_values(y, "y")
    discrete = type.lower() != "continuous"
    return int(_dominates_uni(x_values, y_values, 1, discrete=discrete))


def ssd(x: NDArray[np.float64], y: NDArray[np.float64]) -> int:
    """Second-order stochastic dominance."""
    x_values = _as_sd_values(x, "x")
    y_values = _as_sd_values(y, "y")
    return _sd_result(x_values, y_values, 2)


def ssd_uni(x: NDArray[np.float64], y: NDArray[np.float64]) -> int:
    """Unidirectional second-order stochastic dominance: 1 if x dominates y, else 0."""
    x_values = _as_sd_values(x, "x")
    y_values = _as_sd_values(y, "y")
    return int(_dominates_uni(x_values, y_values, 2, discrete=True))


def tsd(x: NDArray[np.float64], y: NDArray[np.float64]) -> int:
    """Third-order stochastic dominance."""
    x_values = _as_sd_values(x, "x")
    y_values = _as_sd_values(y, "y")
    return _sd_result(x_values, y_values, 3)


def tsd_uni(x: NDArray[np.float64], y: NDArray[np.float64]) -> int:
    """Unidirectional third-order stochastic dominance: 1 if x dominates y, else 0."""
    x_values = _as_sd_values(x, "x")
    y_values = _as_sd_values(y, "y")
    return int(_dominates_uni(x_values, y_values, 3, discrete=True))


def nns_sd_cluster(
    data: NDArray[np.float64],
    degree: int = 1,
    type: str = "discrete",
    min_cluster: int = 1,
    dendrogram: bool = False,
    names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Cluster variables by iteratively peeling stochastic-dominance efficient sets."""
    if dendrogram:
        raise NotImplementedError(
            "nns_sd_cluster(dendrogram=True) requires R hclust-compatible dendrogram "
            "output, which is not yet ported."
        )

    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("data must be a 2D array.")
    if values.shape[0] == 0:
        raise ValueError("data must have at least one row.")
    if not 1 <= int(degree) <= 3:
        raise ValueError("degree must be 1, 2, or 3.")
    if not np.all(np.isfinite(values)):
        raise ValueError("data must contain only finite values.")
    type_value = _sd_type_value(int(degree), type)
    discrete = int(degree) != 1 or type_value != "continuous"
    min_cluster = int(min_cluster)
    if min_cluster < 0:
        raise ValueError("min_cluster must be non-negative.")

    column_count = values.shape[1]
    if names is None:
        all_names = [f"X_{index + 1}" for index in range(column_count)]
    else:
        all_names = [str(name) for name in names]
        if len(all_names) != column_count:
            raise ValueError("names length must match the number of data columns.")

    precomputed = _precompute_sd_table(values, int(degree), discrete=discrete)
    active = list(range(column_count))
    clusters: dict[str, list[str]] = {}
    iteration = 1

    while len(active) > min_cluster:
        sd_set_indices = _sd_efficient_active_indices(precomputed, active, int(degree))
        sd_set = [all_names[index] for index in sd_set_indices]
        if not sd_set:
            break

        clusters[f"Cluster_{iteration}"] = sd_set
        remove_indices = set(sd_set_indices)
        active = [index for index in active if index not in remove_indices]
        iteration += 1

        if len(active) <= min_cluster:
            clusters[f"Cluster_{iteration}"] = [all_names[index] for index in active]
            break

    if len(active) > min_cluster and f"Cluster_{iteration}" not in clusters:
        clusters[f"Cluster_{iteration}"] = [all_names[index] for index in active]

    if clusters:
        final_cluster_name = f"Cluster_{len(clusters)}"
        if len(clusters[final_cluster_name]) < min_cluster and len(clusters) > 1:
            previous_cluster_name = f"Cluster_{len(clusters) - 1}"
            clusters[previous_cluster_name].extend(clusters[final_cluster_name])
            del clusters[final_cluster_name]

    return {"Clusters": clusters}


def sd_efficient_set(
    returns: NDArray[np.float64],
    degree: int,
    type: str = "discrete",
) -> list[int]:
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

    type_value = _sd_type_value(degree, type)
    discrete = degree != 1 or type_value != "continuous"

    if values.shape[1] == 0:
        return []

    precomputed = _precompute_sd_table(values, degree, discrete=discrete)
    return _sd_efficient_active_indices(precomputed, list(range(values.shape[1])), degree)


def _sd_efficient_set_names(
    values: NDArray[np.float64],
    degree: int,
    type: str,
    names: Sequence[str],
) -> list[str]:
    return [names[index] for index in sd_efficient_set(values, degree, type=type)]


def _sd_type_value(degree: int, type: str) -> str:
    type_value = type.lower()
    if degree == 1 and type_value in {"discrete", "continuous"}:
        return type_value
    return "discrete"


def _precompute_sd_table(
    values: NDArray[np.float64],
    degree: int,
    *,
    discrete: bool,
) -> _SDPrecomputed:
    sorted_values = np.sort(values, axis=0)
    return _SDPrecomputed(
        values=values,
        sorted_values=sorted_values,
        curves=_sd_curve_table(sorted_values, degree, discrete=discrete),
        minimums=sorted_values[0, :],
        means=np.mean(values, axis=0),
        identical=np.all(
            sorted_values.T[:, np.newaxis, :] == sorted_values.T[np.newaxis, :, :],
            axis=2,
        ),
    )


def _sd_efficient_active_indices(
    precomputed: _SDPrecomputed,
    active: Sequence[int],
    degree: int,
) -> list[int]:
    if not active:
        return []

    active_array = np.asarray(active, dtype=np.intp)
    tmax = float(np.max(precomputed.values[:, active_array]))
    order_lpm = _lpm_at_target(precomputed.values[:, active_array], tmax, degree)
    order = [
        active[int(position)]
        for position in sorted(
            range(len(active)),
            key=lambda position: (order_lpm[position], active[position]),
        )
    ]

    keep: list[int] = []
    for index in order:
        dominated = any(
            _dominates_from_precomputed(kept, index, degree, precomputed)
            for kept in keep
        )
        if not dominated:
            keep.append(index)
    return keep


def _dominates_from_precomputed(
    x_index: int,
    y_index: int,
    degree: int,
    precomputed: _SDPrecomputed,
) -> bool:
    if precomputed.identical[x_index, y_index]:
        return False
    if precomputed.minimums[x_index] < precomputed.minimums[y_index]:
        return False
    if degree > 1 and precomputed.means[x_index] < precomputed.means[y_index]:
        return False

    x_curve = precomputed.curves[:, x_index]
    y_curve = precomputed.curves[:, y_index]
    if np.array_equal(x_curve, y_curve):
        return False
    return bool(not np.any(x_curve > y_curve))


def _sd_result(x: NDArray[np.float64], y: NDArray[np.float64], degree: int) -> int:
    if _dominates(x, y, degree):
        return 1
    if _dominates(y, x, degree):
        return -1
    return 0


def _dominates(x: NDArray[np.float64], y: NDArray[np.float64], degree: int) -> bool:
    return _dominates_uni(x, y, degree, discrete=True)


def _dominates_uni(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    degree: int,
    *,
    discrete: bool,
) -> bool:
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    if np.array_equal(np.sort(x), np.sort(y)):
        return False
    if np.min(x) < np.min(y):
        return False
    if degree > 1 and np.mean(x) < np.mean(y):
        return False

    grid = np.sort(np.concatenate((x, y)))
    x_lpm = _dominance_curve(x, grid, degree, discrete=discrete)
    y_lpm = _dominance_curve(y, grid, degree, discrete=discrete)
    if np.array_equal(x_lpm, y_lpm):
        return False
    return bool(not np.any(x_lpm > y_lpm))


def _dominance_curve(
    values: NDArray[np.float64],
    grid: NDArray[np.float64],
    degree: int,
    *,
    discrete: bool = True,
) -> NDArray[np.float64]:
    if degree == 1:
        if discrete:
            return np.asarray(lpm(0, grid, values), dtype=np.float64)
        lower = np.asarray(lpm(1, grid, values), dtype=np.float64)
        upper = np.mean(np.maximum(0.0, values - grid[:, np.newaxis]), axis=1)
        ratio: NDArray[np.float64] = np.divide(
            lower,
            lower + upper,
            out=np.zeros_like(lower),
            where=(lower + upper) != 0,
        )
        return ratio
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
    *,
    discrete: bool = True,
) -> NDArray[np.float64]:
    grid = np.unique(sorted_values.reshape(-1))
    observations, columns = sorted_values.shape
    curves = np.empty((grid.size, columns), dtype=np.float64)

    if degree == 1:
        if discrete:
            _fill_cdf_curves(curves, grid, sorted_values)
        else:
            _fill_continuous_fsd_curves(curves, grid, sorted_values)
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


def _fill_continuous_fsd_curves(
    curves: NDArray[np.float64],
    grid: NDArray[np.float64],
    sorted_values: NDArray[np.float64],
) -> None:
    observations = sorted_values.shape[0]
    prefix1 = _prefix_sum(sorted_values)
    totals = prefix1[-1, :]
    for start, stop in _grid_chunks(grid.size, sorted_values.shape[1]):
        thresholds = grid[start:stop]
        for index in range(sorted_values.shape[1]):
            counts = np.searchsorted(sorted_values[:, index], thresholds, side="right")
            sums1 = prefix1[counts, index]
            lower = (counts * thresholds - sums1) / observations
            upper = (totals[index] - sums1 - (observations - counts) * thresholds) / observations
            curves[start:stop, index] = np.divide(
                lower,
                lower + upper,
                out=np.zeros_like(lower),
                where=(lower + upper) != 0,
            )


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
