from __future__ import annotations

import math
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from pynns.central_tendencies import nns_mode
from pynns.dependence import _gravity
from pynns.distance import KValue, nns_distance
from pynns.part import NoiseReduction
from pynns.regression import Order, nns_reg
from pynns.regression import _nns_copula_matrix as _copula_matrix

NBest = int | Literal["all"] | None
MRegResult = dict[str, Any]


def nns_m_reg(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    factor_2_dummy: bool = False,
    order: Order = None,
    n_best: NBest = None,
    type: str | None = None,
    point_est: NDArray[np.float64] | None = None,
    point_only: bool = False,
    plot: bool = False,
    residual_plot: bool = True,
    location: object | None = None,
    noise_reduction: NoiseReduction = "off",
    dist: str = "L2",
    return_values: bool = False,
    plot_regions: bool = False,
    ncores: int | None = None,
    confidence_interval: float | None = None,
) -> MRegResult:
    """Multivariate numeric regression matching R's non-plotting NNS.M.reg path."""
    del plot, residual_plot, location, dist, return_values, plot_regions, ncores
    if type is not None:
        raise NotImplementedError(
            "type='class' classification for NNS.M.reg is deferred to a future batch."
        )
    if confidence_interval is not None:
        raise NotImplementedError(
            "confidence_interval output is deferred until the multivariate interval path is ported."
        )

    x_values, y_values = _validate_inputs(x, y, factor_2_dummy)
    point_values, point_is_matrix = _validate_point_est(point_est, x_values.shape[1])
    noise = _validate_noise(noise_reduction)

    reg_points_matrix = _regression_points_matrix(x_values, y_values, order, noise, factor_2_dummy)
    if order is None or isinstance(order, int):
        reg_points_matrix = _unique_rows_preserve_order(reg_points_matrix)
    if order == "max" and n_best is None:
        n_best = 1

    nns_id_components = _find_interval_matrix(x_values, reg_points_matrix)
    nns_ids = _join_ids(nns_id_components)
    rpm, fitted_y, residuals = _rpm_and_fitted(
        x_values,
        y_values,
        nns_ids,
        noise,
        order_is_numeric=order is None or isinstance(order, int),
    )

    k = _resolve_n_best(n_best, x_values, y_values, rpm)
    if _k_as_count(k, rpm.shape[0]) > 1 and not point_only:
        fitted_y = np.array([nns_distance(rpm, row, k, None) for row in x_values], dtype=np.float64)
        residuals = fitted_y - y_values

    if point_values is None:
        point_predictions: NDArray[np.float64] | None = None
    else:
        point_predictions = _predict_points(
            point_values,
            point_is_matrix,
            x_values,
            rpm,
            k,
        )

    if point_only:
        return {"Point.est": _point_output(point_predictions), "RPM": _rpm_dict(rpm)}

    fitted = _fitted_dict(x_values, y_values, fitted_y, nns_ids, residuals)
    return {
        "R2": _r2(y_values, fitted_y),
        "rhs.partitions": _rhs_partitions_dict(reg_points_matrix),
        "RPM": _rpm_dict(rpm),
        "Point.est": _point_output(point_predictions),
        "pred.int": None,
        "Fitted.xy": fitted,
    }


def _validate_inputs(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    factor_2_dummy: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if factor_2_dummy:
        raise NotImplementedError(
            "factor_2_dummy=True with non-numeric inputs is deferred until "
            "the factor path is ported."
        )
    x_values = np.asarray(x, dtype=np.float64)
    if x_values.ndim == 1:
        x_values = x_values.reshape(-1, 1)
    if x_values.ndim != 2:
        raise ValueError("x must be a 2D numeric matrix.")
    y_values = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_values.shape[0] == 0 or x_values.shape[1] == 0:
        raise ValueError("x must be non-empty.")
    if y_values.size != x_values.shape[0]:
        raise ValueError("x and y must have the same row count.")
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("x and y must contain only finite values.")
    return x_values, y_values


def _validate_point_est(
    point_est: NDArray[np.float64] | None,
    n_cols: int,
) -> tuple[NDArray[np.float64] | None, bool]:
    if point_est is None:
        return None, False
    values = np.asarray(point_est, dtype=np.float64)
    is_matrix = values.ndim == 2
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("point_est must be a vector or 2D matrix.")
    if values.shape[1] != n_cols:
        raise ValueError("point_est must have the same column count as x.")
    if not np.all(np.isfinite(values)):
        raise ValueError("point_est must contain only finite values.")
    return values, is_matrix


def _validate_noise(noise_reduction: str) -> NoiseReduction:
    noise = noise_reduction.lower()
    if noise not in {"off", "mean", "median", "mode", "mode_class"}:
        raise ValueError(
            "noise_reduction must be one of 'mean', 'median', 'mode', 'mode_class', 'off'."
        )
    return cast(NoiseReduction, noise)


def _regression_points_matrix(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    order: Order,
    noise: NoiseReduction,
    factor_2_dummy: bool,
) -> NDArray[np.float64]:
    if order == "max":
        return x.copy()
    columns: list[NDArray[np.float64]] = []
    max_len = 0
    for col in range(x.shape[1]):
        result = nns_reg(
            x[:, col],
            y,
            factor_2_dummy=factor_2_dummy,
            order=order,
            type=None,
            noise_reduction=noise,
            plot=False,
            multivariate_call=True,
            ncores=1,
        )
        points = np.asarray(result["x"], dtype=np.float64)
        columns.append(points)
        max_len = max(max_len, points.size)

    out = np.full((max_len, x.shape[1]), np.nan, dtype=np.float64)
    for col, points in enumerate(columns):
        out[: points.size, col] = points
    return out


def _unique_rows_preserve_order(values: NDArray[np.float64]) -> NDArray[np.float64]:
    seen: set[tuple[float, ...]] = set()
    rows: list[NDArray[np.float64]] = []
    for row in values:
        key = tuple(float(v) if np.isfinite(v) else math.nan for v in row)
        if key not in seen:
            seen.add(key)
            rows.append(row)
    return np.vstack(rows) if rows else values


def _find_interval_matrix(
    x: NDArray[np.float64],
    reg_points_matrix: NDArray[np.float64],
) -> NDArray[np.int64]:
    out = np.empty(x.shape, dtype=np.int64)
    for col in range(x.shape[1]):
        breaks = np.sort(reg_points_matrix[:, col][np.isfinite(reg_points_matrix[:, col])])
        out[:, col] = np.searchsorted(breaks, x[:, col], side="right")
    return out


def _join_ids(components: NDArray[np.int64]) -> NDArray[np.str_]:
    return np.asarray([".".join(str(int(v)) for v in row) for row in components], dtype=str)


def _rpm_and_fitted(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    nns_ids: NDArray[np.str_],
    noise: NoiseReduction,
    *,
    order_is_numeric: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    obs = np.arange(y.size)
    sorted_order = np.lexsort((obs, nns_ids.astype(str)))
    sorted_ids = nns_ids[sorted_order].astype(str)
    unique_ids, first, inverse_sorted = np.unique(
        sorted_ids,
        return_index=True,
        return_inverse=True,
    )

    sorted_matrix = np.column_stack((x[sorted_order], y[sorted_order]))
    group_values = np.empty((unique_ids.size, x.shape[1] + 1), dtype=np.float64)
    for group_index in range(unique_ids.size):
        rows = sorted_matrix[inverse_sorted == group_index]
        group_values[group_index] = _aggregate_rows(rows, noise, order_is_numeric)

    original_group_index = np.searchsorted(unique_ids.astype(str), nns_ids.astype(str))
    initial_yhat = group_values[original_group_index, -1].copy()
    residuals = initial_yhat - y
    bias = np.empty_like(residuals)
    for group_id in np.unique(nns_ids.astype(str)):
        mask = nns_ids.astype(str) == group_id
        bias[mask] = _gravity(residuals[mask])
    fitted_y = initial_yhat - bias
    residuals = fitted_y - y

    rpm = group_values[np.argsort(first)]
    return rpm, fitted_y, residuals


def _aggregate_rows(
    rows: NDArray[np.float64],
    noise: NoiseReduction,
    order_is_numeric: bool,
) -> NDArray[np.float64]:
    if not order_is_numeric:
        return np.asarray(rows[0], dtype=np.float64)
    if noise == "mean":
        return np.asarray(np.mean(rows, axis=0), dtype=np.float64)
    if noise == "median":
        return np.median(rows, axis=0)
    if noise == "mode":
        return np.array([float(nns_mode(rows[:, col])) for col in range(rows.shape[1])])
    if noise == "mode_class":
        return np.array(
            [float(nns_mode(rows[:, col], discrete=True)) for col in range(rows.shape[1])]
        )
    return np.array([_gravity(rows[:, col]) for col in range(rows.shape[1])])


def _resolve_n_best(
    n_best: NBest,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    rpm: NDArray[np.float64],
) -> KValue:
    if n_best == "all":
        return "all"
    if n_best is not None:
        return max(1, int(n_best))
    dependence = _copula_matrix(np.column_stack((x, y)))
    return max(1, math.floor((1.0 - dependence) * math.sqrt(x.shape[1])))


def _k_as_count(k: KValue, row_count: int) -> int:
    if k == "all":
        return row_count
    return int(k)


def _predict_points(
    point_est: NDArray[np.float64],
    point_is_matrix: bool,
    x: NDArray[np.float64],
    rpm: NDArray[np.float64],
    k: KValue,
) -> NDArray[np.float64]:
    minimums = np.min(x, axis=0)
    maximums = np.max(x, axis=0)
    central = np.array([_gravity(rpm[:, col]) for col in range(x.shape[1])], dtype=np.float64)
    out = np.empty(point_est.shape[0], dtype=np.float64)
    outsider_rows = np.flatnonzero(np.any((point_est < minimums) | (point_est > maximums), axis=1))
    for row_index, point in enumerate(point_est):
        outsiders = (point < minimums) | (point > maximums)
        if not np.any(outsiders):
            out[row_index] = nns_distance(rpm, point, k, None)
            continue
        if point_is_matrix and outsider_rows.size == 1:
            # Installed R drops dimensions for one outsider row in the multi-point path:
            # apply(as.matrix(point.est[i, ]), 1, f) passes scalar elements to f and
            # vector assignment keeps the first result. Match that behavior.
            scalar_point = np.full(point.shape, point[0], dtype=np.float64)
            out[row_index] = _outside_prediction(scalar_point, minimums, maximums, central, rpm, k)
            continue
        out[row_index] = _outside_prediction(point, minimums, maximums, central, rpm, k)
    return out if point_is_matrix else out[:1]


def _outside_prediction(
    point: NDArray[np.float64],
    minimums: NDArray[np.float64],
    maximums: NDArray[np.float64],
    central: NDArray[np.float64],
    rpm: NDArray[np.float64],
    k: KValue,
) -> float:
    boundary = np.minimum(np.maximum(point, minimums), maximums)
    mid = (boundary + central) / 2.0
    mid_2 = (boundary + mid) / 2.0
    boundary_est = nns_distance(rpm, boundary, k, None)
    gradients = []
    for compare in (central, mid, mid_2):
        distance = float(np.sqrt(np.sum((boundary - compare) ** 2)))
        if distance == 0.0:
            gradients.append(0.0)
        else:
            gradients.append((boundary_est - nns_distance(rpm, compare, k, None)) / distance)
    last_gradient = float(np.dot(np.asarray(gradients), np.array([3.0, 2.0, 1.0])) / 6.0)
    last_distance = float(np.sqrt(np.sum((point - boundary) ** 2)))
    return last_distance * last_gradient + boundary_est


def _point_output(point_predictions: NDArray[np.float64] | None) -> NDArray[np.float64] | None:
    if point_predictions is None:
        return None
    return point_predictions


def _rpm_dict(rpm: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    out = {f"V{col + 1}": rpm[:, col] for col in range(rpm.shape[1] - 1)}
    out["y.hat"] = rpm[:, -1]
    return out


def _rhs_partitions_dict(values: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    return {f"x{col + 1}": values[:, col] for col in range(values.shape[1])}


def _fitted_dict(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    yhat: NDArray[np.float64],
    nns_ids: NDArray[np.str_],
    residuals: NDArray[np.float64],
) -> dict[str, NDArray[np.float64] | NDArray[np.str_]]:
    out: dict[str, NDArray[np.float64] | NDArray[np.str_]] = {
        f"V{col + 1}": x[:, col].copy() for col in range(x.shape[1])
    }
    out["y"] = y.copy()
    out["y.hat"] = yhat
    out["NNS.ID"] = nns_ids
    out["residuals"] = residuals
    return out


def _r2(y: NDArray[np.float64], yhat: NDArray[np.float64]) -> float:
    y_mean = float(np.mean(y))
    numerator = float(np.sum((y - y_mean) * (yhat - y_mean)) ** 2)
    denominator = float(np.sum((y - y_mean) ** 2) * np.sum((yhat - y_mean) ** 2))
    return numerator / denominator if denominator > 0.0 else float("nan")
