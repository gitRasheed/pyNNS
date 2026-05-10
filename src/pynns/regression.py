from __future__ import annotations

import math
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from pynns._helpers import _fast_lm, _is_fcl
from pynns.central_tendencies import nns_mode
from pynns.copula import _target
from pynns.dependence import _dpm_nd, _gravity, nns_dep
from pynns.part import NoiseReduction, nns_part
from pynns.pm_matrix import pm_matrix

Order = int | Literal["max"] | None


def nns_reg(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    factor_2_dummy: bool = False,
    order: Order = None,
    dim_red_method: object | None = None,
    tau: object | None = None,
    type: str | None = None,
    point_est: NDArray[np.float64] | float | None = None,
    return_values: bool = True,
    plot: bool = False,
    plot_regions: bool = False,
    residual_plot: bool = False,
    confidence_interval: float | None = None,
    threshold: float = 0.0,
    n_best: object | None = None,
    smooth: bool = False,
    noise_reduction: NoiseReduction = "off",
    dist: str = "L2",
    ncores: int | None = None,
    point_only: bool = False,
    multivariate_call: bool = False,
) -> dict[str, Any]:
    """Univariate numeric port of R's NNS.reg."""
    del tau, return_values, plot, plot_regions, residual_plot, threshold, n_best, dist, ncores

    x_values, y_values = _validate_univariate_inputs(x, y, factor_2_dummy)
    _reject_deferred_paths(
        x_values,
        dim_red_method=dim_red_method,
        type=type,
        point_est=point_est,
        confidence_interval=confidence_interval,
        smooth=smooth,
        point_only=point_only,
        multivariate_call=multivariate_call,
    )
    noise = _validate_noise_reduction(noise_reduction)
    point_values = _as_point_est(point_est)

    dependence = _regression_dependence(x_values, y_values)
    dep_order = _dep_reduced_order(dependence, order, y_values.size)
    part_map = _partition_for_regression(x_values, y_values, dependence, dep_order, order, noise)
    nns_ids = part_map["dt"]["quadrant"].astype(str)

    rp = part_map["regression.points"]
    rp_x, rp_y = _initial_regression_points(rp["x"], rp["y"], x_values)
    rp_x, rp_y = _add_central_point(rp_x, rp_y, x_values, y_values)
    rp_x, rp_y = _add_endpoint_points(rp_x, rp_y, x_values, y_values, dependence)
    rp_x = np.minimum(np.max(x_values), np.maximum(np.min(x_values), rp_x))
    rp_y = np.minimum(np.max(y_values), np.maximum(np.min(y_values), rp_y))

    coeff = _coefficients(rp_x, rp_y, x_values, y_values)
    estimate = _fitted_values(x_values, y_values, rp_x, rp_y, coeff, order)

    if point_values is None:
        point_est_y = np.array([], dtype=np.float64)
    else:
        point_est_y = _predict_points(point_values, x_values, y_values, rp_x, rp_y, coeff)

    if isinstance(order, str):
        rp_out_x, rp_out_y = _consolidate_points(part_map["dt"]["x"], part_map["dt"]["y"])
    else:
        rp_out_x, rp_out_y = rp_x, rp_y

    fitted = _fitted_table(x_values, y_values, estimate, nns_ids, coeff)
    se = float(math.sqrt(float(np.sum((estimate - y_values) ** 2)) / (y_values.size - 1)))
    r2 = _r2(y_values, estimate)

    return {
        "R2": r2,
        "SE": se,
        "Prediction.Accuracy": None,
        "equation": None,
        "x.star": None,
        "derivative": {
            "Coefficient": coeff["Coefficient"],
            "X.Lower.Range": coeff["X.Lower.Range"],
            "X.Upper.Range": coeff["X.Upper.Range"],
        },
        "Point.est": point_est_y,
        "pred.int": None,
        "regression.points": {"x": rp_out_x, "y": rp_out_y},
        "Fitted.xy": fitted,
    }


def _validate_univariate_inputs(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    factor_2_dummy: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if factor_2_dummy and (_is_fcl(x) or _is_fcl(y)):
        raise NotImplementedError(
            "factor_2_dummy=True with non-numeric inputs is deferred until "
            "the factor path is ported."
        )
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise NotImplementedError("matrix x input requires NNS.M.reg, which is not yet ported.")
    if x_values.size == 0:
        raise ValueError("x and y must be non-empty.")
    if x_values.size != y_values.size:
        raise ValueError("x and y must have the same length.")
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("x and y must contain only finite values.")
    return x_values, y_values


def _reject_deferred_paths(
    x: NDArray[np.float64],
    *,
    dim_red_method: object | None,
    type: str | None,
    point_est: NDArray[np.float64] | float | None,
    confidence_interval: float | None,
    smooth: bool,
    point_only: bool,
    multivariate_call: bool,
) -> None:
    if x.ndim != 1:
        raise NotImplementedError("matrix x input requires NNS.M.reg, which is not yet ported.")
    if dim_red_method is not None:
        raise NotImplementedError(
            "dim_red_method paths are deferred until dimension reduction is ported."
        )
    if smooth:
        raise NotImplementedError(
            "smooth=True requires the smoothing-spline path, deferred to a later batch."
        )
    if confidence_interval is not None:
        raise NotImplementedError(
            "confidence_interval output is deferred until the regression interval path is ported."
        )
    if type is not None:
        raise NotImplementedError(
            "classification type paths are deferred to a later regression batch."
        )
    if point_only:
        raise NotImplementedError(
            "point_only is deferred until multivariate regression callers are ported."
        )
    if multivariate_call:
        raise NotImplementedError("multivariate_call is deferred until NNS.M.reg is ported.")
    if point_est is not None and np.asarray(point_est).ndim > 1:
        raise NotImplementedError("matrix point_est requires NNS.M.reg, which is not yet ported.")


def _validate_noise_reduction(value: str) -> NoiseReduction:
    noise = value.lower()
    if noise not in {"off", "mean", "median", "mode", "mode_class"}:
        raise ValueError(
            "noise_reduction must be one of 'mean', 'median', 'mode', 'mode_class', 'off'."
        )
    return cast(NoiseReduction, noise)


def _as_point_est(point_est: NDArray[np.float64] | float | None) -> NDArray[np.float64] | None:
    if point_est is None:
        return None
    values = np.asarray(point_est, dtype=np.float64)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1:
        raise NotImplementedError("matrix point_est requires NNS.M.reg, which is not yet ported.")
    if not np.all(np.isfinite(values)):
        raise ValueError("point_est must contain only finite values.")
    return values


def _regression_dependence(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    dep = nns_dep(x, y, asym=True)["Dependence"]
    try:
        scaled = np.column_stack(
            (
                _rescale_01(x),
                _rescale_01(x),
                _rescale_01(y),
            )
        )
        dep = float(np.mean(np.array([dep, _nns_copula_matrix(scaled)], dtype=np.float64)))
    except (ValueError, FloatingPointError):
        dep = float(dep)
    if not math.isfinite(dep):
        dep = 0.1
    return dep


def _rescale_01(values: NDArray[np.float64]) -> NDArray[np.float64]:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax == vmin:
        return np.zeros_like(values, dtype=np.float64)
    return (values - vmin) / (vmax - vmin)


def _nns_copula_matrix(values: NDArray[np.float64]) -> float:
    target = _target(values, None, None)
    discrete_pm = pm_matrix(0.0, 0.0, target, values, pop_adj=False)
    upper = np.triu_indices(values.shape[1], k=1)
    discrete_co = float(np.sum(discrete_pm["cupm"][upper]) + np.sum(discrete_pm["clpm"][upper]))
    if discrete_co == 1.0 or discrete_co == 0.0:
        return 1.0

    continuous_pm = pm_matrix(1.0, 1.0, target, values, pop_adj=True, norm=True)
    continuous_co = float(
        np.sum(continuous_pm["cupm"][upper]) + np.sum(continuous_pm["clpm"][upper])
    )
    n_vars = values.shape[1]
    indep_co = 0.25 * (n_vars * n_vars - n_vars)
    discrete_dep = min(max(abs(discrete_co - indep_co) / indep_co, 0.0), 1.0)
    continuous_dep = min(max(abs(continuous_co - indep_co) / indep_co, 0.0), 1.0)

    discrete_d = _dpm_nd(values, target, 0.0, norm=True)
    continuous_d = _dpm_nd(values, target, 1.0, norm=True)
    indep_d = 1.0 - (0.5**n_vars)
    n_dim_discrete = abs(discrete_d - indep_d) / indep_d
    n_dim_continuous = abs(continuous_d - indep_d) / indep_d
    return math.sqrt((discrete_dep + continuous_dep + n_dim_discrete + n_dim_continuous) / 4.0)


def _dep_reduced_order(dependence: float, order: Order, n: int) -> int | Literal["max"]:
    if order == "max":
        return "max"
    if order is None:
        rounded_dep = math.floor(dependence * 10.0)
        if n < 100:
            rounded_dep = math.floor(rounded_dep / 2.0)
        return max(1, rounded_dep)
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError("order must be an integer, 'max', or None.")
    return max(1, _round_half_up(float(order)))


def _round_half_up(value: float) -> int:
    floor = math.floor(value)
    return floor if value - floor < 0.5 else math.ceil(value)


def _partition_for_regression(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    dependence: float,
    dep_order: int | Literal["max"],
    requested_order: Order,
    noise: NoiseReduction,
) -> dict[str, Any]:
    if dependence == 1.0 or dep_order == "max":
        if requested_order is None or dep_order == "max":
            return _max_order_part_map(x, y)
        return cast(dict[str, Any], nns_part(x, y, order=int(dep_order), obs_req=0))
    return cast(
        dict[str, Any],
        nns_part(
            x,
            y,
            noise_reduction=noise,
            order=int(dep_order),
            type="XONLY",
            obs_req=0,
            min_obs_stop=True,
        ),
    )


def _max_order_part_map(x: NDArray[np.float64], y: NDArray[np.float64]) -> dict[str, Any]:
    quadrants = np.full(x.size, "q", dtype=str)
    seed_map = nns_part(x, y, order=1, obs_req=0)
    return {
        "order": x.size,
        "dt": {"x": x.copy(), "y": y.copy(), "quadrant": quadrants, "prior.quadrant": quadrants},
        "regression.points": seed_map["regression.points"],
    }


def _initial_regression_points(
    point_x: NDArray[np.float64],
    point_y: NDArray[np.float64],
    x: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    clamped_x = np.minimum(float(np.max(x)), np.maximum(point_x, float(np.min(x))))
    return _consolidate_points(clamped_x, point_y)


def _add_central_point(
    rp_x: NDArray[np.float64],
    rp_y: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_points = rp_x.size
    row_positions = np.arange(1, n_points + 1)
    rows = np.array(
        [math.floor(np.median(row_positions)), math.ceil(np.median(row_positions))],
        dtype=np.int64,
    )
    central_x_values = rp_x[rows - 1]
    if np.unique(rows).size > 1:
        mask = (x >= central_x_values[0]) & (x <= central_x_values[1])
        central_y = _gravity(y[mask])
    else:
        central_y = float(rp_y[rows[0] - 1])
    central_x = _gravity(central_x_values)
    return _consolidate_points(
        np.concatenate((rp_x, np.array([central_x], dtype=np.float64))),
        np.concatenate((rp_y, np.array([central_y], dtype=np.float64))),
    )


def _add_endpoint_points(
    rp_x: NDArray[np.float64],
    rp_y: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    dependence: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    interior = (rp_x != float(np.min(x))) & (rp_x != float(np.max(x)))
    endpoint_base_x = rp_x[interior]
    if endpoint_base_x.size == 0:
        endpoint_base_x = rp_x
    if dependence >= 1.0:
        min_y = float(y[np.flatnonzero(x == np.min(x))[0]])
        max_y = float(y[np.flatnonzero(x == np.max(x))[0]])
    else:
        min_y = _endpoint_y(x, y, endpoint_base_x, low=True, dependence=dependence)
        max_y = _endpoint_y(x, y, endpoint_base_x, low=False, dependence=dependence)
    rp_x = rp_x[interior]
    rp_y = rp_y[interior]
    return _consolidate_points(
        np.concatenate((rp_x, np.array([float(np.min(x)), float(np.max(x))]))),
        np.concatenate((rp_y, np.array([min_y, max_y]))),
    )


def _endpoint_y(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    rp_x: NDArray[np.float64],
    *,
    low: bool,
    dependence: float,
) -> float:
    boundary = float(np.min(x) if low else np.max(x))
    reg_range = float(np.min(rp_x) if low else np.max(rp_x))
    mid_range = float(np.mean([boundary, reg_range]))
    boundary_mask = x <= reg_range if low else x >= reg_range
    mid_mask = x <= mid_range if low else x >= mid_range
    y_boundary = y[boundary_mask]
    y_mid = y[mid_mask]
    x_mid = x[mid_mask]
    unique_x_mid = np.unique(x_mid).size

    if unique_x_mid > 1 and y_boundary.size > 5:
        if dependence < 0.95 and y_boundary.size > 1 and y_mid.size > 1:
            fit_boundary = _edge_lm_fit(x[boundary_mask], y_boundary, low=low)
            fit_mid = _edge_lm_fit(x_mid, y_mid, low=low)
            return float(
                (fit_boundary * y_boundary.size + fit_mid * y_mid.size)
                / (y_boundary.size + y_mid.size)
            )
        boundary_values = y[x == boundary]
        return float(np.mean(np.unique(boundary_values)))

    return float(np.mean(np.unique([_gravity(y[x == boundary])])))


def _edge_lm_fit(x: NDArray[np.float64], y: NDArray[np.float64], *, low: bool) -> float:
    intercept, slope = _fast_lm(x, y)
    edge_x = float(np.min(x) if low else np.max(x))
    return intercept + slope * edge_x


def _consolidate_points(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    finite = np.isfinite(x) & np.isfinite(y)
    x_values = x[finite].astype(np.float64)
    y_values = y[finite].astype(np.float64)
    order = np.lexsort((y_values, x_values))
    x_values = x_values[order]
    y_values = y_values[order]
    unique_x, inverse = np.unique(x_values, return_inverse=True)
    out_y = np.empty(unique_x.size, dtype=np.float64)
    for idx in range(unique_x.size):
        out_y[idx] = _gravity(y_values[inverse == idx])
    return unique_x, out_y


def _coefficients(
    rp_x: NDArray[np.float64],
    rp_y: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    if rp_x.size > 1:
        rise = np.diff(rp_y)
        run = np.diff(rp_x)
    else:
        rise = np.array([float(np.max(y) - np.min(y))], dtype=np.float64)
        run_value = float(np.max(x) - np.min(x))
        if run_value == 0.0:
            run_value = 1.0
        run = np.array([run_value], dtype=np.float64)
        rp_x = np.repeat(rp_x, 3)
        rp_y = np.repeat(rp_y, 3)

    with np.errstate(divide="ignore", invalid="ignore"):
        coef = rise / run
    lower = rp_x[:-1] if rp_x.size > 1 else np.array([float(np.unique(rp_x)[0])])
    upper = rp_x[1:] if rp_x.size > 1 else np.array([float(np.unique(rp_x)[0])])
    if np.unique(upper).size <= 1:
        coef = np.zeros_like(upper, dtype=np.float64)
        lower = np.asarray(np.unique(upper), dtype=np.float64)
        upper = np.asarray(np.unique(upper), dtype=np.float64)
    coef = np.where(np.isposinf(coef), 1.0, coef)
    coef = np.where(np.isfinite(coef), coef, 0.0)
    matrix = np.column_stack((coef, lower, upper))
    _, first = np.unique(matrix, axis=0, return_index=True)
    unique = matrix[np.sort(first)]
    return {
        "Coefficient": unique[:, 0],
        "X.Lower.Range": unique[:, 1],
        "X.Upper.Range": unique[:, 2],
    }


def _fitted_values(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    rp_x: NDArray[np.float64],
    rp_y: NDArray[np.float64],
    coeff: dict[str, NDArray[np.float64]],
    order: Order,
) -> NDArray[np.float64]:
    if (order is not None and _is_fcl(order)) or (
        order is not None and not isinstance(order, str) and order >= y.size
    ):
        return y.copy()
    reg_idx = _find_interval(x, rp_x, rightmost_closed=False)
    coef_idx = _find_interval(x, coeff["X.Lower.Range"], rightmost_closed=False)
    return (x - rp_x[reg_idx]) * coeff["Coefficient"][coef_idx] + rp_y[reg_idx]


def _predict_points(
    point_est: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    rp_x: NDArray[np.float64],
    rp_y: NDArray[np.float64],
    coeff: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    reg_idx = _find_interval(point_est, rp_x, rightmost_closed=True)
    coef_idx = _find_interval(point_est, coeff["X.Lower.Range"], rightmost_closed=True)
    out = (point_est - rp_x[reg_idx]) * coeff["Coefficient"][coef_idx] + rp_y[reg_idx]
    if np.any((point_est > np.max(x)) | (point_est < np.min(x))):
        _, first = np.unique(coeff["Coefficient"], return_index=True)
        unique_coef = coeff["Coefficient"][np.sort(first)]
        upper_slope = float(np.mean(unique_coef[-2:]))
        lower_slope = float(np.mean(unique_coef[:2]))
        upper_mask = point_est > np.max(x)
        lower_mask = point_est < np.min(x)
        if np.any(upper_mask):
            out[upper_mask] = (
                (point_est[upper_mask] - float(np.max(x))) * upper_slope
                + float(nns_mode(y[np.flatnonzero(x == np.max(x))]))
            )
        if np.any(lower_mask):
            out[lower_mask] = (
                (point_est[lower_mask] - float(np.min(x))) * lower_slope
                + float(nns_mode(y[np.flatnonzero(x == np.min(x))]))
            )
    return out.astype(np.float64)


def _find_interval(
    values: NDArray[np.float64],
    breaks: NDArray[np.float64],
    *,
    rightmost_closed: bool,
) -> NDArray[np.int64]:
    idx = np.searchsorted(breaks, values, side="right")
    if rightmost_closed:
        idx = np.where(values == breaks[-1], breaks.size, idx)
    idx = idx - 1
    return np.clip(idx, 0, breaks.size - 1).astype(np.int64)


def _fitted_table(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    estimate: NDArray[np.float64],
    nns_ids: NDArray[np.str_],
    coeff: dict[str, NDArray[np.float64]],
) -> dict[str, NDArray[np.float64] | NDArray[np.str_]]:
    y_hat = estimate.copy()
    if np.any(~np.isfinite(y_hat)):
        replacement = _gravity(y_hat[np.isfinite(y_hat)])
        y_hat[~np.isfinite(y_hat)] = replacement
    gradient_idx = _find_interval(x, coeff["X.Lower.Range"], rightmost_closed=False)
    gradient = coeff["Coefficient"][gradient_idx]
    residuals = y_hat - y
    standard_errors = np.empty_like(y_hat)
    for grad in np.unique(gradient):
        mask = gradient == grad
        denom = max(1, int(np.sum(mask)) - 1)
        standard_errors[mask] = math.sqrt(float(np.sum((y_hat[mask] - y[mask]) ** 2)) / denom)
    return {
        "x": x.copy(),
        "y": y.copy(),
        "y.hat": y_hat,
        "NNS.ID": nns_ids,
        "gradient": gradient,
        "residuals": residuals,
        "standard.errors": standard_errors,
    }


def _r2(y: NDArray[np.float64], y_hat: NDArray[np.float64]) -> float:
    y_mean = float(np.mean(y))
    numerator = float(np.sum((y - y_mean) * (y_hat - y_mean)) ** 2)
    denominator = float(np.sum((y - y_mean) ** 2) * np.sum((y_hat - y_mean) ** 2))
    return numerator / denominator if denominator > 0.0 else float("nan")
