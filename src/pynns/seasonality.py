from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

SeasonalityResult = dict[str, object]


def nns_seas(
    variable: NDArray[np.float64],
    modulo: int | list[int] | NDArray[np.int64] | None = None,
    mod_only: bool = True,
    plot: bool = False,
) -> SeasonalityResult:
    """Seasonality test matching R's NNS.seas non-plotting path."""
    del plot
    values = _validate_variable(variable)
    n = values.size
    if n < 5:
        return _result(
            np.array([0], dtype=np.int64),
            np.array([0.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        )

    mean_var = _mean(values)
    use_cv = mean_var != 0.0
    var_cov = (
        abs(_sample_sd(values) / mean_var)
        if use_cv
        else abs(_acf1(values)) ** -1.0
    )
    if not np.isfinite(var_cov):
        var_cov = math.inf

    periods: list[int] = []
    covs: list[float] = []
    half_n = n // 2
    variable_1 = values[:-1]
    variable_2 = variable_1[:-1]
    for period in range(1, half_n + 1):
        t0 = _cv_or_fallback(_reverse_step(values, period), use_cv, var_cov)
        t1 = _cv_or_fallback(_reverse_step(variable_1, period), use_cv, var_cov)
        t2 = _cv_or_fallback(_reverse_step(variable_2, period), use_cv, var_cov)
        if t0 <= var_cov and t1 <= var_cov and t2 <= var_cov:
            periods.append(period)
            covs.append((t0 + t1 + t2) / 3.0)

    if periods:
        period_arr = np.asarray(periods, dtype=np.int64)
        coef_arr = np.asarray(covs, dtype=np.float64)
        var_arr = np.full(period_arr.size, var_cov, dtype=np.float64)
        period_arr, coef_arr, var_arr = _sort_periods(period_arr, coef_arr, var_arr)
    else:
        period_arr = np.array([1], dtype=np.int64)
        coef_arr = np.array([var_cov], dtype=np.float64)
        var_arr = np.array([var_cov], dtype=np.float64)

    if modulo is not None:
        period_arr, coef_arr, var_arr = _apply_modulo(
            period_arr,
            coef_arr,
            var_arr,
            _as_modulo(modulo),
            mod_only=mod_only,
            var_cov=var_cov,
        )

    period_arr, coef_arr, var_arr = _strict_cap(period_arr, coef_arr, var_arr, n, var_cov)
    return _result(period_arr, coef_arr, var_arr)


def _validate_variable(variable: NDArray[np.float64]) -> NDArray[np.float64]:
    try:
        values = np.asarray(variable, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Variable must be numeric") from exc
    if values.ndim != 1:
        values = values.reshape(-1)
    if values.size == 0:
        raise ValueError("Variable must be numeric and non-empty")
    if np.any(np.isnan(values)):
        raise ValueError("You have some missing values, please address.")
    if np.any(np.isinf(values)):
        raise ValueError("Infinite values not allowed")
    return values


def _sample_sd(values: NDArray[np.float64]) -> float:
    if values.size < 2:
        return math.nan
    mean = _mean(values)
    ss = 0.0
    for value in values:
        delta = float(value) - mean
        ss += delta * delta
    return math.sqrt(ss / float(values.size - 1))


def _acf1(values: NDArray[np.float64]) -> float:
    n = values.size
    if n < 2:
        return math.nan
    mean = _mean(values)
    numerator = 0.0
    denom = 0.0
    for index in range(1, n):
        numerator += (float(values[index]) - mean) * (float(values[index - 1]) - mean)
    for value in values:
        delta = float(value) - mean
        denom += delta * delta
    if denom == 0.0:
        return math.nan
    return numerator / denom


def _cv_or_fallback(values: NDArray[np.float64], use_cv: bool, var_cov: float) -> float:
    if values.size < 2:
        return var_cov
    if use_cv:
        mean = _mean(values)
        sd = _sample_sd(values)
        stat = abs(sd / mean) if mean != 0.0 else math.inf
    else:
        acf = _acf1(values)
        stat = abs(acf) ** -1.0
    if not np.isfinite(stat):
        return var_cov
    return float(stat)


def _mean(values: NDArray[np.float64]) -> float:
    total = 0.0
    for value in values:
        total += float(value)
    return total / float(values.size)


def _reverse_step(values: NDArray[np.float64], step: int) -> NDArray[np.float64]:
    return values[::-step]


def _sort_periods(
    periods: NDArray[np.int64],
    coef: NDArray[np.float64],
    var_cov: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    order = np.lexsort((periods, coef))
    return periods[order], coef[order], var_cov[order]


def _as_modulo(modulo: int | list[int] | NDArray[np.int64]) -> NDArray[np.int64]:
    values = np.asarray(modulo, dtype=np.int64).reshape(-1)
    return values


def _apply_modulo(
    periods: NDArray[np.int64],
    coef: NDArray[np.float64],
    var_arr: NDArray[np.float64],
    modulo: NDArray[np.int64],
    *,
    mod_only: bool,
    var_cov: float,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    per_set: set[int] = set()
    for period in periods:
        for mod in modulo:
            m = int(mod)
            if m <= 0:
                continue
            remainder = int(period) % m
            minus = int(period) - remainder
            plus = int(period) + (m - remainder)
            if minus > 0:
                per_set.add(minus)
            if plus > 0:
                per_set.add(plus)

    if mod_only:
        current = {int(period) for period in periods}
        out_periods: list[int] = []
        out_coef: list[float] = []
        for period, cv in zip(periods, coef, strict=True):
            if int(period) in per_set:
                out_periods.append(int(period))
                out_coef.append(float(cv))
        for period in sorted(per_set):
            if period not in current:
                out_periods.append(period)
                out_coef.append(var_cov)
        if not out_periods:
            return (
                np.array([1], dtype=np.int64),
                np.array([var_cov], dtype=np.float64),
                np.array([var_cov], dtype=np.float64),
            )
    else:
        per_set.add(1)
        current = {int(period) for period in periods}
        out_periods = [int(period) for period in periods]
        out_coef = [float(cv) for cv in coef]
        for period in sorted(per_set):
            if period not in current:
                out_periods.append(period)
                out_coef.append(var_cov)

    period_arr = np.asarray(out_periods, dtype=np.int64)
    coef_arr = np.asarray(out_coef, dtype=np.float64)
    var_out = np.full(period_arr.size, var_cov, dtype=np.float64)
    return _sort_periods(period_arr, coef_arr, var_out)


def _strict_cap(
    periods: NDArray[np.int64],
    coef: NDArray[np.float64],
    var_arr: NDArray[np.float64],
    n: int,
    var_cov: float,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    keep = periods < (n / 2.0)
    if np.any(keep):
        return _sort_periods(periods[keep], coef[keep], var_arr[keep])
    return (
        np.array([1], dtype=np.int64),
        np.array([var_cov], dtype=np.float64),
        np.array([var_cov], dtype=np.float64),
    )


def _result(
    periods: NDArray[np.int64],
    coef: NDArray[np.float64],
    var_cov: NDArray[np.float64],
) -> SeasonalityResult:
    return {
        "all.periods": {
            "Period": periods,
            "Coefficient.of.Variation": coef,
            "Variable.Coefficient.of.Variation": var_cov,
        },
        "best.period": int(periods[0]),
        "periods": periods.copy(),
    }
