from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def lpm(
    degree: float,
    target: float | NDArray[np.float64],
    x: NDArray[np.float64],
) -> float | NDArray[np.float64]:
    values = _as_1d_values(x)
    targets = _as_targets(target)
    degree = _as_degree(degree)

    if degree == 0:
        moments = np.mean(values <= targets[:, np.newaxis], axis=1)
        return _result_for_target(moments, target)

    moments = np.mean(np.maximum(0.0, targets[:, np.newaxis] - values) ** degree, axis=1)
    return _result_for_target(moments, target)


def lpm_ratio(
    degree: float,
    target: float | NDArray[np.float64],
    x: NDArray[np.float64],
) -> float | NDArray[np.float64]:
    if degree == 0:
        return lpm(degree, target, x)

    lower = lpm(degree, target, x)
    upper = upm(degree, target, x)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.asarray(lower) / (np.asarray(lower) + np.asarray(upper))
    return _result_for_target(np.asarray(ratio).reshape(-1), target)


def upm(
    degree: float,
    target: float | NDArray[np.float64],
    x: NDArray[np.float64],
) -> float | NDArray[np.float64]:
    values = _as_1d_values(x)
    targets = _as_targets(target)
    degree = _as_degree(degree)

    if degree == 0:
        moments = np.mean(values > targets[:, np.newaxis], axis=1)
        return _result_for_target(moments, target)

    moments = np.mean(np.maximum(0.0, values - targets[:, np.newaxis]) ** degree, axis=1)
    return _result_for_target(moments, target)


def upm_ratio(
    degree: float,
    target: float | NDArray[np.float64],
    x: NDArray[np.float64],
) -> float | NDArray[np.float64]:
    if degree == 0:
        return upm(degree, target, x)

    lower = lpm(degree, target, x)
    upper = upm(degree, target, x)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.asarray(upper) / (np.asarray(lower) + np.asarray(upper))
    return _result_for_target(np.asarray(ratio).reshape(-1), target)


def _as_1d_values(x: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("x must be 1D.")
    if values.size == 0:
        raise ValueError("x must be non-empty.")
    return values


def _as_targets(target: float | NDArray[np.float64]) -> NDArray[np.float64]:
    targets = np.asarray(target, dtype=np.float64)
    if targets.ndim == 0:
        return targets.reshape(1)
    if targets.ndim != 1:
        raise ValueError("target must be scalar or 1D.")
    return targets


def _as_degree(degree: float) -> float:
    degree = float(degree)
    if degree < 0:
        raise ValueError("degree must be non-negative.")
    return degree


def _result_for_target(
    moments: NDArray[np.float64],
    target: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    if np.asarray(target).ndim == 0:
        return float(moments[0])
    return moments
