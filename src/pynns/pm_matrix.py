from __future__ import annotations

from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from pynns.co_moments import co_lpm, co_upm, d_lpm, d_upm
from pynns.core import _as_degree

Target: TypeAlias = float | None | Literal["mean"] | NDArray[np.float64]
PMMatrixResult: TypeAlias = dict[str, NDArray[np.float64]]


def pm_matrix(
    lpm_degree: float,
    upm_degree: float,
    target: Target,
    variable: NDArray[np.float64],
    pop_adj: bool,
    norm: bool = False,
) -> PMMatrixResult:
    lpm_degree = _as_degree(lpm_degree)
    upm_degree = _as_degree(upm_degree)
    values = _as_matrix(variable)
    targets = _as_target(target, values)

    columns = values.shape[1]
    cupm = np.empty((columns, columns), dtype=np.float64)
    dupm = np.empty((columns, columns), dtype=np.float64)
    dlpm = np.empty((columns, columns), dtype=np.float64)
    clpm = np.empty((columns, columns), dtype=np.float64)

    adjust = values.shape[0] / (values.shape[0] - 1) if values.shape[0] > 1 else 1.0
    should_adjust = pop_adj and values.shape[0] > 1 and lpm_degree > 0 and upm_degree > 0

    for i in range(columns):
        x = values[:, i]
        target_x = targets[i]
        for j in range(columns):
            y = values[:, j]
            target_y = targets[j]
            clpm[i, j] = co_lpm(lpm_degree, x, y, target_x, target_y)
            cupm[i, j] = co_upm(upm_degree, x, y, target_x, target_y)
            dlpm[i, j] = d_lpm(lpm_degree, upm_degree, x, y, target_x, target_y)
            dupm[i, j] = d_upm(lpm_degree, upm_degree, x, y, target_x, target_y)

    if should_adjust:
        clpm *= adjust
        cupm *= adjust
        dlpm *= adjust
        dupm *= adjust

    if norm:
        total = cupm + dupm + dlpm + clpm
        np.divide(cupm, total, out=cupm, where=total > 0.0)
        np.divide(dupm, total, out=dupm, where=total > 0.0)
        np.divide(dlpm, total, out=dlpm, where=total > 0.0)
        np.divide(clpm, total, out=clpm, where=total > 0.0)
        cupm[total <= 0.0] = 0.0
        dupm[total <= 0.0] = 0.0
        dlpm[total <= 0.0] = 0.0
        clpm[total <= 0.0] = 0.0

    cov_matrix = cupm + clpm - dupm - dlpm
    return {
        "cupm": cupm,
        "dupm": dupm,
        "dlpm": dlpm,
        "clpm": clpm,
        "cov.matrix": cov_matrix,
    }


def _as_matrix(variable: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(variable, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("variable must be 2D.")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("variable must be non-empty.")
    return values


def _as_target(target: Target, variable: NDArray[np.float64]) -> NDArray[np.float64]:
    if target is None or isinstance(target, str):
        return cast(NDArray[np.float64], np.mean(variable, axis=0))

    targets = np.asarray(target, dtype=np.float64)
    if targets.ndim == 0:
        return np.full(variable.shape[1], float(targets), dtype=np.float64)
    if targets.ndim != 1:
        raise ValueError("target must be 1D.")
    if targets.size != variable.shape[1]:
        raise ValueError("variable matrix cols != target vector length.")
    return targets
