from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pynns.core import lpm_ratio, upm_ratio

_R_OPTIMIZE_TOL = float(np.finfo(float).eps ** 0.25)


def nns_var(
    variables: NDArray[np.float64],
    h: int,
    tau: int | list[int] | list[list[int]] = 1,
    *,
    dim_red_method: str = "cor",
    naive_weights: bool = True,
    obj_fn: Any = None,
    objective: str = "min",
    status: bool = True,
    ncores: int | None = None,
    nowcast: bool = False,
) -> dict[str, Any]:
    """Guarded placeholder for R's NNS.VAR path."""
    del variables, h, tau, dim_red_method, naive_weights, obj_fn, objective, status, ncores, nowcast
    raise NotImplementedError(
        "nns_var default VAR path requires R named-data-frame stack semantics for lagged "
        "variables, which are not yet ported."
    )


def _lag_mtx(
    x: np.ndarray,
    tau: int | Sequence[int] | Sequence[Sequence[int]],
    names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build an R-compatible lag matrix for VAR-style feature construction."""

    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("x must be a 2-D array.")
    n_rows, n_vars = arr.shape

    if isinstance(tau, int):
        lag_by_var: list[list[int]] = [list(range(tau + 1)) for _ in range(n_vars)]
        tau_values = [tau]
        filter_columns = False
    else:
        raw_tau = list(tau)
        if len(raw_tau) != n_vars:
            raise ValueError("tau must have one entry per variable.")

        is_scalar_tau = all(isinstance(item, Integral) for item in raw_tau)
        is_nested_tau = any(
            isinstance(item, Sequence) and not isinstance(item, (str, bytes))
            for item in raw_tau
        )

        if is_nested_tau and not is_scalar_tau:
            lag_by_var = []
            tau_values = []
            for item in raw_tau:
                if not isinstance(item, Sequence):
                    raise ValueError("tau entries must be integer lag vectors.")
                values = [int(value) for value in item]
                lag_by_var.append(values)
                tau_values.extend(values)
        elif is_scalar_tau and not is_nested_tau:
            lag_by_var = []
            tau_values = []
            for item in raw_tau:
                if not isinstance(item, Integral):
                    raise ValueError("tau entries must be integers.")
                lag_by_var.append([int(item)])
                tau_values.append(int(item))
        else:
            raise ValueError("tau entries must be integers or sequences of integers.")
        filter_columns = len(tau_values) > 1

    if len(tau_values) == 0:
        raise ValueError("tau must include at least one lag.")
    if not all(isinstance(value, int) for value in tau_values):
        raise ValueError("tau values must be integers.")
    if not all(value >= 0 for value in tau_values):
        raise ValueError("tau values must be non-negative integers.")

    max_tau = max(tau_values)

    block = max_tau + 1
    lag_matrix = np.empty((n_rows - max_tau, n_vars * block), dtype=np.float64)
    lag_names: list[str] = []
    var_names = list(names) if names is not None else [f"var{idx + 1}" for idx in range(n_vars)]

    for j in range(n_vars):
        col_offset = j * block
        for i in range(block):
            lag_matrix[:, col_offset + i] = arr[max_tau - i : n_rows - i, j]
        for i in range(block):
            lag_names.append(f"{var_names[j]}_tau_{i}")

    if filter_columns:
        requested: list[int] = []
        for j in range(n_vars):
            offset = j * block
            requested.extend(offset + int(lag) for lag in lag_by_var[j])
            if 0 not in lag_by_var[j]:
                requested.append(offset)
        selected = np.array(sorted(set(requested)), dtype=int)
    else:
        selected = np.arange(lag_matrix.shape[1], dtype=int)

    tau_zero_indices = [idx for idx, name in enumerate(lag_names) if name.endswith("_tau_0")]
    zero_set = set(tau_zero_indices)
    selected_zero = [idx for idx in selected if idx in zero_set]
    selected_non_zero = [idx for idx in selected if idx not in zero_set]
    final_indices = np.array(selected_zero + selected_non_zero, dtype=int)
    reordered_names = [lag_names[idx] for idx in final_indices]
    return lag_matrix[:, final_indices], reordered_names


def lpm_var(percentile: float, degree: float, x: NDArray[np.float64]) -> float:
    """Lower partial-moment VaR matching R's LPM.VaR."""
    values = _finite_values(x)
    pct = min(max(float(percentile), 0.0), 1.0)
    if degree == 0:
        return float(np.quantile(values, pct, method="linear"))
    xmin = float(np.min(values))
    xmax = float(np.max(values))
    if xmin == xmax:
        return xmin

    from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]

    result = minimize_scalar(
        lambda target: abs(float(lpm_ratio(degree, target, values)) - pct),
        bounds=(xmin, xmax),
        method="bounded",
        options={"xatol": _R_OPTIMIZE_TOL},
    )
    return float(result.x)


def upm_var(percentile: float, degree: float, x: NDArray[np.float64]) -> float:
    """Upper partial-moment VaR matching R's UPM.VaR."""
    values = _finite_values(x)
    pct = min(max(float(percentile), 0.0), 1.0)
    if degree == 0:
        return float(np.quantile(values, 1.0 - pct, method="linear"))
    xmin = float(np.min(values))
    xmax = float(np.max(values))
    if xmin == xmax:
        return xmin

    from scipy.optimize import minimize_scalar

    result = minimize_scalar(
        lambda target: abs(float(upm_ratio(degree, target, values)) - pct),
        bounds=(xmin, xmax),
        method="bounded",
        options={"xatol": _R_OPTIMIZE_TOL},
    )
    return float(result.x)


def _finite_values(x: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("x must contain at least one finite value.")
    return finite
