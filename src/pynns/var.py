from __future__ import annotations

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
        "nns_var default VAR path depends on nns_arma_optim, which requires nns_reg "
        "smooth=True and is not yet ported."
    )


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
