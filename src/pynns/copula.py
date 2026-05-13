from __future__ import annotations

import math
from typing import cast

import numpy as np
from numpy.typing import NDArray

from pynns.co_moments import _as_pair
from pynns.dependence import _dpm_nd
from pynns.pm_matrix import pm_matrix


def nns_copula(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    target_x: float | None = None,
    target_y: float | None = None,
) -> float:
    """Return R's bivariate NNS.copula dependence value."""
    x_values, y_values = _as_pair(x, y)
    values = np.column_stack((x_values, y_values))
    target = _target(values, target_x, target_y)

    discrete_pm_cov = pm_matrix(0.0, 0.0, target, values, pop_adj=False)
    discrete_co_pm = float(discrete_pm_cov["cupm"][0, 1] + discrete_pm_cov["clpm"][0, 1])
    if discrete_co_pm == 1.0 or discrete_co_pm == 0.0:
        return 1.0

    continuous_pm_cov = pm_matrix(1.0, 1.0, target, values, pop_adj=True, norm=True)
    continuous_co_pm = float(continuous_pm_cov["cupm"][0, 1] + continuous_pm_cov["clpm"][0, 1])

    discrete_d_pm = _dpm_nd(values, target, 0.0, norm=True)
    continuous_d_pm = _dpm_nd(values, target, 1.0, norm=True)

    discrete_dep = min(max(abs(discrete_co_pm - 0.5) / 0.5, 0.0), 1.0)
    continuous_dep = min(max(abs(continuous_co_pm - 0.5) / 0.5, 0.0), 1.0)
    n_dim_discrete_dep = abs(discrete_d_pm - 0.75) / 0.75
    n_dim_continuous_dep = abs(continuous_d_pm - 0.75) / 0.75

    return math.sqrt(
        (discrete_dep + continuous_dep + n_dim_discrete_dep + n_dim_continuous_dep) / 4.0
    )


def _target(
    values: NDArray[np.float64],
    target_x: float | None,
    target_y: float | None,
) -> NDArray[np.float64]:
    targets = cast(NDArray[np.float64], np.mean(values, axis=0))
    if target_x is not None:
        targets[0] = float(target_x)
    if target_y is not None:
        targets[1] = float(target_y)
    return targets
