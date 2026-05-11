from __future__ import annotations

from typing import Any

from pynns.pm_matrix import pm_matrix as pm_matrix

_EXPORTS = {
    "causal_matrix": ("pynns.causation", "causal_matrix"),
    "co_lpm": ("pynns.co_moments", "co_lpm"),
    "co_upm": ("pynns.co_moments", "co_upm"),
    "d_lpm": ("pynns.co_moments", "d_lpm"),
    "d_upm": ("pynns.co_moments", "d_upm"),
    "ecdf_pm": ("pynns.classical", "ecdf_pm"),
    "fsd": ("pynns.stochastic_dominance", "fsd"),
    "kurt_pm": ("pynns.classical", "kurt_pm"),
    "lpm": ("pynns.core", "lpm"),
    "lpm_ratio": ("pynns.core", "lpm_ratio"),
    "lpm_var": ("pynns.var", "lpm_var"),
    "mean_pm": ("pynns.classical", "mean_pm"),
    "nns_anova": ("pynns.anova", "nns_anova"),
    "nns_boost": ("pynns.boost", "nns_boost"),
    "nns_causation": ("pynns.causation", "nns_causation"),
    "nns_copula": ("pynns.copula", "nns_copula"),
    "nns_cor": ("pynns.dependence", "nns_cor"),
    "nns_dep": ("pynns.dependence", "nns_dep"),
    "nns_diff": ("pynns.diff", "nns_diff"),
    "nns_distance": ("pynns.distance", "nns_distance"),
    "nns_distance_bulk": ("pynns.distance", "nns_distance_bulk"),
    "nns_mode": ("pynns.central_tendencies", "nns_mode"),
    "nns_m_reg": ("pynns.multivariate_regression", "nns_m_reg"),
    "nns_norm": ("pynns.norm", "nns_norm"),
    "nns_part": ("pynns.part", "nns_part"),
    "nns_reg": ("pynns.regression", "nns_reg"),
    "nns_rescale": ("pynns.central_tendencies", "nns_rescale"),
    "nns_seas": ("pynns.seasonality", "nns_seas"),
    "nns_stack": ("pynns.stack", "nns_stack"),
    "sd_efficient_set": ("pynns.stochastic_dominance", "sd_efficient_set"),
    "skew_pm": ("pynns.classical", "skew_pm"),
    "ssd": ("pynns.stochastic_dominance", "ssd"),
    "tsd": ("pynns.stochastic_dominance", "tsd"),
    "upm": ("pynns.core", "upm"),
    "upm_ratio": ("pynns.core", "upm_ratio"),
    "upm_var": ("pynns.var", "upm_var"),
    "var_pm": ("pynns.classical", "var_pm"),
}

__all__ = sorted((*_EXPORTS, "pm_matrix"))


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'pynns' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
