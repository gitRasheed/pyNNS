# Deferred Paths

Every explicit `NotImplementedError` path is listed here so temporary scope
limits stay visible.

| Area | Deferred path | Reason | Dependency / next action |
|---|---|---|---|
| ARMA.optim | default optimizer path | Default optimizer evaluates `nns_reg(..., smooth=True)`, so it cannot be faithfully ported until smooth regression is real | Port `smooth=True` first, then `NNS.ARMA.optim` |
| Boost | `n_features > 10` stochastic epoch keeper loop | R uses stochastic epoch loop not yet ported | Port stochastic boost epoch loop |
| Boost | class `pred_int` | Classification interval wiring is not ported | Port class interval parity if an R caller requires it |
| Boost | `ts_test` | Time-series boost evaluation not ported | Port after ARMA/time-series stack paths |
| Boost | `pred_int` | Numeric regression and stack intervals are ported, but boost-specific interval wiring is not | Wire boost outputs through the existing numeric interval paths |
| Stack | class `pred_int` | Classification interval tables are not ported | Port class interval parity if an R caller requires it |
| Regression | `confidence_interval` with `smooth=True` | Requires R `stats::smooth.spline(..., spar=...)` fixed-spar compatibility before interval tables can be generated on smoothed fits | Port a minimal R-compatible fixed-spar smoothing spline backend |
| Regression | `smooth=True` | Requires R `stats::smooth.spline(..., spar=...)` fixed-spar compatibility; SciPy smoothers are not parity-compatible | Port a minimal R-compatible fixed-spar smoothing spline backend, then enable `nns_reg(smooth=True)` |
| Regression | classification confidence intervals | Numeric class prediction is ported, but class interval parity is not covered yet | Port class interval parity if an R caller requires it |
| Regression | factor/dummy predictor paths | Factor target encoding is supported with `class_levels`, but factor predictor expansion is not wired | Enable predictor dummy expansion in `nns_reg` factor paths |
| Regression | `tau="ts"` in dimension reduction | `nns_seas` is ported, but NNS.seas-derived lags are not wired into dim-red regression | Reuse `nns_seas` period selection in the dim-red path |
| Regression | `multivariate_call=True` with `dim_red_method` | R does not use this combination and PyNNS has no faithful call path for it | Keep rejected unless an R path requiring it is found |
| Regression | bare univariate `point_only=True` | R uses point-only through multivariate/dim-red callers; PyNNS supports those paths but not bare univariate point-only | Keep univariate-only guard unless R parity requires it |
| Multivariate regression | `factor_2_dummy=True` | Factor target encoding is supported with `class_levels`, but factor predictor expansion is not wired | Enable predictor dummy expansion in `nns_m_reg` factor paths |
| Public API | `NNS.CDF` | Public CDF wrapper includes univariate/multivariate CDF, survival, hazard, cumulative-hazard, and regression-backed target evaluation branches; it is not a thin wrapper | Port as a dedicated public-function batch |
| Public API | `dy.dx` / `dy.d_` | Derivative helpers depend on finite-difference grids and `nns_reg(..., smooth=True)`, so they are not thin wrappers over existing derivative output | Port after R-compatible `smooth=True` |
| VAR | default VAR path | Depends on `NNS.ARMA.optim` and `nns_stack(ts_test)`; `ts_test` is done, but `NNS.ARMA.optim` remains blocked by smooth regression | Port `smooth=True`, then `NNS.ARMA.optim`, then `NNS.VAR` |
