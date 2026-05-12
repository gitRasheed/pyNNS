# Deferred Paths

Every explicit `NotImplementedError` path is listed here so temporary scope
limits stay visible.

| Area | Deferred path | Reason | Dependency / next action |
|---|---|---|---|
| ARMA.optim | default optimizer path | Default optimizer evaluates `nns_reg(..., smooth=True)`, so it cannot be faithfully ported until smooth regression is real | Port `smooth=True` first, then `NNS.ARMA.optim` |
| Boost | `n_features > 10` stochastic epoch keeper loop | R uses stochastic epoch loop not yet ported | Port stochastic boost epoch loop |
| Boost | `balance=True` | R uses down/up sampling helpers not yet ported | Port classification balancing helpers |
| Boost | class `pred_int` | Classification interval wiring is not ported | Port class interval parity if an R caller requires it |
| Boost | `ts_test` | Time-series boost evaluation not ported | Port after ARMA/time-series stack paths |
| Boost | `pred_int` | Boost interval wiring is not ported | Reuse numeric regression/stack intervals in the boost path |
| Stack | class `pred_int` | Classification interval tables are not ported | Port class interval parity if an R caller requires it |
| Regression | `confidence_interval` with `smooth=True` | Requires R `stats::smooth.spline(..., spar=...)` fixed-spar compatibility before interval tables can be generated on smoothed fits | Port a minimal R-compatible fixed-spar smoothing spline backend |
| Regression | `smooth=True` | Requires R `stats::smooth.spline(..., spar=...)` fixed-spar compatibility; SciPy smoothers are not parity-compatible | Port a minimal R-compatible fixed-spar smoothing spline backend, then enable `nns_reg(smooth=True)` |
| Regression | classification confidence intervals | Numeric class prediction is ported, but class interval parity is not covered yet | Port class interval parity if an R caller requires it |
| Regression | factor/dummy predictor paths | Factor target encoding is supported with `class_levels`, but factor predictor expansion is not wired | Enable predictor dummy expansion in `nns_reg` factor paths |
| Regression | `tau="ts"` in dimension reduction | Time-series causation branch in dim-red regression not wired | Reuse `nns_seas` period selection in dim-red path |
| Regression | `multivariate_call=True` with `dim_red_method` | R does not use this combination and PyNNS has no faithful call path for it | Keep rejected unless an R path requiring it is found |
| Regression | univariate `point_only=True` | R uses this through multivariate callers; PyNNS supports point-only in `nns_m_reg` | Keep univariate-only guard unless R parity requires it |
| Multivariate regression | `factor_2_dummy=True` | Factor target encoding is supported with `class_levels`, but factor predictor expansion is not wired | Enable predictor dummy expansion in `nns_m_reg` factor paths |
| VAR | default VAR path | Depends on `NNS.ARMA.optim` and `nns_stack(ts_test)`; `ts_test` is done, but `NNS.ARMA.optim` remains blocked by smooth regression | Port `smooth=True`, then `NNS.ARMA.optim`, then `NNS.VAR` |
