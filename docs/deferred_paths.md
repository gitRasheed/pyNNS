# Deferred Paths

Every explicit `NotImplementedError` path is listed here so temporary scope
limits stay visible.

| Area | Deferred path | Reason | Dependency / next action |
|---|---|---|---|
| ARMA.optim | default optimizer path | Default optimizer evaluates `nns_reg(..., smooth=True)`, so it cannot be faithfully ported until smooth regression is real | Port `smooth=True` first, then `NNS.ARMA.optim` |
| Boost | `n_features > 10` stochastic epoch keeper loop | R uses stochastic epoch loop not yet ported | Port stochastic boost epoch loop |
| Boost | `ts_test` | Time-series boost evaluation not ported | Port after ARMA/time-series stack paths |
| Regression | `confidence_interval` with `smooth=True` | Requires R `stats::smooth.spline(..., spar=...)` fixed-spar compatibility before interval tables can be generated on smoothed fits | Port a minimal R-compatible fixed-spar smoothing spline backend |
| Regression | `smooth=True` | Requires R `stats::smooth.spline(..., spar=...)` fixed-spar compatibility; SciPy smoothers are not parity-compatible | Port a minimal R-compatible fixed-spar smoothing spline backend, then enable `nns_reg(smooth=True)` |
| Stack | factor predictor method-2 diagnostics | `nns_stack` method 1 supports explicit `factor_levels`; method 2 with dummy-expanded factors needs more installed-R parity investigation for diagnostic fields | Investigate R `NNS.stack` method 2 factor-X diagnostics before broadening parity claims |
| Regression | `tau="ts"` in dimension reduction | `nns_seas` is ported, but NNS.seas-derived lags are not wired into dim-red regression | Reuse `nns_seas` period selection in the dim-red path |
| Regression | `multivariate_call=True` with `dim_red_method` | R does not use this combination and PyNNS has no faithful call path for it | Keep rejected unless an R path requiring it is found |
| Regression | bare univariate `point_only=True` | R uses point-only through multivariate/dim-red callers; PyNNS supports those paths but not bare univariate point-only | Keep univariate-only guard unless R parity requires it |
| Multivariate regression | `factor_2_dummy=True` | Factor target encoding is supported with `class_levels`, but factor predictor expansion is not wired | Enable predictor dummy expansion in `nns_m_reg` factor paths |
| Public API | `NNS.SD.cluster(dendrogram=TRUE)` | Default efficient-set clustering is ported, but dendrogram output requires R `hclust`-compatible object construction | Port or shim the hclust object shape if callers need dendrograms |
| Public API | `dy.dx` / `dy.d_` | Derivative helpers depend on finite-difference grids and `nns_reg(..., smooth=True)`, so they are not thin wrappers over existing derivative output | Port after R-compatible `smooth=True` |
| VAR | default VAR path | Depends on `NNS.ARMA.optim` and `nns_stack(ts_test)`; `ts_test` is done, but `NNS.ARMA.optim` remains blocked by smooth regression | Port `smooth=True`, then `NNS.ARMA.optim`, then `NNS.VAR` |
