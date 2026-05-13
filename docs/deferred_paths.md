# Deferred Paths

Every explicit `NotImplementedError` path is listed here so temporary scope
limits stay visible.

| Area | Deferred path | Reason | Dependency / next action |
|---|---|---|---|
| ARMA.optim | default optimizer path | Default optimizer evaluates `nns_reg(..., smooth=True)`, so it cannot be faithfully ported until smooth regression is real | Port `smooth=True` first, then `NNS.ARMA.optim` |
| Boost | `n_features > 10` stochastic epoch keeper loop | R uses stochastic epoch loop not yet ported | Port stochastic boost epoch loop |
| Boost | `ts_test` | Installed-R probe of the deterministic small-feature branch diverged from a direct tail-train/prefix-test split port on keeper frequencies and one final prediction | Map R `NNS.boost` `ts.test` learner scoring/final stack interaction before porting |
| Boost | factor predictor paths | R `data.matrix` integer-codes factor predictors before deterministic feature selection, but a direct integer-code port matched predictions while diverging on deterministic feature-frequency diagnostics in an installed-R probe | Map R feature-set scoring/keeper diagnostics for factor-coded predictors before porting |
| Regression | `confidence_interval` with `smooth=True` | Requires R `stats::smooth.spline(..., spar=...)` fixed-spar compatibility before interval tables can be generated on smoothed fits | Port a minimal R-compatible fixed-spar smoothing spline backend |
| Regression | `smooth=True` | Requires R `stats::smooth.spline(..., spar=...)` fixed-spar compatibility; SciPy smoothers are not parity-compatible | Port a minimal R-compatible fixed-spar smoothing spline backend, then enable `nns_reg(smooth=True)` |
| Stack | factor predictor method-2 diagnostics | `nns_stack` method 1 supports explicit `factor_levels`; method 2 with dummy-expanded factors needs more installed-R parity investigation for diagnostic fields | Investigate R `NNS.stack` method 2 factor-X diagnostics before broadening parity claims |
| Regression | `multivariate_call=True` with `dim_red_method` | R does not use this combination and PyNNS has no faithful call path for it | Keep rejected unless an R path requiring it is found |
| Regression | bare univariate `point_only=True` | R uses point-only through multivariate/dim-red callers; PyNNS supports those paths but not bare univariate point-only | Keep univariate-only guard unless R parity requires it |
| Regression | matrix `point_est` with bare univariate `x` | Matrix `point_est` is supported through matrix-X dispatch and dim-red paths; bare univariate `NNS.reg` expects scalar/vector point estimates | Keep rejected unless installed R exposes a distinct bare-univariate matrix-`point.est` convention |
| Multivariate regression | direct `factor_2_dummy=True` raw predictor path | Public `nns_reg` factor predictors are supported, but installed R direct `NNS.M.reg` raw factor input errors; PyNNS keeps direct `nns_m_reg` raw factor input rejected unless another public R path requires it | Keep rejected; revisit only if a public installed-R path depends on direct `NNS.M.reg` factor expansion |
| Public API | `dy.dx` / `dy.d_` | Derivative helpers depend on finite-difference grids and `nns_reg(..., smooth=True)`, so they are not thin wrappers over existing derivative output | Port after R-compatible `smooth=True` |
| VAR | default VAR path | Depends on `NNS.ARMA.optim` and `nns_stack(ts_test)`; `ts_test` is done, but `NNS.ARMA.optim` remains blocked by smooth regression | Port `smooth=True`, then `NNS.ARMA.optim`, then `NNS.VAR` |
