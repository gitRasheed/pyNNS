# Deferred Paths

Every explicit `NotImplementedError` path is listed here so temporary scope
limits stay visible.

| Area | Deferred path | Reason | Dependency / next action |
|---|---|---|---|
| ARMA | `pred_int` | Requires Monte Carlo/bootstrap prediction intervals | Port `NNS.MC` and `NNS.meboot`, then enable |
| Boost | `n_features > 10` stochastic epoch keeper loop | R uses stochastic epoch loop not yet ported | Port stochastic boost epoch loop |
| Boost | `type="class"`, `balance=True` | Classification path not ported | Port factor/classification boost path |
| Boost | `ts_test` | Time-series boost evaluation not ported | Port after ARMA/time-series stack paths |
| Boost | `pred_int` | Requires regression interval/bootstrap logic | Port interval stack |
| Stack | `type="class"`, `balance=True` | Classification path not ported | Port factor/classification stack path |
| Stack | `ts_test` | Time-series stack path not ported | Port after ARMA workflows |
| Stack | `pred_int` | Requires regression interval/bootstrap logic | Port interval stack |
| Regression | `confidence_interval` | Requires interval/bootstrap logic | Port after `NNS.MC` / `NNS.meboot` or direct R interval path |
| Regression | `smooth=True` | Requires R `smooth.spline` equivalent | Decide SciPy `UnivariateSpline` parity strategy |
| Regression | classification `type` paths | Classification path not ported | Port class mode parity |
| Regression | factor/dummy paths | Factor and dummy expansion path not ported | Port `factor_2_dummy` and factor regression paths |
| Regression | `tau="ts"` in dimension reduction | Time-series causation branch in dim-red regression not wired | Reuse `nns_seas` period selection in dim-red path |
| Regression | `multivariate_call=True` with `dim_red_method` | R does not use this combination and PyNNS has no faithful call path for it | Keep rejected unless an R path requiring it is found |
| Regression | univariate `point_only=True` | R uses this through multivariate callers; PyNNS supports point-only in `nns_m_reg` | Keep univariate-only guard unless R parity requires it |
| Multivariate regression | `type="class"` | Classification mode deferred | Port class mode parity |
| Multivariate regression | `confidence_interval` | Multivariate interval path not ported | Port interval/bootstrap logic |
| Multivariate regression | `factor_2_dummy=True` | Factor/dummy expansion not ported | Port `factor_2_dummy` and class distance paths |
