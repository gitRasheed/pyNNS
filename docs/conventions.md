# Conventions

## Degree-Zero Boundary

At degree zero, `LPM` uses `x <= T` and `UPM` uses `x > T`.
Equality is counted by `LPM` only.

For any non-empty finite input, `LPM + UPM = 1` at degree zero.

## Empty Input Divergence From R

R NNS returns `NaN` for empty input.
PyNNS raises `ValueError`.

Rationale: empty arrays in Python are upstream bugs, and NumPy convention is to warn or fail on empty reductions rather than silently produce a meaningful statistic.

## Co-Moment Length Mismatch Divergence From R

R NNS warns when `x` and `y` lengths differ, computes over the shorter length, and divides by the longer length.
PyNNS raises `ValueError`.

Rationale: mismatched co-moment inputs lose observations silently in R. Python callers should fix alignment before computing a bivariate statistic.

## PM Matrix Target Defaults

R `PM.matrix` uses column means when `target` is `NULL` or any non-numeric value.
PyNNS accepts `None` and `"mean"` for this behavior. PyNNS also broadcasts a
scalar numeric target across all variables; R requires callers to pass an
explicit vector such as `rep(0, ncol(variable))`. Target vectors whose length
does not match the number of variables raise `ValueError`.

## Classical Moment Normalization

`mean_pm`, `var_pm`, `skew_pm`, and `kurt_pm` use population normalization by
default, matching NumPy defaults and `NNS.moments(population = TRUE)`. `var_pm`
accepts `ddof` for NumPy-style variance scaling. `skew_pm` and `kurt_pm` do not
apply SciPy's optional finite-sample bias correction.

## Dependence

`nns_dep` follows R's `NNS.dep` bivariate path, including `NNS.gravity` handling
for zero-range inputs and non-positive or non-finite bin widths. PyNNS also caps
the internal gravity bin count at `4 * len(input)` to prevent pathological
allocations on inputs where R's C++ `int` conversion effectively collapses an
absurd bin count. `abs(Correlation) <= Dependence` is not guaranteed by
`NNS.dep`; both R and PyNNS can return signed correlation magnitudes above the
dependence component for near-binary inputs.

## Copula

`nns_copula(x, y)` is the bivariate scalar form of R's `NNS.copula(cbind(x, y))`.
When targets are omitted, PyNNS uses column means, matching R's `target = NULL`.
The `target_x` and `target_y` arguments map to R's two-element target vector.

## Causation

`nns_causation(x, y)` maps to R's `NNS.caus(x, y, tau = 0, p.value = FALSE)`
numeric-vector path. It returns the two directional components and the named
signed net log-ratio key selected by R, either `C(x--->y)` or `C(y--->x)`.
`causal_matrix` maps to R's `NNS.caus.matrix` antisymmetric matrix convention.
`tau='ts'` uses `nns_seas(... )["periods"]` exactly like installed R: the first
period not exceeding `sqrt(length(x))` is selected per variable, including
harmonics when R selects them. Inputs with no eligible selected period follow
R's failure convention and raise. Numeric `tau` lag values remain fully
supported.

## Partition

`nns_part` maps to R's `NNS.part` but returns plain NumPy arrays instead of
`data.table` objects: `"dt"` and `"regression.points"` are dictionaries of
arrays. Installed R 12.0 only distinguishes `type = NULL` from any non-null
`type`: `None` uses XY quadrant splits, while every non-`None` value uses
X-only splits. This differs from documentation that implies separate `"X"`,
`"Y"`, and `"XONLY"` modes. PyNNS matches the installed binary.
`order="max"` is rejected with `TypeError`; installed R coerces it to `NA` and
returns a useless zero-order map. All five `noise_reduction` modes are
supported: `"off"`, `"mean"`, `"median"`, `"mode"`, and `"mode_class"`.

## Regression

`nns_reg` currently maps to R's univariate numeric `NNS.reg` path with
`factor.2.dummy = FALSE`, plotting disabled, and no confidence interval or
smoothing. Return keys match R's list names, but data.table outputs are plain
dictionaries of NumPy arrays. `multivariate_call=True` returns R's internal
two-column regression-point structure as `{"x": ..., "y": ...}` for
`nns_m_reg`. Matrix `x` without dimension reduction dispatches to `nns_m_reg`.
Classification, smooth splines, factor dummy expansion, and confidence intervals
are explicit future batches and raise `NotImplementedError`.

Numeric dimension reduction is supported for `"cor"`, `"NNS.dep"`,
`"NNS.caus"`, `"all"`, `"equal"`, and numeric coefficient vectors. The
synthetic `x.star` projection follows R's min-max normalization and denominator
conventions, including joint normalization for `point_est`. `tau="ts"` remains
deferred in the dim-red regression path. The `"NNS.caus"` branch uses the ported
`Uni.caus` internals and may differ from installed R at small asymmetric
dependence granularity.

`order="max"` follows installed R's univariate convention: fitted values are the
observed `y` values and `regression.points` is the sorted observed `(x, y)` map.
The derivative table still comes from R's pre-reset regression-point construction,
which PyNNS matches rather than recomputing adjacent slopes from all observations.

The `"mode"` and `"mode_class"` noise-reduction modes are accepted in the
univariate path and use the shared `nns_part`/`nns_mode` implementation. The
`"mode_class"` default-order path can produce segment `standard.errors` values
that differ from R at floating grouping granularity: installed R groups the
`gradient` column through data.table's numeric radix grouping, while NumPy keeps
near-identical binary floating values as separate groups. Regression points,
coefficients, fitted values, and point estimates still match R on that path.

## Multivariate Regression

`nns_m_reg` maps to installed R's numeric `NNS.M.reg` path with
`factor.2.dummy = FALSE`, plotting disabled, and no confidence interval.
Outputs use R's keys (`R2`, `rhs.partitions`, `RPM`, `Point.est`, `pred.int`,
and `Fitted.xy`) with data.table objects represented as dictionaries of NumPy
arrays. Classification mode (`type="class"`), factor dummy expansion, and
confidence intervals are deferred and raise `NotImplementedError`.

Point estimates match installed R, including the one-row outsider behavior in
the multi-point path where R drops matrix dimensions before extrapolating.
`order="max"` follows R's convention of using the original regressor matrix as
the regression-point matrix and defaulting `n.best` to 1.

## Stack

`nns_stack` maps to R's numeric `NNS.stack` path using the real `nns_reg`
dimension-reduction and multivariate-regression internals. Classification
(`type="class"`), class balancing, prediction intervals, and `ts_test` are
deferred and raise `NotImplementedError` because they depend on the unported
classification/interval/time-series branches. R's `CV.size = NULL` samples a
random value between 0.2 and 1/3; PyNNS uses a deterministic default of `0.25`.
Pass `cv_size` explicitly for exact R parity.

## Boost

`nns_boost` maps to R's numeric deterministic `NNS.boost` path and uses the
real `nns_reg` and `nns_stack` implementations. The small-feature path
(`n_features <= 10`, where R evaluates all feature combinations) is supported.
The stochastic epoch keeper path for `n_features > 10` is not yet ported and
raises `NotImplementedError`. Classification, balancing, prediction intervals,
and `ts_test` are also deferred and raise `NotImplementedError`. R requires
usable column names for matrix inputs; PyNNS uses positional numeric columns. As
with `nns_stack`, R samples a random CV size when `CV.size = NULL`; PyNNS uses
deterministic `cv_size=0.25` unless specified.

## Seasonality

`nns_seas` maps to installed R's non-plotting `NNS.seas` path and ignores
`plot`, consistent with other PyNNS ports. Inputs shorter than five observations
return R's sentinel period `0`. For mean-zero data, R falls back from coefficient
of variation to `abs(acf1) ** -1`; PyNNS follows the same fallback and
non-finite handling. Installed R can report harmonics rather than the visually
obvious period, so PyNNS matches R's candidate-period screening instead of a
textbook seasonality heuristic. Results are cached by input content and modulo
arguments with defensive copies on return; this preserves R semantics while
avoiding repeated reverse-step scans for identical series.

## ARMA

`nns_arma` maps to R's no-prediction-interval `NNS.ARMA` forecast path and
returns a NumPy forecast vector of length `h`. Forecasts are recursive: each
estimate is appended before the next horizon step. Plot arguments are ignored.
`pred_int` is deferred because it requires `NNS.MC` / `NNS.meboot`.
`seasonal_factor=True` uses only the first detected period from `nns_seas`,
matching `ARMA.seas.weighting(TRUE, ...)`; `seasonal_factor=False` uses the
selected `best_periods` rows. `dynamic=True` with numeric seasonal factors
raises with R's static-seasonality error. Constant-series behavior follows
installed R, including zero forecasts for automatic seasonality paths and `NaN`
forecasts for some explicit numeric-lag paths. Character `weights` with numeric
multi-lag seasonal factors is rejected because installed R errors during numeric
multiplication on that path.

## Normalization

`nns_norm(x, linear=False)` maps to R's numeric matrix `NNS.norm` path with
plotting disabled. PyNNS accepts finite 2D arrays. `linear=True` uses R's
mean-ratio scaling, while `linear=False` additionally weights scaling by
absolute correlation for fewer than 10 columns and NNS dependence for 10 or
more columns.

## Distance

`nns_distance` and `nns_distance_bulk` map to R's regression-point-matrix
helpers. PyNNS accepts `rpm` as a finite 2D numeric array with R's `y.hat`
column in the final position. `nns_distance` applies R's per-target min-max
rescaling before computing weighted nearest-neighbor predictions. `nns_distance_bulk`
matches R's compiled bulk helper, including its raw-feature distance convention.
For `nns_distance` with `k > 1`, PyNNS matches the installed R 12.0 binary:
the exponential rank-weight family uses the R C API's `Rf_dexp` scale argument
as `1 / k`. This differs from the nearby source-code comment that describes it
as a rate.

## Differentiation

`nns_diff` maps to R's scalar callable `NNS.diff` path with plotting and trace
output disabled. It returns a dictionary keyed by R's matrix row names and
rounds results to `digits`, matching R's default output convention.

## ANOVA

`nns_anova` maps to R's non-plotting `NNS.ANOVA` paths. Binary comparisons
return a dictionary keyed like R's list output, aggregate multi-group
comparisons return `{"Certainty": value}`, and `pairwise=True` returns R's
symmetric certainty matrix. Confidence interval bootstrapping is structurally
identical to R but uses NumPy RNG instead of R's `sample()`, so exact per-call
parity is not achievable; numeric values converge to the same population CI.
Pass `random_seed` for reproducible PyNNS results. Degenerate zero-variance
groups preserve R's `NaN` CDF/certainty convention.
