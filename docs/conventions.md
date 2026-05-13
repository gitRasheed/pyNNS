# Conventions

## Build

PyNNS is currently a pure-Python/NumPy/SciPy port. The earlier native extension
scaffolding was removed after the core port demonstrated pure NumPy/SciPy parity
and competitive performance. Reintroduce native code only as a deliberate future
change backed by benchmarks.

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

`nns_moments` is the public `NNS.moments` wrapper and returns R's dictionary
shape with `mean`, `variance`, `skewness`, and `kurtosis`. `nns_gravity` exposes
R's public `NNS.gravity` central-tendency helper. `fsd_uni`, `ssd_uni`, and
`tsd_uni` are the unidirectional stochastic-dominance wrappers behind R's
`.uni` exports. `co_lpm_nd`, `co_upm_nd`, and `dpm_nd` expose the public
n-dimensional partial-moment wrappers.

`nns_ss` maps to R's `NNS.SS` stochastic-superiority function, not to the
stochastic-dominance tests. It returns `p_gt = P(X > Y)`, `p_tie = P(X = Y)`,
and `p_star = p_gt + 0.5 * p_tie`. `NaN` values are omitted independently from
`x` and `y`, matching R's `na.omit` preprocessing. With
`confidence_interval=True`, intervals are computed through `nns_meboot`,
`lpm_var`, and `upm_var`; exact bootstrap parity with R is not expected because
the RNG streams differ. `random_seed` is a PyNNS-only reproducibility
convenience for that stochastic path.

`nns_sd_cluster` maps to R's `NNS.SD.cluster` default path. It iteratively
peels `sd_efficient_set` results and returns a dictionary of `Cluster_1`,
`Cluster_2`, ... memberships. The output contains variable names, not numeric
cluster labels; when names are omitted, PyNNS uses R-style `X_1`, `X_2`, ...
names. `type="continuous"` is supported for first-degree efficient sets.
`dendrogram=True` returns a plain dictionary mirroring R's `hclust` fields:
`merge`, `height`, `order`, `labels`, `method`, `call`, and `dist.method`.
PyNNS does not plot the dendrogram; it only returns the object data.

`nns_cdf` maps to R's `NNS.CDF` deterministic non-plotting paths. It is a
partial-moment distribution wrapper rather than a textbook ECDF: `degree = 0`
uses R's lower-partial-moment frequency convention, and positive degrees use
`LPM.ratio` deformation. Univariate output columns follow installed R (`x` plus
`CDF`, `S(x)`, `h(x)`, or `H(x)`), while multivariate output keeps the final
column named `CDF` for all types, including survival, hazard, and cumulative
hazard. Plotting is ignored. The univariate `NA`/`Inf` comparison quirks are
handled inside `nns_cdf` without loosening the global partial-moment APIs.

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
`factor.2.dummy = FALSE`, plotting disabled, and no smoothing. Return keys match R's list names, but data.table outputs are plain
dictionaries of NumPy arrays. `multivariate_call=True` returns R's internal
two-column regression-point structure as `{"x": ..., "y": ...}` for
`nns_m_reg`. Matrix `x` without dimension reduction dispatches to `nns_m_reg`.
Classification is supported for numeric/logical/factor-like class-code targets.
Smooth splines remain an explicit future batch and raise `NotImplementedError`.
Factor predictor expansion is supported through the public `nns_reg` path.
When combined with dimension reduction, factor predictors are expanded with
R's full-rank dummy convention before synthetic `x.star` coefficients are
computed.

Numeric dimension reduction is supported for `"cor"`, `"NNS.dep"`,
`"NNS.caus"`, `"all"`, `"equal"`, and numeric coefficient vectors. The
synthetic `x.star` projection follows R's min-max normalization and denominator
conventions, including joint normalization for `point_est`. In this dim-red
regression path, `tau="ts"` follows R's direct `Uni.caus` call and maps to a
fixed lag of `3`; public `nns_causation(..., tau="ts")` still uses the
`NNS.seas`-derived lag path. The `"NNS.caus"` branch uses the ported `Uni.caus`
internals and may differ from installed R at small asymmetric dependence
granularity.

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

Regression confidence intervals are deterministic and use R's `LPM.VaR` /
`UPM.VaR` logic, not `nns_mc` / `nns_meboot`. In the univariate fitted table,
both `conf.int.pos` and `conf.int.neg` use `UPM.VaR(..., degree = 1)` on
segment residuals, matching installed R even though the lower side might look
like an `LPM` candidate. Univariate `point_est` prediction intervals use
`UPM.VaR(..., degree = 0)` for the upper column and `LPM.VaR(..., degree = 0)`
for the lower column. Below-range univariate point estimates follow R's
`findInterval`/data.table behavior: index `0` rows are dropped, so `pred.int`
can have fewer rows than `Point.est`. For class mode, fitted confidence columns
remain raw numeric values, while univariate `pred.int` columns are rounded with
R's `x %% 1 < 0.5` rule. `smooth=True` with `confidence_interval` remains
deferred because it depends on R-compatible smoothing splines.

## Multivariate Regression

`nns_m_reg` maps to installed R's numeric `NNS.M.reg` path with
`factor.2.dummy = FALSE` and plotting disabled.
Outputs use R's keys (`R2`, `rhs.partitions`, `RPM`, `Point.est`, `pred.int`,
and `Fitted.xy`) with data.table objects represented as dictionaries of NumPy
arrays. Numeric and class confidence intervals are deterministic and use the
global residual `UPM.VaR(..., degree = 1)` offset from installed R. In class
mode, fitted predictions and point estimates are rounded/clamped to class codes,
but `pred.int` lower/upper bounds and fitted confidence columns remain raw
numeric values. Classification mode (`type="class"`) is supported for
numeric/logical/factor-like targets and returns numeric class codes. Direct
`nns_m_reg(..., factor_2_dummy=True)` remains rejected for raw factor
predictors because installed R errors on that path. Public `nns_reg` factor
predictor expansion is supported with `factor_2_dummy=True` and explicit
`factor_levels=` metadata; it combines training `x` and `point_est` before
full-rank dummy expansion, matching installed R's `factor_2_dummy_FR` path.

Point estimates match installed R, including the one-row outsider behavior in
the multi-point path where R drops matrix dimensions before extrapolating.
`order="max"` follows R's convention of using the original regressor matrix as
the regression-point matrix and defaulting `n.best` to 1.

## Stack

`nns_stack` maps to R's numeric and deterministic classification `NNS.stack`
paths using the real `nns_reg` dimension-reduction and multivariate-regression
internals. `type="class"` is supported for numeric/logical/factor-like targets
and returns numeric class codes, not labels. Use `class_levels=` to reproduce R
factor level ordering. Raw string labels remain rejected unless explicit levels
are supplied. `balance=True` is supported for classification and follows R's
`downSample` + `upSample` structure: each non-empty class is downsampled to the
minority count without replacement, each class is upsampled to the majority
count with replacement, and the downsampled rows are concatenated before the
upsampled rows. Exact sampled-row parity with R is not expected because PyNNS
uses NumPy's RNG; `random_seed` is a PyNNS-only reproducibility convenience.
Numeric and class prediction intervals are supported and are combined by
installed R's weighted data.table arithmetic. For class stacks, single-method
`method=1` and `method=2` return the delegated interval table unchanged; when
`method=(1,2)`, the weighted final interval table is rounded with R's
`x %% 1 < 0.5` rule.
`ts_test` is supported and follows installed R's split exactly: CV training uses
the tail `ts_test` rows, while CV testing uses the earlier rows
`1:(n - ts_test)`. This is intentionally not changed even though it is
counterintuitive. R's `CV.size = NULL` samples a random value between 0.2 and
1/3; PyNNS uses a deterministic default of `0.25`. Pass `cv_size` explicitly for
exact R parity.
Factor predictor expansion is supported for `nns_stack(method=1)` with explicit
`factor_levels=` metadata. PyNNS expands training and test predictors together
using the same full-rank dummy convention as installed R's aligned train/test
builder. Method-2 factor diagnostics remain deferred pending installed-R parity
investigation.

## Boost

`nns_boost` maps to R's numeric and deterministic classification `NNS.boost`
paths and uses the real `nns_reg` and `nns_stack` implementations. The
small-feature path (`n_features <= 10`, where R evaluates all feature
combinations) is supported. `type="class"` returns numeric class codes, not
labels; use `class_levels=` to reproduce R factor level ordering. Raw string
labels remain rejected unless explicit levels are supplied. `balance=True` is
supported for deterministic small-feature classification and uses the same
R-style `downSample` + `upSample` structure as `nns_stack`; exact sampled-row
parity with R is not expected because PyNNS uses NumPy's RNG, and `random_seed`
is PyNNS-only. The stochastic epoch keeper path for `n_features > 10` is not yet
ported and raises `NotImplementedError`. Factor predictors remain deferred:
installed R integer-codes them via `data.matrix`, but a direct code-port probe
matched final predictions while diverging on deterministic feature-frequency
diagnostics. Numeric `pred_int` is supported and
delegates to `nns_stack(pred_int=...)`, matching installed R; it is deterministic
and does not use MC/meboot. `features_only=True` returns before the final stack
fit and ignores `pred_int`, matching R. Classification `pred_int` is supported
and delegates to final stack `method=1`, so interval bounds remain raw numeric
values. `ts_test` remains deferred and raises `NotImplementedError`. R requires
usable column names for matrix inputs; PyNNS uses positional numeric columns. A
direct deterministic `ts_test` split port was investigated and rejected because
it diverged from installed R on boost keeper diagnostics and one final
prediction. As with `nns_stack`, R
samples a random CV size when `CV.size = NULL`; PyNNS uses deterministic
`cv_size=0.25` unless specified. For classification boost, final predictions,
feature weights, and feature frequencies are parity-tested against installed R
when balance is disabled and structurally tested when balance sampling is
enabled. The public `n.best` value is structural-only because R's final internal
`NNS.stack` call samples its own `CV.size = NULL` split, while PyNNS keeps the
deterministic stack default.

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

`nns_arma` maps to R's installed `NNS.ARMA` forecast path. Without prediction
intervals it returns a NumPy forecast vector of length `h`; with `pred_int` it
returns a dict keyed like R's data.table columns (`Estimates`,
`Lower <percent>% pred.int`, `Upper <percent>% pred.int`). Forecasts are
recursive: each estimate is appended before the next horizon step. Plot
arguments are ignored. Prediction intervals use `nns_mc` / `nns_meboot`; exact
stochastic parity with R is not expected because RNG streams differ.
`random_seed` is a PyNNS-only convenience for reproducible interval tests.
No-`pred_int` deterministic forecasts remain exact parity-tested.
`seasonal_factor=True` uses only the first detected period from `nns_seas`,
matching `ARMA.seas.weighting(TRUE, ...)`; `seasonal_factor=False` uses the
selected `best_periods` rows. `dynamic=True` with numeric seasonal factors
raises with R's static-seasonality error. Constant-series behavior follows
installed R, including zero forecasts for automatic seasonality paths and `NaN`
forecasts for some explicit numeric-lag paths. Character `weights` with numeric
multi-lag seasonal factors is rejected because installed R errors during numeric
multiplication on that path.

## Meboot

`nns_meboot` maps to R's `NNS.meboot` maximum-entropy bootstrap algorithm and
returns plain Python dictionaries instead of R's vectorized list-matrix wrapper.
Scalar `rho` returns one result dictionary; vector `rho` returns a list of result
dictionaries in R's vectorized order. `rho=None` follows installed R's empty
output behavior, and length-one input returns only `{"x": x}`.

Exact replicate parity with R is not expected because PyNNS uses NumPy's random
number generator and SciPy's optimizer while R uses its global RNG and
`optim()`. Deterministic diagnostics (`xx`, `z`, `dv`, `dvtrim`, `xmin`,
`xmax`, `desintxb`, `ordxx`, and `kappa`) are parity-tested exactly. Stochastic
outputs are tested structurally and statistically. `random_seed` is a PyNNS-only
convenience for reproducible bootstrap draws.

## Monte Carlo

`nns_mc` maps to R's `NNS.MC` wrapper around `NNS.meboot`. The rho grid and
exponential rho transformation are parity-tested exactly against installed R.
As with `nns_meboot`, exact stochastic replicate parity is not expected because
R and PyNNS use different RNG streams and optimizer implementations.
`random_seed` is a PyNNS-only convenience passed through to `nns_meboot`.

PyNNS returns `{"ensemble": array, "replicates": dict}`. The `replicates`
mapping preserves R's names, such as `"rho = 1"` and `"rho = -0.5"`, with each
value containing that rho block's replicate matrix. Sampling-vignette examples
are covered as smoke tests, but installed R behavior remains the parity source.

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

Classification distance mode returns numeric class codes, not original labels.
For single-target `nns_distance(..., class_=...)`, installed R uses weighted
mode with integer replication counts `ceil(100 * weight)`. PyNNS follows that
behavior. Installed R's `NNS.distance.bulk(..., class=...)` currently ignores
the class flag in its compiled bulk helper and returns the same inverse-distance
numeric weighted average as non-class bulk distance; PyNNS matches the installed
binary rather than the higher-level classification intent.

## Classification

R classification paths work with numeric class codes. R factors become
1-indexed numeric codes in factor-level order and predictions are returned as
codes rather than decoded labels. PyNNS provides `factor_2_dummy`,
`factor_2_dummy_fr`, and `encode_factor_codes`; pass explicit `levels=` to
reproduce R factor level order because NumPy arrays do not carry factor
metadata.

`nns_reg(..., type="class")`, `nns_m_reg(..., type="class")`, and
`nns_stack(..., type="class")` are supported for numeric, logical, and
factor-like targets. Use `class_levels=` when passing string/object labels so
PyNNS can reproduce R factor codes explicitly. Raw string classification remains
rejected where installed R errors or produces unusable `NA` conversions.
Predictions and point estimates are numeric class codes, not original labels,
matching installed R. Class confidence intervals are supported in `nns_reg` and
`nns_m_reg`; stack/boost class `pred_int` is supported through those regression
interval tables.

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
