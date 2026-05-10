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

## Normalization

`nns_norm(x, linear=False)` maps to R's numeric matrix `NNS.norm` path with
plotting disabled. PyNNS accepts finite 2D arrays. `linear=True` uses R's
mean-ratio scaling, while `linear=False` additionally weights scaling by
absolute correlation for fewer than 10 columns and NNS dependence for 10 or
more columns.
