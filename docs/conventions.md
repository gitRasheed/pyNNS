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
