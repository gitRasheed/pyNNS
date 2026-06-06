# PyNNS API Status

This page summarizes the public PyNNS API surface, known gaps, guarded paths,
and design boundaries.

PyNNS is an alpha, parity-focused Python port of installed R NNS 12.1 beta,
implemented natively in Python on top of NumPy, SciPy, and Polars. It does not
wrap R, call the R package at runtime, or depend on compiled R/C++ shims. The
goal is public input/output compatibility where R behavior is stable,
documented, and useful. The goal is not to copy every R internal helper name,
data-frame quirk, or runtime side effect as a public Python API.

Current release-relevant state: the core partial-moment APIs, deterministic
regression/classification/forecasting surfaces, and scalar multivariate
derivative modes are parity-covered on focused fixtures. The largest remaining
API gaps are guarded vectorized `dy_d` non-mean/mixed modes, direct raw-factor
`nns_m_reg(..., factor_2_dummy=True)`, and named data-frame factor ordering
quirks. Performance gaps remain mostly in large stochastic-dominance workloads
where R uses compiled kernels.

Status labels:

- `implemented`: covered public behavior with no known release-blocking gap.
- `partial`: useful public behavior exists, with documented guarded paths or
  caveats.
- `guarded`: intentionally rejected with an explicit error.
- `known gap`: public structure may exist, but parity is not yet aligned.

Confidence labels are release-maintainer judgments based on current parity,
invariant, and property coverage.

## Public API Status

| API / group | Status | Confidence | Notes |
|---|---|---|---|
| Core partial moments: `lpm`, `upm`, `lpm_ratio`, `upm_ratio` | implemented | high | Matches R partial-moment conventions, including degree-zero equality handling. |
| Partial-moment matrices and n-dimensional wrappers: `pm_matrix`, `co_lpm_nd`, `co_upm_nd`, `dpm_nd` | implemented | high | Public matrix and n-dimensional partial-moment surfaces are covered. |
| Pairwise co-moments: `co_lpm`, `co_upm`, `d_lpm`, `d_upm` | implemented | high | Python raises on length mismatch instead of silently truncating like R. |
| Classical helpers: `ecdf_pm`, `mean_pm`, `var_pm`, `skew_pm`, `kurt_pm`, `nns_moments` | implemented | high | Population-normalized defaults are documented in `docs/conventions.md`. |
| VaR helpers: `lpm_var`, `upm_var` | implemented | high | Used by deterministic confidence and prediction interval paths. |
| Central tendencies: `nns_gravity`, `nns_mode`, `nns_rescale` | implemented | high | Public helper behavior is covered through direct and dependent tests. |
| Dependence and correlation: `nns_dep`, `nns_cor` | implemented | high | Follows installed R bivariate public path; dependence can be below signed correlation magnitude in known R-compatible cases. |
| Copula: `nns_copula` | implemented | high | Bivariate scalar public form is implemented. |
| Causation: `nns_causation`, `causal_matrix` | implemented | medium | Numeric lag paths and `tau="ts"` behavior are covered; some internal asymmetry granularity can differ in regression dimension reduction. |
| Distribution functions: `nns_cdf` | implemented | high | Deterministic non-plotting paths are implemented; plotting is ignored. |
| Distance helpers: `nns_distance`, `nns_distance_bulk` | implemented | high | Numeric and classification conventions follow installed R behavior. |
| Partitioning: `nns_part` | implemented | high | Returns plain dictionaries/arrays instead of R `data.table` objects. |
| Regression: `nns_reg` | implemented | high | Numeric, class-code, confidence interval, smoothing, dimension-reduction, and public factor-expansion paths are covered. |
| Multivariate regression: `nns_m_reg` | partial | medium | Numeric and class paths are implemented; direct raw factor expansion remains guarded. |
| Stack: `nns_stack` | implemented | medium | Numeric/class paths, intervals, factor expansion, and `ts_test` are covered; exact stochastic sample parity is not expected. |
| Boost: `nns_boost` | partial | medium | Deterministic and stochastic structures are implemented; one high-feature threshold path remains guarded to match installed-R failure behavior. |
| Seasonality: `nns_seas` | implemented | high | Non-plotting installed-R path is implemented and cached defensively. |
| ARMA and VAR: `nns_arma`, `nns_arma_optim`, `nns_var` | implemented | medium | Numeric forecasting and supported VAR dimension-reduction paths are implemented. Stochastic interval streams are structural/statistical parity only. |
| Nowcast panel: `nns_nowcast_panel` | implemented | medium | Python-native deterministic monthly panel helper backed by `nns_var`. R NNS 12.1 beta removed `NNS.nowcast`, so this is no longer an R-export parity target. |
| Providers: `CsvNowcastProvider`, `FredApiNowcastProvider` | implemented | medium | Providers produce explicit payloads for `nns_nowcast_panel`. CSV is local/offline. FRED requires optional `fredapi` extra and a supplied API key. |
| Bootstrap/Monte Carlo: `nns_meboot`, `nns_mc` | implemented | medium | Deterministic diagnostics are parity-tested; exact stochastic replicate parity with R is not expected. |
| Stochastic dominance/superiority: `fsd`, `ssd`, `tsd`, `.uni` wrappers, `nns_ss`, `nns_sd_cluster`, `sd_efficient_set` | implemented | medium | Public structures and deterministic paths are covered. SD uses exact pure-NumPy prefix-pair kernels plus a degree-1 discrete order-statistic matrix path; R's C++ core remains faster on full finance fixtures. Stochastic intervals use PyNNS RNG. |
| ANOVA: `nns_anova` | implemented | high | Binary, multi-group, pairwise, and degenerate `NaN` conventions are covered. |
| Normalization: `nns_norm` | implemented | high | Numeric matrix path is implemented. |
| Categorical helpers: `encode_factor_codes`, `factor_2_dummy`, `factor_2_dummy_fr` | implemented | high | Explicit `levels=` should be used to reproduce R factor ordering. |
| Scalar differentiation: `nns_diff`, `dy_dx` | implemented | high | `dy_dx(..., eval_point="overall")` and numeric evaluation points are covered. |
| Multivariate differentiation: `dy_d` | partial | medium | Scalar point modes `mean`, `median`, `last`, `obs`, and `apd` plus vectorized `mean` with `mixed=False` are covered on focused parity fixtures. Vectorized non-mean and vectorized `mixed=True` modes remain guarded. |

## Guarded And Deferred Paths

| Area | Path | Current behavior | Reason / next action |
|---|---|---|---|
| Differentiation | `dy_d` vectorized `wrt` for non-mean modes | Guarded with `NotImplementedError`. | Vectorized `wrt` parity is enforced only for `eval_points="mean"` with `mixed=False`. |
| Differentiation | `dy_d` vectorized `wrt` with `mixed=True` | Guarded with `NotImplementedError`. | Mixed vectorized semantics are not yet aligned. Call per regressor where supported. |
| Multivariate regression | direct `factor_2_dummy=True` raw predictor path | Guarded with `NotImplementedError` in direct `nns_m_reg(..., factor_2_dummy=True)`. | Installed R direct `NNS.M.reg` raw factor input errors. Public `nns_reg` factor expansion is supported. |
| Boost | `threshold` on the `n_features > 10` stochastic path | Guarded with `NotImplementedError` on the high-feature stochastic epoch path. | Installed R errors because `test.features` is never built. PyNNS keeps this explicit. |
| Boost/factor predictors | named data-frame factor predictor ordering | Deferred, not represented as a named-column API. | PyNNS uses positional `X1`, `X2`, ... semantics. Installed R named data frames can reorder columns alphabetically before `data.matrix`. |

## Intentional Design Boundaries

- No hidden network fetching happens by default.
- PyNNS does not export `nns_nowcast`; R NNS 12.1 beta removed `NNS.nowcast`.
- Nowcast providers are payload builders for `nns_nowcast_panel`, not implicit
  public forecast wrappers.
- `CsvNowcastProvider` is local/offline. `FredApiNowcastProvider` requires the
  optional `fredapi` dependency and a FRED API key passed explicitly or present
  in `FRED_API_KEY`.
- Library code does not auto-load `.env` files.
- `fredapi`, Yahoo clients, and pandas are not hard dependencies.
- PyNNS uses explicit Python errors for some cases where R silently truncates,
  coerces, warns, or returns unusable values. Important divergences are recorded
  in `docs/conventions.md`.
- Stochastic exact stream parity is not expected. Stochastic paths use NumPy RNG
  and are tested structurally/statistically.
- Plotting side effects from R APIs are generally ignored; PyNNS returns data.
- Stochastic-dominance performance work stays pure NumPy for alpha. The current
  implementation mirrors R's sorted-column/prefix-sum algorithm and adds
  Python-specific guard pruning, kept-only active-set scans for degree 2/3 and
  degree-1 continuous calls, and an exact order-statistic matrix for large
  degree-1 discrete calls. Optional compiled SD backends remain deferred until
  benchmark evidence justifies the added packaging and maintenance cost.

## Provider Boundary

Nowcast provider support is explicit. Providers return payloads; callers pass
the payload to `nns_nowcast_panel`:

```python
from pynns import nns_nowcast_panel
from pynns.providers import CsvNowcastProvider

provider = CsvNowcastProvider("monthly_panel.csv")
payload = provider.fetch((), "2000-01-03")
result = nns_nowcast_panel(payload["series"], h=2, tau=12, dates=payload["dates"])
```

`FredApiNowcastProvider` follows the same boundary but requires
`pip install "nns-pm[fred]"` and a FRED API key. PyNNS does not ship a default
Yahoo or FRED workflow hidden behind a public nowcast wrapper.

## Intentional Divergences And Caveats

The detailed behavior notes live in `docs/conventions.md`. Release-relevant
examples include:

- Empty numeric inputs raise `ValueError`; R NNS often returns `NaN`.
- Co-moment length mismatches raise `ValueError`; R warns, truncates, and divides
  by the longer length.
- Factor and class labels are explicit. R factor levels become numeric codes;
  PyNNS callers should pass `levels=` or `class_levels=` when ordering matters.
- Public outputs use NumPy arrays and plain dictionaries instead of R
  `data.table` objects.
- Some installed-R quirks are intentionally matched when they affect stable
  public output, such as selected interval and `ts_test` conventions.

## Release-Relevant Caveats

- PyNNS is alpha. The public API is parity-focused but not declared stable.
- This is not full R parity yet.
- The remaining `dy_d` API gaps are vectorized non-mean modes and vectorized
  `mixed=True`; scalar point and distribution modes are parity-covered on
  focused fixtures.
- Optional provider support should remain explicit and dependency-light.
- Version changes and release metadata should be handled separately from API
  status documentation.

## Internal Or Out Of Scope

Some R NNS helper names are implementation details or lower-level surfaces in
the R package rather than APIs PyNNS should expose one-for-one. Examples include
`NNS.ANOVA.bin`, `Uni.caus`, compiled `*_cpp` shims, sampling helpers, and
generated-vector helpers.

PyNNS implements the corresponding behavior natively in Python where it is
needed by public APIs. It does not mirror every R helper name as a top-level
Python export. Matrix-style public behavior is exposed where supported through
Python names such as `causal_matrix`; not exposing an exact R helper name does
not mean the implementation delegates to R or compiled code.
