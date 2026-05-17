# Benchmarks

Run with:

```bash
mkdir -p docs/benchmark_reports
uv run pytest -n0 -m benchmark --benchmark-enable \
  --benchmark-json=docs/benchmark_reports/benchmark_latest.json tests/benchmarks/
Rscript scripts/benchmark_realistic_sd_r.R \
  --repeats=3 --max-repeats=1 \
  --output=docs/benchmark_reports/realistic_sd_r_latest.csv
uv run python scripts/update_benchmarks_doc.py docs/benchmark_reports/benchmark_latest.json \
  --realistic-sd-r-csv=docs/benchmark_reports/realistic_sd_r_latest.csv
```

## Results

`Python speed vs R` is computed as `R baseline / Python mean`. Values above `1.00x` mean Python is faster; values below `1.00x` mean Python is slower.

| Benchmark | Python mean | R baseline | Python speed vs R |
| --- | ---: | ---: | ---: |
| `lpm small` | 0.014 ms | 0.075 ms | 5.27x |
| `pm matrix scale, 10` | 0.092 ms | 0.400 ms | 4.33x |
| `pm matrix scale, 50` | 0.372 ms | 3.800 ms | 10.21x |
| `pm matrix scale, 100` | 2.516 ms | 16.800 ms | 6.68x |
| `sd efficient set degree 2 scale` | 13.615 ms | 4.600 ms | 0.34x |
| `nns sd cluster 252x50 degree2` | 28.975 ms | 13.800 ms | 0.48x |
| `nns sd cluster 252x50 degree2 dendrogram` | 27.492 ms | 26.333 ms | 0.96x |
| `nns cdf 1000 degree0` | 0.034 ms | 1.150 ms | 34.04x |
| `nns cdf 1000 degree2` | 0.120 ms | 1.400 ms | 11.65x |
| `nns cdf 500x3 degree1` | 52.273 ms | 54.400 ms | 1.04x |
| `nns dep 1000` | 8.620 ms | 11.300 ms | 1.31x |
| `nns dep asym 1000` | 8.433 ms | 9.800 ms | 1.16x |
| `nns copula 1000` | 0.435 ms | 4.800 ms | 11.04x |
| `nns causation 1000` | 18.578 ms | 97.600 ms | 5.25x |
| `nns norm 1000x3` | 0.141 ms | 1.500 ms | 10.68x |
| `nns distance 1000x3` | 0.772 ms | 1.160 ms | 1.50x |
| `nns distance bulk 1000x3 100` | 7.042 ms | 9.200 ms | 1.31x |
| `nns distance class 500x3` | 0.677 ms | 0.900 ms | 1.33x |
| `nns distance bulk class 500x3 50` | 1.523 ms | 2.260 ms | 1.48x |
| `nns diff sin` | 1.469 ms | 4.050 ms | 2.76x |
| `dy dx numeric eval points` | 25.919 ms | 53.333 ms | 2.06x |
| `dy_d`, scalar wrt=1, eval_points=mean, N=2, T_obs=100 | 99.794 ms | 266.200 ms | 2.67x |
| `dy_d`, scalar wrt=1, eval_points=median, N=2, T_obs=100 | 105.217 ms | 245.600 ms | 2.33x |
| `dy_d`, scalar wrt=1, eval_points=last, N=2, T_obs=100 | 106.697 ms | 279.200 ms | 2.62x |
| `dy_d`, scalar wrt=1, eval_points=obs, N=2, T_obs=100 | 104.391 ms | 290.000 ms | 2.78x |
| `dy_d`, scalar wrt=1, eval_points=apd, N=2, T_obs=100 | 588.859 ms | 790.200 ms | 1.34x |
| `nns anova 100x2` | 8.523 ms | 4.400 ms | 0.52x |
| `nns part 500` | 0.622 ms | 3.750 ms | 6.03x |
| `nns reg 500` | 63.704 ms | 37.200 ms | 0.58x |
| `nns reg 200 confidence interval` | 60.473 ms | 93.400 ms | 1.54x |
| `nns reg 200 smooth` | 13.922 ms | 37.000 ms | 2.66x |
| `nns reg factor predictor 200` | 23.312 ms | 119.400 ms | 5.12x |
| `nns reg factor predictor dimred 120` | 27.955 ms | 35.800 ms | 1.28x |
| `nns reg class 200` | 12.976 ms | 33.200 ms | 2.56x |
| `nns reg class 200 confidence interval` | 25.447 ms | 57.000 ms | 2.24x |
| `nns reg dimred 200x3` | 37.422 ms | 42.600 ms | 1.14x |
| `nns m reg 200x3` | 94.174 ms | 97.600 ms | 1.04x |
| `nns m reg 200x3 confidence interval` | 97.257 ms | 130.400 ms | 1.34x |
| `nns m reg class 200x3` | 52.913 ms | 126.600 ms | 2.39x |
| `nns m reg class 200x3 confidence interval` | 54.459 ms | 150.400 ms | 2.76x |
| `nns stack 100x3` | 226.376 ms | 369.667 ms | 1.63x |
| `nns stack factor predictor 60 method1` | 33.446 ms | 214.333 ms | 6.41x |
| `nns stack mixed factor predictor 60 method2` | 39.833 ms | 168.400 ms | 4.23x |
| `nns stack mixed factor predictor 100x3 method12` | 317.172 ms | 341.333 ms | 1.08x |
| `nns stack 100x3 pred int` | 154.342 ms | 286.000 ms | 1.85x |
| `nns stack 100x3 ts test` | 176.405 ms | 300.333 ms | 1.70x |
| `nns stack class 100x3` | 117.654 ms | 270.667 ms | 2.30x |
| `nns stack class 100x3 pred int` | 125.313 ms | 251.667 ms | 2.01x |
| `nns stack class balance 150x3` | 182.711 ms | 246.000 ms | 1.35x |
| `nns boost 50x3` | 192.262 ms | 2919.500 ms | 15.19x |
| `nns boost 50x3 pred int` | 144.958 ms | 3676.000 ms | 25.36x |
| `nns boost 50x3 ts test` | 154.818 ms | 4128.000 ms | 26.66x |
| `nns boost stochastic 64x11` | 253.723 ms | 3311.667 ms | 13.05x |
| `nns boost stochastic ts test 64x11` | 202.057 ms | 5851.000 ms | 28.96x |
| `nns boost factor predictor 50x2` | 129.227 ms | 4747.000 ms | 36.73x |
| `nns boost multi factor predictor 50x3` | 187.540 ms | 5186.400 ms | 27.65x |
| `nns boost class 50x3` | 170.659 ms | 5790.000 ms | 33.93x |
| `nns boost class 50x3 pred int` | 174.318 ms | 3827.000 ms | 21.95x |
| `nns boost class balance 80x3` | 357.962 ms | 3120.000 ms | 8.72x |
| `nns mode continuous 1000` | 0.478 ms | 0.100 ms | 0.21x |
| `nns seas 1000` | 0.012 ms | 1.100 ms | 91.87x |
| `nns seas 5000` | 0.033 ms | 4.000 ms | 120.24x |
| `nns arma 500 auto nonlin` | 16.784 ms | 313.333 ms | 18.67x |
| `nns arma 500 explicit12 nonlin` | 73.024 ms | 318.333 ms | 4.36x |
| `nns arma 200 explicit4 lin predint` | 156.917 ms | 207.800 ms | 1.32x |
| `nns arma 200 auto nonlin predint` | 172.737 ms | 380.000 ms | 2.20x |
| `nns arma optim 80 small` | 22.617 ms | 163.333 ms | 7.22x |
| `nns_var`, dim_red_method=cor, N=3, T_obs=80, h=3, tau=2 | 719.240 ms | 3603.667 ms | 5.01x |
| `nns_var`, dim_red_method=NNS.dep, N=3, T_obs=80, h=3, tau=2 | 1324.319 ms | 3995.333 ms | 3.02x |
| `nns_var`, dim_red_method=NNS.caus, N=3, T_obs=80, h=3, tau=2 | 2973.048 ms | 7236.000 ms | 2.43x |
| `nns_var`, dim_red_method=all, N=3, T_obs=80, h=3, tau=2 | 3419.715 ms | 7409.667 ms | 2.17x |
| `nns meboot 500 reps100` | 74.212 ms | 78.000 ms | 1.05x |
| `nns meboot 1000 reps100` | 113.010 ms | 108.000 ms | 0.96x |
| `nns mc 500 reps30 by02` | 336.957 ms | 736.000 ms | 2.18x |
| `nns mc 500 reps30 by01` | 648.405 ms | 1437.000 ms | 2.22x |
| `nns ss 1000` | 0.055 ms | 0.200 ms | 3.63x |
| `nns ss 200 ci reps100` | 172.269 ms | 152.667 ms | 0.89x |

## Realistic Finance SD North Stars

These benchmarks use the static daily-return fixture at
`tests/fixtures/finance/sp500_daily_returns_2019_2023.csv`. The fixture contains
1257 daily return rows and 480 clean return columns after dropping
tickers with missing or non-finite returns. Constituent-universe benchmarks exclude
`SPY` and `GSPC`, leaving 478 columns. Market-relative workflows
prefer `GSPC` and fall back to `SPY`; tradable-proxy examples use `SPY`.

Benchmark-column sanity metadata:

- SPY/GSPC correlation: 0.998873
- Mean absolute daily return difference: 0.000372
- Max absolute daily return difference: 0.010417

Python timings come from `pytest-benchmark`. R timings come from
`scripts/benchmark_realistic_sd_r.R` when `--realistic-sd-r-csv` is supplied to
the updater. Rows marked `manual placeholder` use the last manually recorded R
baseline so Python/R comparisons remain visible when R has not been rerun.

Run only the realistic Python benchmarks with:

```bash
PYNNS_OFFLINE=1 uv run pytest -q -n0 -m benchmark --benchmark-enable \
  --benchmark-json=docs/benchmark_reports/realistic_sd_python_latest.json \
  tests/benchmarks/test_stochastic_dominance_realistic.py \
  tests/benchmarks/test_finance_sd_rolling.py \
  tests/benchmarks/test_finance_partial_moment_workflows.py
```

Run matching R baselines with:

```bash
Rscript scripts/benchmark_realistic_sd_r.R \
  --repeats=3 --max-repeats=1 \
  --output=docs/benchmark_reports/realistic_sd_r_latest.csv
```

`Python/R slowdown` is computed as `Python mean / R mean`. Values above `1.00x`
mean Python is slower than R.

| Realistic benchmark | Python mean | R mean | R source | Python/R slowdown |
| --- | ---: | ---: | --- | ---: |
| `nns_sd_cluster`, degree=1, N=50, T_obs=252 | 30.992 ms | 2.333 ms | measured | 13.28x |
| `sd_efficient_set`, degree=1, N=50, T_obs=252 | 29.286 ms | 2.667 ms | measured | 10.98x |
| `nns_sd_cluster`, degree=2, N=50, T_obs=252 | 18.254 ms | 7.000 ms | measured | 2.61x |
| `sd_efficient_set`, degree=2, N=50, T_obs=252 | 15.348 ms | 2.333 ms | measured | 6.58x |
| `nns_sd_cluster`, degree=1, N=100, T_obs=252 | 4.523 ms | 6.333 ms | measured | 0.71x |
| `sd_efficient_set`, degree=1, N=100, T_obs=252 | 4.248 ms | 5.667 ms | measured | 0.75x |
| `nns_sd_cluster`, degree=2, N=100, T_obs=252 | 19.035 ms | 15.000 ms | measured | 1.27x |
| `sd_efficient_set`, degree=2, N=100, T_obs=252 | 8.459 ms | 4.333 ms | measured | 1.95x |
| `nns_sd_cluster`, degree=2, N=250, T_obs=252 | 74.904 ms | 58.000 ms | measured | 1.29x |
| `sd_efficient_set`, degree=2, N=250, T_obs=252 | 26.818 ms | 14.333 ms | measured | 1.87x |
| `nns_sd_cluster`, degree=2, N=478, T_obs=252 | 288.492 ms | 180.000 ms | measured | 1.60x |
| `sd_efficient_set`, degree=2, N=478, T_obs=252 | 103.832 ms | 40.333 ms | measured | 2.57x |
| `sd_efficient_set`, degree=2, N=100, T_obs=1257 | 29.586 ms | 21.000 ms | measured | 1.41x |
| `nns_sd_cluster`, degree=2, N=250, T_obs=1257 | 243.840 ms | 202.333 ms | measured | 1.21x |
| `sd_efficient_set`, degree=2, N=250, T_obs=1257 | 114.313 ms | 66.333 ms | measured | 1.72x |
| `nns_sd_cluster`, degree=2, N=478, T_obs=1257 | 970.412 ms | 645.000 ms | measured | 1.50x |
| `sd_efficient_set`, degree=2, N=478, T_obs=1257 | 499.727 ms | 188.000 ms | measured | 2.66x |

Additional realistic finance workflow benchmarks:

| Benchmark | Python mean | R mean | R source | Python/R slowdown | Summary metadata |
| --- | ---: | ---: | --- | ---: | --- |
| Lower/upper constituent dispersion ratio, N=100, T_obs=252 | 0.127 ms | n/a | n/a | n/a | n/a |
| Magnificent Seven downside stress components with SPY | 0.361 ms | n/a | n/a | n/a | n/a |
| Magnificent Seven market-downside stress components | 11.208 ms | 44.000 ms | measured | 0.25x | downside obs: 172; stress R2: 0.7852; SPY/GSPC corr: 0.9989; mean abs diff: 0.0003716; max abs diff: 0.01042 |
| Market-relative daily dispersion, full fixture | 15.542 ms | 33.000 ms | measured | 0.47x | signal len: 1257; finite: 1257; next-day corr: 0.06635; SPY/GSPC corr: 0.9989; mean abs diff: 0.0003716; max abs diff: 0.01042 |
| Market-relative rolling dispersion signal, 252d | 16.556 ms | 27.333 ms | measured | 0.61x | signal len: 1006; finite: 1006; next-day corr: 0.03746; SPY/GSPC corr: 0.9989; mean abs diff: 0.0003716; max abs diff: 0.01042 |
| Market-relative rolling dispersion signal, 63d | 16.451 ms | 27.667 ms | measured | 0.59x | signal len: 1195; finite: 1195; next-day corr: 0.02139; SPY/GSPC corr: 0.9989; mean abs diff: 0.0003716; max abs diff: 0.01042 |
| Partial-moment covariance workflow, 1257d-degree1-mean | 34.865 ms | 1.442 s | measured | 0.02x | rows: 1257; cols: 478; matrix N: 478 |
| Partial-moment covariance workflow, 252d-degree1-mean | 26.757 ms | 287.000 ms | measured | 0.09x | rows: 252; cols: 478; matrix N: 478 |
| Partial-moment covariance workflow, 252d-degree2-zero | 20.959 ms | 296.333 ms | measured | 0.07x | rows: 252; cols: 478; matrix N: 478 |
| Rolling SD cluster, 252-day monthly, degree=2, n100 | 939.343 ms | 799.667 ms | measured | 1.17x | windows: 48; avg set: 14.29; avg clusters: 8.375 |
| Rolling SD cluster, 252-day monthly, degree=2, nmax | 11.558 s | 9.384 s | measured | 1.23x | windows: 48; avg set: 29.48; avg clusters: 13.65 |
| Rolling SD cluster, 252-day quarterly, degree=1 | 2.103 s | 1.161 s | measured | 1.81x | windows: 16; avg set: 468.5; avg clusters: 1.812 |
| Rolling SD cluster, 756-day quarterly, degree=2 | 6.346 s | 4.200 s | measured | 1.51x | windows: 9; avg set: 33.11; avg clusters: 11.89 |
| Rolling SD efficient set, 252-day monthly, degree=2, n100 | 355.903 ms | 280.000 ms | measured | 1.27x | windows: 48; avg set: 14.29; avg turnover: 0.4598 |
| Rolling SD efficient set, 252-day monthly, degree=2, nmax | 3.838 s | 2.078 s | measured | 1.85x | windows: 48; avg set: 29.48; avg turnover: 0.5228 |
| Rolling SD efficient set, 252-day quarterly, degree 1 vs 2 | 3.687 s | 1.847 s | measured | 2.00x | windows: 16; avg d1 set: 468.5; avg d2 set: 29.56 |
| Rolling SD efficient set, 252-day quarterly, degree=1 | 2.272 s | 1.149 s | measured | 1.98x | windows: 16; avg set: 468.5; avg turnover: 0.03102 |

Interpretation:

- Large degree-1 discrete SD uses an exact order-statistic dominance
  matrix: one empirical sample FSD-dominates another iff every sorted
  order statistic is at least as large, with at least one strict
  improvement.
- Guarded prefix-pair evaluation skips curve work for min/mean/identical
  impossible pairs, and the standalone efficient-set path only checks
  already-kept candidates for degree 2/3 and degree-1 continuous cases.
- The implementation deliberately follows R's C++ SD algorithmic structure:
  sorted columns, prefix sums, pair-threshold dominance checks, exact guards, and
  no tolerance-based shortcuts.
- Full-fixture PyNNS runs are feasible for research iteration, but R's C++ SD
  core remains materially faster on the largest cluster cases.
