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
| `lpm small` | 0.012 ms | 0.075 ms | 6.42x |
| `pm matrix scale, 10` | 0.083 ms | 0.400 ms | 4.80x |
| `pm matrix scale, 50` | 0.332 ms | 3.800 ms | 11.45x |
| `pm matrix scale, 100` | 2.132 ms | 16.800 ms | 7.88x |
| `sd efficient set degree 2 scale` | 13.029 ms | 4.600 ms | 0.35x |
| `nns sd cluster 252x50 degree2` | 27.180 ms | 13.800 ms | 0.51x |
| `nns sd cluster 252x50 degree2 dendrogram` | 26.579 ms | 26.333 ms | 0.99x |
| `nns cdf 1000 degree0` | 0.029 ms | 1.150 ms | 40.07x |
| `nns cdf 1000 degree2` | 0.102 ms | 1.400 ms | 13.75x |
| `nns cdf 500x3 degree1` | 62.974 ms | 54.400 ms | 0.86x |
| `nns dep 1000` | 7.663 ms | 11.300 ms | 1.47x |
| `nns dep asym 1000` | 7.606 ms | 9.800 ms | 1.29x |
| `nns copula 1000` | 0.400 ms | 4.800 ms | 12.01x |
| `nns causation 1000` | 16.700 ms | 97.600 ms | 5.84x |
| `nns norm 1000x3` | 0.131 ms | 1.500 ms | 11.47x |
| `nns distance 1000x3` | 0.756 ms | 1.160 ms | 1.53x |
| `nns distance bulk 1000x3 100` | 6.687 ms | 9.200 ms | 1.38x |
| `nns distance class 500x3` | 0.613 ms | 0.900 ms | 1.47x |
| `nns distance bulk class 500x3 50` | 1.526 ms | 2.260 ms | 1.48x |
| `nns diff sin` | 1.334 ms | 4.050 ms | 3.04x |
| `dy dx numeric eval points` | 24.026 ms | 53.333 ms | 2.22x |
| `dy_d`, scalar wrt=1, eval_points=mean, N=2, T_obs=100 | 94.763 ms | 266.200 ms | 2.81x |
| `dy_d`, scalar wrt=1, eval_points=median, N=2, T_obs=100 | 98.229 ms | 245.600 ms | 2.50x |
| `dy_d`, scalar wrt=1, eval_points=last, N=2, T_obs=100 | 93.825 ms | 279.200 ms | 2.98x |
| `dy_d`, scalar wrt=1, eval_points=obs, N=2, T_obs=100 | 95.522 ms | 290.000 ms | 3.04x |
| `dy_d`, scalar wrt=1, eval_points=apd, N=2, T_obs=100 | 530.328 ms | 790.200 ms | 1.49x |
| `nns anova 100x2` | 7.369 ms | 4.400 ms | 0.60x |
| `nns part 500` | 0.596 ms | 3.750 ms | 6.29x |
| `nns reg 500` | 63.956 ms | 37.200 ms | 0.58x |
| `nns reg 200 confidence interval` | 53.848 ms | 93.400 ms | 1.73x |
| `nns reg 200 smooth` | 11.976 ms | 37.000 ms | 3.09x |
| `nns reg factor predictor 200` | 21.581 ms | 119.400 ms | 5.53x |
| `nns reg factor predictor dimred 120` | 25.863 ms | 35.800 ms | 1.38x |
| `nns reg class 200` | 11.788 ms | 33.200 ms | 2.82x |
| `nns reg class 200 confidence interval` | 22.882 ms | 57.000 ms | 2.49x |
| `nns reg dimred 200x3` | 34.003 ms | 42.600 ms | 1.25x |
| `nns m reg 200x3` | 85.174 ms | 97.600 ms | 1.15x |
| `nns m reg 200x3 confidence interval` | 88.800 ms | 130.400 ms | 1.47x |
| `nns m reg class 200x3` | 48.805 ms | 126.600 ms | 2.59x |
| `nns m reg class 200x3 confidence interval` | 49.444 ms | 150.400 ms | 3.04x |
| `nns stack 100x3` | 207.839 ms | 369.667 ms | 1.78x |
| `nns stack factor predictor 60 method1` | 30.374 ms | 214.333 ms | 7.06x |
| `nns stack mixed factor predictor 60 method2` | 32.947 ms | 168.400 ms | 5.11x |
| `nns stack mixed factor predictor 100x3 method12` | 291.540 ms | 341.333 ms | 1.17x |
| `nns stack 100x3 pred int` | 142.865 ms | 286.000 ms | 2.00x |
| `nns stack 100x3 ts test` | 164.970 ms | 300.333 ms | 1.82x |
| `nns stack class 100x3` | 109.755 ms | 270.667 ms | 2.47x |
| `nns stack class 100x3 pred int` | 111.061 ms | 251.667 ms | 2.27x |
| `nns stack class balance 150x3` | 162.850 ms | 246.000 ms | 1.51x |
| `nns boost 50x3` | 174.099 ms | 2919.500 ms | 16.77x |
| `nns boost 50x3 pred int` | 131.979 ms | 3676.000 ms | 27.85x |
| `nns boost 50x3 ts test` | 143.643 ms | 4128.000 ms | 28.74x |
| `nns boost stochastic 64x11` | 227.449 ms | 3311.667 ms | 14.56x |
| `nns boost stochastic ts test 64x11` | 183.176 ms | 5851.000 ms | 31.94x |
| `nns boost factor predictor 50x2` | 117.477 ms | 4747.000 ms | 40.41x |
| `nns boost multi factor predictor 50x3` | 168.833 ms | 5186.400 ms | 30.72x |
| `nns boost class 50x3` | 159.432 ms | 5790.000 ms | 36.32x |
| `nns boost class 50x3 pred int` | 156.908 ms | 3827.000 ms | 24.39x |
| `nns boost class balance 80x3` | 334.459 ms | 3120.000 ms | 9.33x |
| `nns mode continuous 1000` | 0.440 ms | 0.100 ms | 0.23x |
| `nns seas 1000` | 0.012 ms | 1.100 ms | 94.55x |
| `nns seas 5000` | 0.034 ms | 4.000 ms | 117.90x |
| `nns arma 500 auto nonlin` | 15.628 ms | 313.333 ms | 20.05x |
| `nns arma 500 explicit12 nonlin` | 69.226 ms | 318.333 ms | 4.60x |
| `nns arma 200 explicit4 lin predint` | 144.550 ms | 207.800 ms | 1.44x |
| `nns arma 200 auto nonlin predint` | 154.502 ms | 380.000 ms | 2.46x |
| `nns arma optim 80 small` | 21.054 ms | 163.333 ms | 7.76x |
| `nns_var`, dim_red_method=cor, N=3, T_obs=80, h=3, tau=2 | 666.550 ms | 3603.667 ms | 5.41x |
| `nns_var`, dim_red_method=NNS.dep, N=3, T_obs=80, h=3, tau=2 | 1221.328 ms | 3995.333 ms | 3.27x |
| `nns_var`, dim_red_method=NNS.caus, N=3, T_obs=80, h=3, tau=2 | 2796.609 ms | 7236.000 ms | 2.59x |
| `nns_var`, dim_red_method=all, N=3, T_obs=80, h=3, tau=2 | 3122.019 ms | 7409.667 ms | 2.37x |
| `nns meboot 500 reps100` | 72.212 ms | 78.000 ms | 1.08x |
| `nns meboot 1000 reps100` | 102.386 ms | 108.000 ms | 1.05x |
| `nns mc 500 reps30 by02` | 310.910 ms | 736.000 ms | 2.37x |
| `nns mc 500 reps30 by01` | 600.646 ms | 1437.000 ms | 2.39x |
| `nns ss 1000` | 0.050 ms | 0.200 ms | 3.98x |
| `nns ss 200 ci reps100` | 162.123 ms | 152.667 ms | 0.94x |

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
| `nns_sd_cluster`, degree=1, N=50, T_obs=252 | 24.430 ms | 3.000 ms | measured | 8.14x |
| `sd_efficient_set`, degree=1, N=50, T_obs=252 | 24.672 ms | 2.667 ms | measured | 9.25x |
| `nns_sd_cluster`, degree=2, N=50, T_obs=252 | 15.385 ms | 7.000 ms | measured | 2.20x |
| `sd_efficient_set`, degree=2, N=50, T_obs=252 | 12.319 ms | 2.333 ms | measured | 5.28x |
| `nns_sd_cluster`, degree=1, N=100, T_obs=252 | 34.311 ms | 6.667 ms | measured | 5.15x |
| `sd_efficient_set`, degree=1, N=100, T_obs=252 | 34.209 ms | 5.667 ms | measured | 6.04x |
| `nns_sd_cluster`, degree=2, N=100, T_obs=252 | 17.095 ms | 14.667 ms | measured | 1.17x |
| `sd_efficient_set`, degree=2, N=100, T_obs=252 | 7.550 ms | 4.667 ms | measured | 1.62x |
| `nns_sd_cluster`, degree=2, N=250, T_obs=252 | 72.025 ms | 59.000 ms | measured | 1.22x |
| `sd_efficient_set`, degree=2, N=250, T_obs=252 | 24.456 ms | 14.667 ms | measured | 1.67x |
| `nns_sd_cluster`, degree=2, N=478, T_obs=252 | 277.594 ms | 185.000 ms | measured | 1.50x |
| `sd_efficient_set`, degree=2, N=478, T_obs=252 | 96.173 ms | 39.000 ms | measured | 2.47x |
| `sd_efficient_set`, degree=2, N=100, T_obs=1257 | 29.896 ms | 20.000 ms | measured | 1.49x |
| `nns_sd_cluster`, degree=2, N=250, T_obs=1257 | 247.877 ms | 193.000 ms | measured | 1.28x |
| `sd_efficient_set`, degree=2, N=250, T_obs=1257 | 123.026 ms | 64.667 ms | measured | 1.90x |
| `nns_sd_cluster`, degree=2, N=478, T_obs=1257 | 853.033 ms | 618.000 ms | measured | 1.38x |
| `sd_efficient_set`, degree=2, N=478, T_obs=1257 | 556.045 ms | 178.000 ms | measured | 3.12x |

Additional realistic finance workflow benchmarks:

| Benchmark | Python mean | R mean | R source | Python/R slowdown | Summary metadata |
| --- | ---: | ---: | --- | ---: | --- |
| Lower/upper constituent dispersion ratio, N=100, T_obs=252 | 0.119 ms | n/a | n/a | n/a | n/a |
| Magnificent Seven downside stress components with SPY | 0.332 ms | n/a | n/a | n/a | n/a |
| Magnificent Seven market-downside stress components | 11.141 ms | 41.667 ms | measured | 0.27x | downside obs: 172; stress R2: 0.7852; SPY/GSPC corr: 0.9989; mean abs diff: 0.0003716; max abs diff: 0.01042 |
| Market-relative daily dispersion, full fixture | 17.801 ms | 29.667 ms | measured | 0.60x | signal len: 1257; finite: 1257; next-day corr: 0.06635; SPY/GSPC corr: 0.9989; mean abs diff: 0.0003716; max abs diff: 0.01042 |
| Market-relative rolling dispersion signal, 252d | 15.298 ms | 28.667 ms | measured | 0.53x | signal len: 1006; finite: 1006; next-day corr: 0.03746; SPY/GSPC corr: 0.9989; mean abs diff: 0.0003716; max abs diff: 0.01042 |
| Market-relative rolling dispersion signal, 63d | 16.328 ms | 28.667 ms | measured | 0.57x | signal len: 1195; finite: 1195; next-day corr: 0.02139; SPY/GSPC corr: 0.9989; mean abs diff: 0.0003716; max abs diff: 0.01042 |
| Partial-moment covariance workflow, 1257d-degree1-mean | 99.030 ms | 1.385 s | measured | 0.07x | rows: 1257; cols: 478; matrix N: 478 |
| Partial-moment covariance workflow, 252d-degree1-mean | 29.938 ms | 268.333 ms | measured | 0.11x | rows: 252; cols: 478; matrix N: 478 |
| Partial-moment covariance workflow, 252d-degree2-zero | 43.531 ms | 271.000 ms | measured | 0.16x | rows: 252; cols: 478; matrix N: 478 |
| Rolling SD cluster, 252-day monthly, degree=2, n100 | 776.773 ms | 755.667 ms | measured | 1.03x | windows: 48; avg set: 14.29; avg clusters: 8.375 |
| Rolling SD cluster, 252-day monthly, degree=2, nmax | 11.069 s | 9.132 s | measured | 1.21x | windows: 48; avg set: 29.48; avg clusters: 13.65 |
| Rolling SD cluster, 756-day quarterly, degree=2 | 4.673 s | 3.915 s | measured | 1.19x | windows: 9; avg set: 33.11; avg clusters: 11.89 |
| Rolling SD efficient set, 252-day monthly, degree=2, n100 | 318.017 ms | 257.000 ms | measured | 1.24x | windows: 48; avg set: 14.29; avg turnover: 0.4598 |
| Rolling SD efficient set, 252-day monthly, degree=2, nmax | 3.819 s | 2.024 s | measured | 1.89x | windows: 48; avg set: 29.48; avg turnover: 0.5228 |
| Rolling SD efficient set, 252-day quarterly, degree 1 vs 2 | 13.881 s | 1.748 s | measured | 7.94x | windows: 16; avg d1 set: 468.5; avg d2 set: 29.56 |

Interpretation:

- Guarded prefix-pair evaluation skips curve work for min/mean/identical
  impossible pairs, and the standalone efficient-set path only checks
  already-kept candidates.
- The implementation deliberately follows R's C++ SD algorithmic structure:
  sorted columns, prefix sums, pair-threshold dominance checks, exact guards, and
  no tolerance-based shortcuts.
- Full-fixture PyNNS runs are feasible for research iteration, but R's C++ SD
  core remains materially faster on the largest cluster cases.
