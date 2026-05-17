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
| `lpm small` | 0.012 ms | 0.075 ms | 6.50x |
| `pm matrix scale, 10` | 0.081 ms | 0.400 ms | 4.96x |
| `pm matrix scale, 50` | 0.313 ms | 3.800 ms | 12.14x |
| `pm matrix scale, 100` | 4.436 ms | 16.800 ms | 3.79x |
| `sd efficient set degree 2 scale` | 13.719 ms | 4.600 ms | 0.34x |
| `nns sd cluster 252x50 degree2` | 28.936 ms | 13.800 ms | 0.48x |
| `nns sd cluster 252x50 degree2 dendrogram` | 31.319 ms | 26.333 ms | 0.84x |
| `nns cdf 1000 degree0` | 0.028 ms | 1.150 ms | 40.91x |
| `nns cdf 1000 degree2` | 0.105 ms | 1.400 ms | 13.36x |
| `nns cdf 500x3 degree1` | 60.491 ms | 54.400 ms | 0.90x |
| `nns dep 1000` | 7.815 ms | 11.300 ms | 1.45x |
| `nns dep asym 1000` | 7.752 ms | 9.800 ms | 1.26x |
| `nns copula 1000` | 0.411 ms | 4.800 ms | 11.69x |
| `nns causation 1000` | 17.470 ms | 97.600 ms | 5.59x |
| `nns norm 1000x3` | 0.137 ms | 1.500 ms | 10.96x |
| `nns distance 1000x3` | 0.742 ms | 1.160 ms | 1.56x |
| `nns distance bulk 1000x3 100` | 6.453 ms | 9.200 ms | 1.43x |
| `nns distance class 500x3` | 0.639 ms | 0.900 ms | 1.41x |
| `nns distance bulk class 500x3 50` | 1.514 ms | 2.260 ms | 1.49x |
| `nns diff sin` | 1.390 ms | 4.050 ms | 2.91x |
| `dy dx numeric eval points` | 24.072 ms | 53.333 ms | 2.22x |
| `dy_d`, scalar wrt=1, eval_points=mean, N=2, T_obs=100 | 93.966 ms | 266.200 ms | 2.83x |
| `dy_d`, scalar wrt=1, eval_points=median, N=2, T_obs=100 | 92.736 ms | 245.600 ms | 2.65x |
| `dy_d`, scalar wrt=1, eval_points=last, N=2, T_obs=100 | 95.118 ms | 279.200 ms | 2.94x |
| `dy_d`, scalar wrt=1, eval_points=obs, N=2, T_obs=100 | 93.415 ms | 290.000 ms | 3.10x |
| `dy_d`, scalar wrt=1, eval_points=apd, N=2, T_obs=100 | 527.622 ms | 790.200 ms | 1.50x |
| `nns anova 100x2` | 7.410 ms | 4.400 ms | 0.59x |
| `nns part 500` | 0.579 ms | 3.750 ms | 6.48x |
| `nns reg 500` | 59.065 ms | 37.200 ms | 0.63x |
| `nns reg 200 confidence interval` | 54.436 ms | 93.400 ms | 1.72x |
| `nns reg 200 smooth` | 13.076 ms | 37.000 ms | 2.83x |
| `nns reg factor predictor 200` | 21.377 ms | 119.400 ms | 5.59x |
| `nns reg factor predictor dimred 120` | 25.904 ms | 35.800 ms | 1.38x |
| `nns reg class 200` | 11.954 ms | 33.200 ms | 2.78x |
| `nns reg class 200 confidence interval` | 23.362 ms | 57.000 ms | 2.44x |
| `nns reg dimred 200x3` | 34.221 ms | 42.600 ms | 1.24x |
| `nns m reg 200x3` | 90.316 ms | 97.600 ms | 1.08x |
| `nns m reg 200x3 confidence interval` | 87.897 ms | 130.400 ms | 1.48x |
| `nns m reg class 200x3` | 49.614 ms | 126.600 ms | 2.55x |
| `nns m reg class 200x3 confidence interval` | 50.216 ms | 150.400 ms | 3.00x |
| `nns stack 100x3` | 211.392 ms | 369.667 ms | 1.75x |
| `nns stack factor predictor 60 method1` | 29.364 ms | 214.333 ms | 7.30x |
| `nns stack mixed factor predictor 60 method2` | 33.894 ms | 168.400 ms | 4.97x |
| `nns stack mixed factor predictor 100x3 method12` | 287.189 ms | 341.333 ms | 1.19x |
| `nns stack 100x3 pred int` | 141.967 ms | 286.000 ms | 2.01x |
| `nns stack 100x3 ts test` | 167.354 ms | 300.333 ms | 1.79x |
| `nns stack class 100x3` | 105.467 ms | 270.667 ms | 2.57x |
| `nns stack class 100x3 pred int` | 110.177 ms | 251.667 ms | 2.28x |
| `nns stack class balance 150x3` | 158.134 ms | 246.000 ms | 1.56x |
| `nns boost 50x3` | 176.916 ms | 2919.500 ms | 16.50x |
| `nns boost 50x3 pred int` | 131.367 ms | 3676.000 ms | 27.98x |
| `nns boost 50x3 ts test` | 135.968 ms | 4128.000 ms | 30.36x |
| `nns boost stochastic 64x11` | 230.278 ms | 3311.667 ms | 14.38x |
| `nns boost stochastic ts test 64x11` | 182.047 ms | 5851.000 ms | 32.14x |
| `nns boost factor predictor 50x2` | 122.068 ms | 4747.000 ms | 38.89x |
| `nns boost multi factor predictor 50x3` | 172.094 ms | 5186.400 ms | 30.14x |
| `nns boost class 50x3` | 156.957 ms | 5790.000 ms | 36.89x |
| `nns boost class 50x3 pred int` | 161.080 ms | 3827.000 ms | 23.76x |
| `nns boost class balance 80x3` | 334.607 ms | 3120.000 ms | 9.32x |
| `nns mode continuous 1000` | 0.444 ms | 0.100 ms | 0.23x |
| `nns seas 1000` | 0.012 ms | 1.100 ms | 92.54x |
| `nns seas 5000` | 0.026 ms | 4.000 ms | 151.69x |
| `nns arma 500 auto nonlin` | 15.974 ms | 313.333 ms | 19.62x |
| `nns arma 500 explicit12 nonlin` | 67.557 ms | 318.333 ms | 4.71x |
| `nns arma 200 explicit4 lin predint` | 152.635 ms | 207.800 ms | 1.36x |
| `nns arma 200 auto nonlin predint` | 153.637 ms | 380.000 ms | 2.47x |
| `nns arma optim 80 small` | 20.855 ms | 163.333 ms | 7.83x |
| `nns_var`, dim_red_method=cor, N=3, T_obs=80, h=3, tau=2 | 667.484 ms | 3603.667 ms | 5.40x |
| `nns_var`, dim_red_method=NNS.dep, N=3, T_obs=80, h=3, tau=2 | 1201.887 ms | 3995.333 ms | 3.32x |
| `nns_var`, dim_red_method=NNS.caus, N=3, T_obs=80, h=3, tau=2 | 2730.148 ms | 7236.000 ms | 2.65x |
| `nns_var`, dim_red_method=all, N=3, T_obs=80, h=3, tau=2 | 3096.932 ms | 7409.667 ms | 2.39x |
| `nns meboot 500 reps100` | 68.833 ms | 78.000 ms | 1.13x |
| `nns meboot 1000 reps100` | 106.321 ms | 108.000 ms | 1.02x |
| `nns mc 500 reps30 by02` | 329.133 ms | 736.000 ms | 2.24x |
| `nns mc 500 reps30 by01` | 609.766 ms | 1437.000 ms | 2.36x |
| `nns ss 1000` | 0.053 ms | 0.200 ms | 3.77x |
| `nns ss 200 ci reps100` | 166.830 ms | 152.667 ms | 0.92x |

## Realistic Finance SD North Stars

These benchmarks use the static daily-return fixture at
`tests/fixtures/finance/sp500_daily_returns_2019_2023.csv`. The fixture contains
1257 daily return rows and 479 clean return columns after dropping tickers with
missing or non-finite returns.

Python timings come from `pytest-benchmark`. R timings come from
`scripts/benchmark_realistic_sd_r.R` when `--realistic-sd-r-csv` is supplied to
the updater. Rows marked `manual placeholder` use the last manually recorded R
baseline so Python/R comparisons remain visible when R has not been rerun.

Run only the realistic Python benchmarks with:

```bash
PYNNS_OFFLINE=1 uv run pytest -q -n0 -m benchmark --benchmark-enable \
  --benchmark-json=docs/benchmark_reports/realistic_sd_python_latest.json \
  tests/benchmarks/test_stochastic_dominance_realistic.py
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
| `nns_sd_cluster`, degree=1, N=50, T_obs=252 | 28.239 ms | 2.667 ms | measured | 10.59x |
| `sd_efficient_set`, degree=1, N=50, T_obs=252 | 29.751 ms | 3.000 ms | measured | 9.92x |
| `nns_sd_cluster`, degree=2, N=50, T_obs=252 | 16.262 ms | 7.667 ms | measured | 2.12x |
| `sd_efficient_set`, degree=2, N=50, T_obs=252 | 12.455 ms | 2.333 ms | measured | 5.34x |
| `nns_sd_cluster`, degree=1, N=100, T_obs=252 | 34.189 ms | 5.667 ms | measured | 6.03x |
| `sd_efficient_set`, degree=1, N=100, T_obs=252 | 34.279 ms | 4.333 ms | measured | 7.91x |
| `nns_sd_cluster`, degree=2, N=100, T_obs=252 | 17.298 ms | 24.667 ms | measured | 0.70x |
| `sd_efficient_set`, degree=2, N=100, T_obs=252 | 7.811 ms | 6.333 ms | measured | 1.23x |
| `nns_sd_cluster`, degree=2, N=250, T_obs=252 | 73.409 ms | 73.667 ms | measured | 1.00x |
| `sd_efficient_set`, degree=2, N=250, T_obs=252 | 24.262 ms | 19.000 ms | measured | 1.28x |
| `nns_sd_cluster`, degree=2, N=479, T_obs=252 | 278.985 ms | 199.667 ms | measured | 1.40x |
| `sd_efficient_set`, degree=2, N=479, T_obs=252 | 91.879 ms | 48.667 ms | measured | 1.89x |
| `sd_efficient_set`, degree=2, N=100, T_obs=1257 | 27.638 ms | 23.333 ms | measured | 1.18x |
| `nns_sd_cluster`, degree=2, N=250, T_obs=1257 | 235.910 ms | 213.333 ms | measured | 1.11x |
| `sd_efficient_set`, degree=2, N=250, T_obs=1257 | 119.636 ms | 68.667 ms | measured | 1.74x |
| `nns_sd_cluster`, degree=2, N=479, T_obs=1257 | 916.604 ms | 732.000 ms | measured | 1.25x |
| `sd_efficient_set`, degree=2, N=479, T_obs=1257 | 439.872 ms | 227.000 ms | measured | 1.94x |

Additional Python-only realistic building-block benchmarks from the same file:

| Benchmark | Python mean |
| --- | ---: |
| Lower/upper constituent dispersion ratio, N=100, T_obs=252 | 0.112 ms |
| Magnificent Seven downside stress components with SPY | 0.337 ms |

Interpretation:

- Guarded prefix-pair evaluation skips curve work for min/mean/identical
  impossible pairs. Large efficient-set and cluster paths now use the same lazy
  active-set scan, so they only check already-kept candidates instead of
  materializing a full dominance matrix.
- The implementation deliberately follows R's C++ SD algorithmic structure:
  sorted columns, prefix sums, pair-threshold dominance checks, exact guards, and
  no tolerance-based shortcuts.
- Full-fixture PyNNS runs are feasible for research iteration. R's C++ SD core
  is still faster on the largest cases, but the gap is now roughly 1.25x for the
  full-history/full-width cluster fixture on this run.
