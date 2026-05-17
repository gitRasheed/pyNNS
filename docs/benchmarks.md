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
| `lpm small` | 0.012 ms | 0.075 ms | 6.28x |
| `pm matrix scale, 10` | 0.091 ms | 0.400 ms | 4.42x |
| `pm matrix scale, 50` | 0.339 ms | 3.800 ms | 11.20x |
| `pm matrix scale, 100` | 2.100 ms | 16.800 ms | 8.00x |
| `sd efficient set degree 2 scale` | 13.695 ms | 4.600 ms | 0.34x |
| `nns sd cluster 252x50 degree2` | 30.078 ms | 13.800 ms | 0.46x |
| `nns sd cluster 252x50 degree2 dendrogram` | 31.138 ms | 26.333 ms | 0.85x |
| `nns cdf 1000 degree0` | 0.028 ms | 1.150 ms | 40.95x |
| `nns cdf 1000 degree2` | 0.104 ms | 1.400 ms | 13.51x |
| `nns cdf 500x3 degree1` | 59.454 ms | 54.400 ms | 0.92x |
| `nns dep 1000` | 7.580 ms | 11.300 ms | 1.49x |
| `nns dep asym 1000` | 7.640 ms | 9.800 ms | 1.28x |
| `nns copula 1000` | 0.413 ms | 4.800 ms | 11.61x |
| `nns causation 1000` | 19.997 ms | 97.600 ms | 4.88x |
| `nns norm 1000x3` | 0.167 ms | 1.500 ms | 9.00x |
| `nns distance 1000x3` | 1.365 ms | 1.160 ms | 0.85x |
| `nns distance bulk 1000x3 100` | 7.492 ms | 9.200 ms | 1.23x |
| `nns distance class 500x3` | 0.699 ms | 0.900 ms | 1.29x |
| `nns distance bulk class 500x3 50` | 1.614 ms | 2.260 ms | 1.40x |
| `nns diff sin` | 1.841 ms | 4.050 ms | 2.20x |
| `dy dx numeric eval points` | 28.360 ms | 53.333 ms | 1.88x |
| `dy_d`, scalar wrt=1, eval_points=mean, N=2, T_obs=100 | 96.274 ms | 266.200 ms | 2.77x |
| `dy_d`, scalar wrt=1, eval_points=median, N=2, T_obs=100 | 102.993 ms | 245.600 ms | 2.38x |
| `dy_d`, scalar wrt=1, eval_points=last, N=2, T_obs=100 | 102.215 ms | 279.200 ms | 2.73x |
| `dy_d`, scalar wrt=1, eval_points=obs, N=2, T_obs=100 | 106.264 ms | 290.000 ms | 2.73x |
| `dy_d`, scalar wrt=1, eval_points=apd, N=2, T_obs=100 | 574.712 ms | 790.200 ms | 1.37x |
| `nns anova 100x2` | 8.225 ms | 4.400 ms | 0.53x |
| `nns part 500` | 0.604 ms | 3.750 ms | 6.21x |
| `nns reg 500` | 63.723 ms | 37.200 ms | 0.58x |
| `nns reg 200 confidence interval` | 58.271 ms | 93.400 ms | 1.60x |
| `nns reg 200 smooth` | 13.186 ms | 37.000 ms | 2.81x |
| `nns reg factor predictor 200` | 23.165 ms | 119.400 ms | 5.15x |
| `nns reg factor predictor dimred 120` | 26.373 ms | 35.800 ms | 1.36x |
| `nns reg class 200` | 11.695 ms | 33.200 ms | 2.84x |
| `nns reg class 200 confidence interval` | 27.992 ms | 57.000 ms | 2.04x |
| `nns reg dimred 200x3` | 37.936 ms | 42.600 ms | 1.12x |
| `nns m reg 200x3` | 94.724 ms | 97.600 ms | 1.03x |
| `nns m reg 200x3 confidence interval` | 95.743 ms | 130.400 ms | 1.36x |
| `nns m reg class 200x3` | 53.882 ms | 126.600 ms | 2.35x |
| `nns m reg class 200x3 confidence interval` | 53.768 ms | 150.400 ms | 2.80x |
| `nns stack 100x3` | 216.808 ms | 369.667 ms | 1.71x |
| `nns stack factor predictor 60 method1` | 29.869 ms | 214.333 ms | 7.18x |
| `nns stack mixed factor predictor 60 method2` | 34.763 ms | 168.400 ms | 4.84x |
| `nns stack mixed factor predictor 100x3 method12` | 316.444 ms | 341.333 ms | 1.08x |
| `nns stack 100x3 pred int` | 157.518 ms | 286.000 ms | 1.82x |
| `nns stack 100x3 ts test` | 177.009 ms | 300.333 ms | 1.70x |
| `nns stack class 100x3` | 108.459 ms | 270.667 ms | 2.50x |
| `nns stack class 100x3 pred int` | 112.421 ms | 251.667 ms | 2.24x |
| `nns stack class balance 150x3` | 179.666 ms | 246.000 ms | 1.37x |
| `nns boost 50x3` | 185.555 ms | 2919.500 ms | 15.73x |
| `nns boost 50x3 pred int` | 143.393 ms | 3676.000 ms | 25.64x |
| `nns boost 50x3 ts test` | 151.082 ms | 4128.000 ms | 27.32x |
| `nns boost stochastic 64x11` | 241.029 ms | 3311.667 ms | 13.74x |
| `nns boost stochastic ts test 64x11` | 185.618 ms | 5851.000 ms | 31.52x |
| `nns boost factor predictor 50x2` | 135.084 ms | 4747.000 ms | 35.14x |
| `nns boost multi factor predictor 50x3` | 180.169 ms | 5186.400 ms | 28.79x |
| `nns boost class 50x3` | 167.937 ms | 5790.000 ms | 34.48x |
| `nns boost class 50x3 pred int` | 172.119 ms | 3827.000 ms | 22.23x |
| `nns boost class balance 80x3` | 347.361 ms | 3120.000 ms | 8.98x |
| `nns mode continuous 1000` | 0.467 ms | 0.100 ms | 0.21x |
| `nns seas 1000` | 0.014 ms | 1.100 ms | 79.85x |
| `nns seas 5000` | 0.028 ms | 4.000 ms | 142.47x |
| `nns arma 500 auto nonlin` | 16.973 ms | 313.333 ms | 18.46x |
| `nns arma 500 explicit12 nonlin` | 72.488 ms | 318.333 ms | 4.39x |
| `nns arma 200 explicit4 lin predint` | 158.550 ms | 207.800 ms | 1.31x |
| `nns arma 200 auto nonlin predint` | 164.159 ms | 380.000 ms | 2.31x |
| `nns arma optim 80 small` | 21.570 ms | 163.333 ms | 7.57x |
| `nns_var`, dim_red_method=cor, N=3, T_obs=80, h=3, tau=2 | 691.371 ms | 3603.667 ms | 5.21x |
| `nns_var`, dim_red_method=NNS.dep, N=3, T_obs=80, h=3, tau=2 | 1274.260 ms | 3995.333 ms | 3.14x |
| `nns_var`, dim_red_method=NNS.caus, N=3, T_obs=80, h=3, tau=2 | 2860.078 ms | 7236.000 ms | 2.53x |
| `nns_var`, dim_red_method=all, N=3, T_obs=80, h=3, tau=2 | 3272.555 ms | 7409.667 ms | 2.26x |
| `nns meboot 500 reps100` | 72.653 ms | 78.000 ms | 1.07x |
| `nns meboot 1000 reps100` | 102.670 ms | 108.000 ms | 1.05x |
| `nns mc 500 reps30 by02` | 300.924 ms | 736.000 ms | 2.45x |
| `nns mc 500 reps30 by01` | 614.840 ms | 1437.000 ms | 2.34x |
| `nns ss 1000` | 0.056 ms | 0.200 ms | 3.59x |
| `nns ss 200 ci reps100` | 169.784 ms | 152.667 ms | 0.90x |

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
| `nns_sd_cluster`, degree=1, N=50, T_obs=252 | 27.290 ms | 2.667 ms | measured | 10.23x |
| `sd_efficient_set`, degree=1, N=50, T_obs=252 | 28.160 ms | 3.000 ms | measured | 9.39x |
| `nns_sd_cluster`, degree=2, N=50, T_obs=252 | 16.381 ms | 7.667 ms | measured | 2.14x |
| `sd_efficient_set`, degree=2, N=50, T_obs=252 | 11.949 ms | 2.333 ms | measured | 5.12x |
| `nns_sd_cluster`, degree=1, N=100, T_obs=252 | 98.578 ms | 5.667 ms | measured | 17.40x |
| `sd_efficient_set`, degree=1, N=100, T_obs=252 | 57.518 ms | 4.333 ms | measured | 13.27x |
| `nns_sd_cluster`, degree=2, N=100, T_obs=252 | 67.807 ms | 24.667 ms | measured | 2.75x |
| `sd_efficient_set`, degree=2, N=100, T_obs=252 | 27.513 ms | 6.333 ms | measured | 4.34x |
| `nns_sd_cluster`, degree=2, N=250, T_obs=252 | 372.127 ms | 73.667 ms | measured | 5.05x |
| `sd_efficient_set`, degree=2, N=250, T_obs=252 | 98.961 ms | 19.000 ms | measured | 5.21x |
| `nns_sd_cluster`, degree=2, N=479, T_obs=252 | 1.444 s | 199.667 ms | measured | 7.23x |
| `sd_efficient_set`, degree=2, N=479, T_obs=252 | 282.558 ms | 48.667 ms | measured | 5.81x |
| `sd_efficient_set`, degree=2, N=100, T_obs=1257 | 143.252 ms | 23.333 ms | measured | 6.14x |
| `nns_sd_cluster`, degree=2, N=250, T_obs=1257 | 1.941 s | 213.333 ms | measured | 9.10x |
| `sd_efficient_set`, degree=2, N=250, T_obs=1257 | 692.120 ms | 68.667 ms | measured | 10.08x |
| `nns_sd_cluster`, degree=2, N=479, T_obs=1257 | 9.335 s | 732.000 ms | measured | 12.75x |
| `sd_efficient_set`, degree=2, N=479, T_obs=1257 | 1.810 s | 227.000 ms | measured | 7.97x |

Additional Python-only realistic building-block benchmarks from the same file:

| Benchmark | Python mean |
| --- | ---: |
| Lower/upper constituent dispersion ratio, N=100, T_obs=252 | 0.128 ms |
| Magnificent Seven downside stress components with SPY | 0.361 ms |

Interpretation:

- Guarded prefix-pair evaluation skips curve work for min/mean/identical
  impossible pairs, and the standalone efficient-set path only checks
  already-kept candidates.
- The implementation deliberately follows R's C++ SD algorithmic structure:
  sorted columns, prefix sums, pair-threshold dominance checks, exact guards, and
  no tolerance-based shortcuts.
- Full-fixture PyNNS runs are feasible for research iteration, but R's C++ SD
  core remains materially faster on the largest cluster cases.
