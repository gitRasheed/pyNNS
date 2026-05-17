# Benchmarks

Run with:

```bash
mkdir -p docs/benchmark_reports
uv run pytest -n0 -m benchmark --benchmark-enable \
  --benchmark-json=docs/benchmark_reports/benchmark_latest.json tests/benchmarks/
uv run python scripts/update_benchmarks_doc.py docs/benchmark_reports/benchmark_latest.json
```

## Results

`Python speed vs R` is computed as `R baseline / Python mean`. Values above `1.00x` mean Python is faster; values below `1.00x` mean Python is slower.

| Benchmark | Python mean | R baseline | Python speed vs R |
| --- | ---: | ---: | ---: |
| `pm_matrix`, N=10, T_obs=500 | 0.079 ms | 0.400 ms | 5.06x |
| `pm_matrix`, N=50, T_obs=500 | 0.292 ms | 3.800 ms | 13.01x |
| `pm_matrix`, N=100, T_obs=500 | 2.343 ms | 16.800 ms | 7.17x |
| `sd_efficient_set`, degree=2, N=50, T_obs=252 | 13.043 ms | 4.600 ms | 0.35x |
| `nns_sd_cluster`, degree=2, N=50, T_obs=252 | 31.020 ms | 13.800 ms | 0.44x |
| `nns_sd_cluster`, degree=2, N=50, T_obs=252, dendrogram=True | 39.100 ms | 26.333 ms | 0.67x |
| `nns_sd_cluster`, degree=2, N=100, T_obs=252 | 688.179 ms | 131.200 ms | 0.19x |
| `nns_sd_cluster`, degree=1 continuous, N=50, T_obs=252 | 47.505 ms | 21.600 ms | 0.45x |
| `nns_cdf`, T_obs=1000, degree=0, type=CDF | 0.024 ms | 1.150 ms | 47.92x |
| `nns_cdf`, T_obs=1000, degree=1, type=CDF | 0.073 ms | 1.350 ms | 18.49x |
| `nns_cdf`, T_obs=1000, degree=2, type=CDF | 0.093 ms | 1.400 ms | 15.05x |
| `nns_cdf`, T_obs=1000, degree=3, type=CDF | 0.149 ms | 1.500 ms | 10.07x |
| `nns_cdf`, T_obs=1000, degree=2, type=survival | 0.096 ms | 1.400 ms | 14.58x |
| `nns_cdf`, T_obs=1000, degree=2, type=cumulative hazard | 0.102 ms | 1.400 ms | 13.73x |
| `nns_cdf`, T_obs=1000, degree=2, type=hazard | 37.207 ms | 34.000 ms | 0.91x |
| `nns_cdf`, N=3, T_obs=500, degree=1, type=CDF | 54.407 ms | 54.400 ms | 1.00x |
| `nns_dep`, T_obs=1000 | 7.465 ms | 11.300 ms | 1.51x |
| `nns_copula`, T_obs=1000 | 0.524 ms | 4.800 ms | 9.16x |
| `nns_causation`, T_obs=1000 | 23.572 ms | 97.600 ms | 4.14x |
| `nns_norm`, N=3, T_obs=1000 | 0.167 ms | 1.500 ms | 8.98x |
| `nns_distance`, N=3, T_obs=1000 | 1.266 ms | 1.160 ms | 0.92x |
| `nns_distance_bulk`, N=3, T_obs=1000, T_test=100 | 13.780 ms | 9.200 ms | 0.67x |
| `nns_distance` class, N=3, T_obs=500 | 0.725 ms | 0.900 ms | 1.24x |
| `nns_distance_bulk` class, N=3, T_obs=500, T_test=50 | 3.200 ms | 2.260 ms | 0.71x |
| `nns_diff`, f=sin, point=1 | 1.356 ms | 4.050 ms | 2.99x |
| `dy_dx`, T=100, eval_point=c(-1,0,1) | 26.480 ms | 53.333 ms | 2.01x |
| `dy_d`, scalar wrt=1, eval_points=mean, N=2, T_obs=100 | 95.307 ms | 266.200 ms | 2.79x |
| `dy_d`, scalar wrt=1, eval_points=median, N=2, T_obs=100 | 94.077 ms | 245.600 ms | 2.61x |
| `dy_d`, scalar wrt=1, eval_points=last, N=2, T_obs=100 | 94.694 ms | 279.200 ms | 2.95x |
| `dy_d`, scalar wrt=1, eval_points=obs, N=2, T_obs=100 | 96.901 ms | 290.000 ms | 2.99x |
| `dy_d`, scalar wrt=1, eval_points=apd, N=2, T_obs=100 | 496.503 ms | 790.200 ms | 1.59x |
| `nns_anova`, binary, T_obs=100 | 6.271 ms | 4.400 ms | 0.70x |
| `nns_reg`, dim-red cor, N=3, T_obs=200 | 31.526 ms | 42.600 ms | 1.35x |
| `nns_reg`, T_obs=200, T_test=20, confidence_interval=0.95 | 54.557 ms | 93.400 ms | 1.71x |
| `nns_reg`, T_obs=200, T_test=20, smooth=True, confidence_interval=0.95 | 11.504 ms | 37.000 ms | 3.22x |
| `nns_reg`, factor predictor, T_obs=200, T_test=4 | 18.800 ms | 119.400 ms | 6.35x |
| `nns_reg` class, T_obs=200, T_test=20 | 12.168 ms | 33.200 ms | 2.73x |
| `nns_reg` class, T_obs=200, T_test=20, confidence_interval=0.95 | 21.035 ms | 57.000 ms | 2.71x |
| `nns_m_reg`, N=3, T_obs=200 | 84.356 ms | 97.600 ms | 1.16x |
| `nns_m_reg`, N=3, T_obs=200, T_test=20, confidence_interval=0.95 | 90.106 ms | 130.400 ms | 1.45x |
| `nns_m_reg` class, N=3, T_obs=200, T_test=20 | 50.501 ms | 126.600 ms | 2.51x |
| `nns_m_reg` class, N=3, T_obs=200, T_test=20, confidence_interval=0.95 | 47.601 ms | 150.400 ms | 3.16x |
| `nns_stack`, N=3, T_obs=100, T_test=20 | 197.228 ms | 369.667 ms | 1.87x |
| `nns_stack`, factor predictor method=1, T_obs=60, T_test=5 | 28.700 ms | 214.333 ms | 7.47x |
| `nns_stack`, mixed factor predictor method=2, T_obs=60, T_test=5 | 34.689 ms | 168.400 ms | 4.85x |
| `nns_stack`, mixed factor predictor, method=1,2, T_obs=100, T_test=20 | 270.019 ms | 341.333 ms | 1.26x |
| `nns_stack`, N=3, T_obs=100, T_test=20, pred_int=0.95 | 144.093 ms | 286.000 ms | 1.98x |
| `nns_stack`, N=3, T_obs=100, T_test=20, ts_test=20 | 159.132 ms | 300.333 ms | 1.89x |
| `nns_stack` class, N=3, T_obs=100, T_test=20 | 122.462 ms | 270.667 ms | 2.21x |
| `nns_stack` class, N=3, T_obs=100, T_test=20, pred_int=0.95 | 107.610 ms | 251.667 ms | 2.34x |
| `nns_stack` class balance, N=3, T_obs=150, T_test=20 | 142.479 ms | 246.000 ms | 1.73x |
| `nns_boost`, N=3, T_obs=50, T_test=10 | 180.320 ms | 2919.500 ms | 16.19x |
| `nns_boost`, N=3, T_obs=50, T_test=10, pred_int=0.95 | 125.194 ms | 3676.000 ms | 29.36x |
| `nns_boost`, N=3, T_obs=50, T_test=10, ts_test=8 | 127.812 ms | 4128.000 ms | 32.30x |
| `nns_boost`, N=11, T_obs=64, T_test=3, stochastic epochs=4 | 212.194 ms | 3311.667 ms | 15.61x |
| `nns_boost`, N=11, T_obs=64, T_test=3, stochastic epochs=4, ts_test=5 | 230.165 ms | 5851.000 ms | 25.42x |
| `nns_boost`, factor predictor, T_obs=50, T_test=10 | 122.398 ms | 4747.000 ms | 38.78x |
| `nns_boost`, multiple factor predictors, T_obs=50, T_test=10 | 245.107 ms | 5186.400 ms | 21.16x |
| `nns_boost` class, N=3, T_obs=50, T_test=10 | 161.442 ms | 5790.000 ms | 35.86x |
| `nns_boost` class, N=3, T_obs=50, T_test=10, pred_int=0.95 | 151.179 ms | 3827.000 ms | 25.31x |
| `nns_boost` class balance, N=3, T_obs=80, T_test=10 | 312.715 ms | 3120.000 ms | 9.98x |
| `nns_seas`, T_obs=1000 | 0.011 ms | 1.100 ms | 100.00x |
| `nns_seas`, T_obs=5000 | 0.026 ms | 4.000 ms | 153.85x |
| `nns_arma`, T_obs=500, h=12, seasonal_factor=True, method=nonlin | 17.355 ms | 313.333 ms | 18.05x |
| `nns_arma`, T_obs=500, h=12, seasonal_factor=12, method=nonlin | 86.596 ms | 318.333 ms | 3.68x |
| `nns_arma`, T_obs=200, h=5, seasonal_factor=c(3,4), method=lin, pred_int=0.95 | 153.921 ms | 207.800 ms | 1.35x |
| `nns_arma`, T_obs=200, h=5, seasonal_factor=True, method=nonlin, pred_int=0.95 | 156.288 ms | 380.000 ms | 2.43x |
| `nns_arma_optim`, T_obs=80, h=5, seasonal_factor=c(3:10), lin_only=True | 26.946 ms | 163.333 ms | 6.06x |
| `nns_var`, dim_red_method=cor, N=3, T_obs=80, h=3, tau=2 | 652.514 ms | 3603.667 ms | 5.52x |
| `nns_var`, dim_red_method=NNS.dep, N=3, T_obs=80, h=3, tau=2 | 1189.121 ms | 3995.333 ms | 3.36x |
| `nns_var`, dim_red_method=NNS.caus, N=3, T_obs=80, h=3, tau=2 | 2606.211 ms | 7236.000 ms | 2.78x |
| `nns_var`, dim_red_method=all, N=3, T_obs=80, h=3, tau=2 | 2979.431 ms | 7409.667 ms | 2.49x |
| `nns_meboot`, T_obs=500, reps=100, rho=0 | 74.937 ms | 78.000 ms | 1.04x |
| `nns_meboot`, T_obs=1000, reps=100, rho=0 | 115.564 ms | 108.000 ms | 0.93x |
| `nns_mc`, T_obs=500, reps=30, by=0.2 | 319.045 ms | 736.000 ms | 2.31x |
| `nns_mc`, T_obs=500, reps=30, by=0.1 | 608.958 ms | 1437.000 ms | 2.36x |
| `nns_ss`, T_obs=1000 | 0.413 ms | 0.200 ms | 0.48x |
| `nns_ss`, T_obs=200, reps=100, confidence_interval=TRUE | 163.931 ms | 152.667 ms | 0.93x |
