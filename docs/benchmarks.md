# Benchmarks

Run with:

```bash
uv run pytest -n0 -m benchmark --benchmark-enable tests/benchmarks/
```

## Results

`Python faster vs R` is computed as `(R baseline / Python mean - 1) * 100`. Positive values mean Python is faster; negative values mean Python is slower.

| Benchmark | Python mean | R baseline | Python faster vs R |
| --- | ---: | ---: | ---: |
| `pm_matrix`, N=10, T_obs=500 | 0.079 ms | 0.400 ms | +406.3% |
| `pm_matrix`, N=50, T_obs=500 | 0.292 ms | 3.800 ms | +1201.4% |
| `pm_matrix`, N=100, T_obs=500 | 2.343 ms | 16.800 ms | +617.0% |
| `sd_efficient_set`, degree=2, N=50, T_obs=252 | 13.043 ms | 4.600 ms | -64.7% |
| `nns_sd_cluster`, degree=2, N=50, T_obs=252 | 31.020 ms | 13.800 ms | -55.5% |
| `nns_sd_cluster`, degree=2, N=50, T_obs=252, dendrogram=True | 39.100 ms | 26.333 ms | -32.7% |
| `nns_sd_cluster`, degree=2, N=100, T_obs=252 | 688.179 ms | 131.200 ms | -80.9% |
| `nns_sd_cluster`, degree=1 continuous, N=50, T_obs=252 | 47.505 ms | 21.600 ms | -54.5% |
| `nns_cdf`, T_obs=1000, degree=0, type=CDF | 0.024 ms | 1.150 ms | +4691.7% |
| `nns_cdf`, T_obs=1000, degree=1, type=CDF | 0.073 ms | 1.350 ms | +1749.3% |
| `nns_cdf`, T_obs=1000, degree=2, type=CDF | 0.093 ms | 1.400 ms | +1405.4% |
| `nns_cdf`, T_obs=1000, degree=3, type=CDF | 0.149 ms | 1.500 ms | +906.7% |
| `nns_cdf`, T_obs=1000, degree=2, type=survival | 0.096 ms | 1.400 ms | +1358.3% |
| `nns_cdf`, T_obs=1000, degree=2, type=cumulative hazard | 0.102 ms | 1.400 ms | +1272.5% |
| `nns_cdf`, T_obs=1000, degree=2, type=hazard | 37.207 ms | 34.000 ms | -8.6% |
| `nns_cdf`, N=3, T_obs=500, degree=1, type=CDF | 54.407 ms | 54.400 ms | -0.0% |
| `nns_dep`, T_obs=1000 | 7.465 ms | 11.300 ms | +51.4% |
| `nns_copula`, T_obs=1000 | 0.524 ms | 4.800 ms | +816.0% |
| `nns_causation`, T_obs=1000 | 23.572 ms | 97.600 ms | +314.1% |
| `nns_norm`, N=3, T_obs=1000 | 0.167 ms | 1.500 ms | +798.2% |
| `nns_distance`, N=3, T_obs=1000 | 1.266 ms | 1.160 ms | -8.4% |
| `nns_distance_bulk`, N=3, T_obs=1000, T_test=100 | 13.780 ms | 9.200 ms | -33.2% |
| `nns_distance` class, N=3, T_obs=500 | 0.725 ms | 0.900 ms | +24.1% |
| `nns_distance_bulk` class, N=3, T_obs=500, T_test=50 | 3.200 ms | 2.260 ms | -29.4% |
| `nns_diff`, f=sin, point=1 | 1.356 ms | 4.050 ms | +198.7% |
| `dy_dx`, T=100, eval_point=c(-1,0,1) | 26.480 ms | 53.333 ms | +101.4% |
| `nns_anova`, binary, T_obs=100 | 6.271 ms | 4.400 ms | -29.8% |
| `nns_reg`, dim-red cor, N=3, T_obs=200 | 31.526 ms | 42.600 ms | +35.1% |
| `nns_reg`, T_obs=200, T_test=20, confidence_interval=0.95 | 54.557 ms | 93.400 ms | +71.2% |
| `nns_reg`, T_obs=200, T_test=20, smooth=True, confidence_interval=0.95 | 11.504 ms | 37.000 ms | +221.6% |
| `nns_reg`, factor predictor, T_obs=200, T_test=4 | 18.800 ms | 119.400 ms | +535.1% |
| `nns_reg` class, T_obs=200, T_test=20 | 12.168 ms | 33.200 ms | +172.8% |
| `nns_reg` class, T_obs=200, T_test=20, confidence_interval=0.95 | 21.035 ms | 57.000 ms | +171.0% |
| `nns_m_reg`, N=3, T_obs=200 | 84.356 ms | 97.600 ms | +15.7% |
| `nns_m_reg`, N=3, T_obs=200, T_test=20, confidence_interval=0.95 | 90.106 ms | 130.400 ms | +44.7% |
| `nns_m_reg` class, N=3, T_obs=200, T_test=20 | 50.501 ms | 126.600 ms | +150.7% |
| `nns_m_reg` class, N=3, T_obs=200, T_test=20, confidence_interval=0.95 | 47.601 ms | 150.400 ms | +216.0% |
| `nns_stack`, N=3, T_obs=100, T_test=20 | 197.228 ms | 369.667 ms | +87.4% |
| `nns_stack`, factor predictor method=1, T_obs=60, T_test=5 | 28.700 ms | 214.333 ms | +646.8% |
| `nns_stack`, mixed factor predictor method=2, T_obs=60, T_test=5 | 34.689 ms | 168.400 ms | +385.5% |
| `nns_stack`, mixed factor predictor, method=1,2, T_obs=100, T_test=20 | 270.019 ms | 341.333 ms | +26.4% |
| `nns_stack`, N=3, T_obs=100, T_test=20, pred_int=0.95 | 144.093 ms | 286.000 ms | +98.5% |
| `nns_stack`, N=3, T_obs=100, T_test=20, ts_test=20 | 159.132 ms | 300.333 ms | +88.7% |
| `nns_stack` class, N=3, T_obs=100, T_test=20 | 122.462 ms | 270.667 ms | +121.0% |
| `nns_stack` class, N=3, T_obs=100, T_test=20, pred_int=0.95 | 107.610 ms | 251.667 ms | +133.9% |
| `nns_stack` class balance, N=3, T_obs=150, T_test=20 | 142.479 ms | 246.000 ms | +72.7% |
| `nns_boost`, N=3, T_obs=50, T_test=10 | 180.320 ms | 2919.500 ms | +1519.1% |
| `nns_boost`, N=3, T_obs=50, T_test=10, pred_int=0.95 | 125.194 ms | 3676.000 ms | +2836.2% |
| `nns_boost`, N=3, T_obs=50, T_test=10, ts_test=8 | 127.812 ms | 4128.000 ms | +3129.7% |
| `nns_boost`, N=11, T_obs=64, T_test=3, stochastic epochs=4 | 212.194 ms | 3311.667 ms | +1460.7% |
| `nns_boost`, N=11, T_obs=64, T_test=3, stochastic epochs=4, ts_test=5 | 230.165 ms | 5851.000 ms | +2442.1% |
| `nns_boost`, factor predictor, T_obs=50, T_test=10 | 122.398 ms | 4747.000 ms | +3778.3% |
| `nns_boost`, multiple factor predictors, T_obs=50, T_test=10 | 245.107 ms | 5186.400 ms | +2016.0% |
| `nns_boost` class, N=3, T_obs=50, T_test=10 | 161.442 ms | 5790.000 ms | +3486.4% |
| `nns_boost` class, N=3, T_obs=50, T_test=10, pred_int=0.95 | 151.179 ms | 3827.000 ms | +2431.4% |
| `nns_boost` class balance, N=3, T_obs=80, T_test=10 | 312.715 ms | 3120.000 ms | +897.7% |
| `nns_seas`, T_obs=1000 | 0.011 ms | 1.100 ms | +9900.0% |
| `nns_seas`, T_obs=5000 | 0.026 ms | 4.000 ms | +15284.6% |
| `nns_arma`, T_obs=500, h=12, seasonal_factor=True, method=nonlin | 17.355 ms | 313.333 ms | +1705.4% |
| `nns_arma`, T_obs=500, h=12, seasonal_factor=12, method=nonlin | 86.596 ms | 318.333 ms | +267.6% |
| `nns_arma`, T_obs=200, h=5, seasonal_factor=c(3,4), method=lin, pred_int=0.95 | 153.921 ms | 207.800 ms | +35.0% |
| `nns_arma`, T_obs=200, h=5, seasonal_factor=True, method=nonlin, pred_int=0.95 | 156.288 ms | 380.000 ms | +143.1% |
| `nns_arma_optim`, T_obs=80, h=5, seasonal_factor=c(3:10), lin_only=True | 26.946 ms | 163.333 ms | +506.1% |
| `nns_meboot`, T_obs=500, reps=100, rho=0 | 74.937 ms | 78.000 ms | +4.1% |
| `nns_meboot`, T_obs=1000, reps=100, rho=0 | 115.564 ms | 108.000 ms | -6.5% |
| `nns_mc`, T_obs=500, reps=30, by=0.2 | 319.045 ms | 736.000 ms | +130.7% |
| `nns_mc`, T_obs=500, reps=30, by=0.1 | 608.958 ms | 1437.000 ms | +136.0% |
| `nns_ss`, T_obs=1000 | 0.413 ms | 0.200 ms | -51.6% |
| `nns_ss`, T_obs=200, reps=100, confidence_interval=TRUE | 163.931 ms | 152.667 ms | -6.9% |
