# Benchmarks

Run with:

```bash
uv run pytest tests/benchmarks/ --benchmark-only
```

## Results

| Benchmark | Python mean | R baseline | Python / R |
| --- | ---: | ---: | ---: |
| `pm_matrix`, N=10, T_obs=500 | 0.079 ms | 0.400 ms | 0.20x |
| `pm_matrix`, N=50, T_obs=500 | 0.292 ms | 3.800 ms | 0.08x |
| `pm_matrix`, N=100, T_obs=500 | 2.343 ms | 16.800 ms | 0.14x |
| `sd_efficient_set`, degree=2, N=50, T_obs=252 | 13.043 ms | 4.600 ms | 2.84x |
| `nns_dep`, T_obs=1000 | 7.465 ms | 11.300 ms | 0.66x |
| `nns_copula`, T_obs=1000 | 0.524 ms | 4.800 ms | 0.11x |
| `nns_causation`, T_obs=1000 | 23.572 ms | 97.600 ms | 0.24x |
| `nns_norm`, N=3, T_obs=1000 | 0.167 ms | 1.500 ms | 0.11x |
| `nns_distance`, N=3, T_obs=1000 | 1.266 ms | 1.160 ms | 1.09x |
| `nns_distance_bulk`, N=3, T_obs=1000, T_test=100 | 13.780 ms | 9.200 ms | 1.50x |
| `nns_distance` class, N=3, T_obs=500 | 0.725 ms | 0.900 ms | 0.81x |
| `nns_distance_bulk` class, N=3, T_obs=500, T_test=50 | 3.200 ms | 2.260 ms | 1.42x |
| `nns_diff`, f=sin, point=1 | 1.356 ms | 4.050 ms | 0.33x |
| `nns_anova`, binary, T_obs=100 | 6.271 ms | 4.400 ms | 1.43x |
| `nns_reg`, dim-red cor, N=3, T_obs=200 | 31.526 ms | 42.600 ms | 0.74x |
| `nns_reg`, T_obs=200, T_test=20, confidence_interval=0.95 | 54.557 ms | 93.400 ms | 0.58x |
| `nns_reg` class, T_obs=200, T_test=20 | 12.168 ms | 33.200 ms | 0.37x |
| `nns_m_reg`, N=3, T_obs=200 | 84.356 ms | 97.600 ms | 0.86x |
| `nns_m_reg`, N=3, T_obs=200, T_test=20, confidence_interval=0.95 | 90.106 ms | 130.400 ms | 0.69x |
| `nns_m_reg` class, N=3, T_obs=200, T_test=20 | 50.501 ms | 126.600 ms | 0.40x |
| `nns_stack`, N=3, T_obs=100, T_test=20 | 197.228 ms | 369.667 ms | 0.53x |
| `nns_stack`, N=3, T_obs=100, T_test=20, pred_int=0.95 | 144.093 ms | 286.000 ms | 0.50x |
| `nns_stack`, N=3, T_obs=100, T_test=20, ts_test=20 | 159.132 ms | 300.333 ms | 0.53x |
| `nns_stack` class, N=3, T_obs=100, T_test=20 | 122.462 ms | 270.667 ms | 0.45x |
| `nns_stack` class balance, N=3, T_obs=150, T_test=20 | 142.479 ms | 246.000 ms | 0.58x |
| `nns_boost`, N=3, T_obs=50, T_test=10 | 180.320 ms | 2919.500 ms | 0.06x |
| `nns_boost` class, N=3, T_obs=50, T_test=10 | 161.442 ms | 5790.000 ms | 0.03x |
| `nns_boost` class balance, N=3, T_obs=80, T_test=10 | 312.715 ms | 3120.000 ms | 0.10x |
| `nns_seas`, T_obs=1000 | 0.011 ms | 1.100 ms | 0.01x |
| `nns_seas`, T_obs=5000 | 0.026 ms | 4.000 ms | 0.01x |
| `nns_arma`, T_obs=500, h=12, seasonal_factor=True, method=nonlin | 17.355 ms | 313.333 ms | 0.06x |
| `nns_arma`, T_obs=500, h=12, seasonal_factor=12, method=nonlin | 86.596 ms | 318.333 ms | 0.27x |
| `nns_arma`, T_obs=200, h=5, seasonal_factor=c(3,4), method=lin, pred_int=0.95 | 153.921 ms | 207.800 ms | 0.74x |
| `nns_arma`, T_obs=200, h=5, seasonal_factor=True, method=nonlin, pred_int=0.95 | 156.288 ms | 380.000 ms | 0.41x |
| `nns_meboot`, T_obs=500, reps=100, rho=0 | 74.937 ms | 78.000 ms | 0.96x |
| `nns_meboot`, T_obs=1000, reps=100, rho=0 | 115.564 ms | 108.000 ms | 1.07x |
| `nns_mc`, T_obs=500, reps=30, by=0.2 | 319.045 ms | 736.000 ms | 0.43x |
| `nns_mc`, T_obs=500, reps=30, by=0.1 | 608.958 ms | 1437.000 ms | 0.42x |
