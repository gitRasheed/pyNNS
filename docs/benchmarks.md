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
| `nns_diff`, f=sin, point=1 | 1.356 ms | 4.050 ms | 0.33x |
| `nns_anova`, binary, T_obs=100 | 6.271 ms | 4.400 ms | 1.43x |
| `nns_reg`, dim-red cor, N=3, T_obs=200 | 31.526 ms | 42.600 ms | 0.74x |
| `nns_m_reg`, N=3, T_obs=200 | 84.356 ms | 97.600 ms | 0.86x |
