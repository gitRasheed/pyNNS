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
