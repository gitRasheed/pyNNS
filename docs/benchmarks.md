# Benchmarks

Run with:

```bash
uv run pytest tests/benchmarks/ --benchmark-only
```

## Results

| Benchmark | Python mean | R baseline | Python / R |
| --- | ---: | ---: | ---: |
| `pm_matrix`, N=10, T_obs=500 | 5.641 ms | 0.400 ms | 14.10x |
| `pm_matrix`, N=50, T_obs=500 | 140.291 ms | 3.800 ms | 36.92x |
| `pm_matrix`, N=100, T_obs=500 | 566.221 ms | 16.800 ms | 33.70x |

`pm_matrix` is currently at least 5x slower than R at small N due to Python
per-call overhead in the inner co-moment loop. Revisit this when adding C++
infrastructure for `nns_reg`, since the same C++ kernel can be exposed for
`pm_matrix` at that point.
