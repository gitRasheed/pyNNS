# pynns

Python port of the NNS (Nonlinear Nonparametric Statistics) R package.

## Status

Alpha / 0.1.0 alpha. Parity-tested port of installed R NNS 12.0, with deferred paths documented below.

The package API is not stable yet. Current work is focused on matching the
reference R implementation before adding higher-level Python conveniences.
Native extension scaffolding has been removed; PyNNS is currently a
pure-Python/NumPy/SciPy port.

## Install

```bash
uv sync --group dev
```

Parity tests that miss the local cache require R plus the NNS R package:

```r
install.packages("NNS")
```

Set `PYNNS_OFFLINE=1` to force parity tests to use only the committed R cache.

Deferred paths:
- default `nns_nowcast` provider fetching; explicit CSV and optional FRED API
  providers are available through `nns_nowcast(fetch=True, provider_backend=...)`
- vectorized `dy_d` wrt for non-mean modes and `mixed=True`
- scalar `dy_d` `eval_points="last"`, `"obs"`, and `"apd"` parity gaps
- direct raw-factor `nns_m_reg`
- boost threshold on stochastic path (`n_features > 10`)

The package is GPL-3.0-only and imports as `pynns`.

For user installs, the planned distribution package is `nns-pm`:

```bash
pip install nns-pm
```

## Run Tests

```bash
uv run pytest
uv run ruff format .
uv run ruff check . --fix
uv run ruff check .
uv run mypy
```

Pytest uses 4 xdist workers by default. Override with
`PYNNS_PYTEST_WORKERS=<n> uv run pytest` for larger or smaller machines.

## Attribution

NNS was created by Fred Viole as the companion R package to Viole, F. and
Nawrocki, D. (2013), *Nonlinear Nonparametric Statistics: Using Partial Moments*.

Upstream: https://github.com/OVVO-Financial/NNS

## License

GPL-3.0-only
