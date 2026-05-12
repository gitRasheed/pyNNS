# pynns

Python port of the NNS (Nonlinear Nonparametric Statistics) R package.

## Status

Pre-alpha, parity-tested port of NNS 12.0.

The package API is not stable yet. Current work is focused on matching the
reference R implementation before adding higher-level Python conveniences.
Native extension scaffolding has been removed; PyNNS is currently a
pure-Python/NumPy/SciPy port.

## Install

```bash
uv sync --extra dev
```

Parity tests that miss the local cache require R plus the NNS R package:

```r
install.packages("NNS")
```

Set `PYNNS_OFFLINE=1` to force parity tests to use only the committed R cache.

## Run Tests

```bash
uv run pytest
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
