# PyNNS

Nonlinear nonparametric statistics in Python.

PyNNS is an alpha Python port of the R NNS 12.0 package. It provides
partial-moment statistics, nonlinear dependence, regression, classification,
forecasting, bootstrap, and related tools in Python on top of NumPy, SciPy, and
Polars.

- PyPI package: `nns-pm`
- Import name: `pynns`
- Status: alpha, parity-focused
- R required for normal use: no
- License: GPL-3.0-only

## Documentation

- [API status and known gaps](docs/api_status.md)
- [Behavior conventions and intentional divergences](docs/conventions.md)
- [Benchmarks](docs/benchmarks.md)
- [Examples](docs/examples/README.md)

PyNNS aims to match installed R NNS public behavior where it is stable,
documented, and useful. It does not try to reproduce every R internal helper,
plotting side effect, data-frame quirk, or hidden runtime data fetch.

## Installation

```bash
pip install nns-pm
```

Optional FRED nowcast provider:

```bash
pip install "nns-pm[fred]"
```

The optional FRED provider loads `fredapi` only when used. PyNNS does not
auto-load `.env` files and does not fetch network data unless an explicit
provider is passed.

R and the R `NNS` package are only needed by maintainers regenerating live parity
fixtures. Normal Python users do not need R installed.

## Minimal Examples

### Partial Moments

```python
import numpy as np
from pynns import lpm, upm

x = np.array([-2.0, -1.0, 0.5, 3.0])

downside = lpm(2, 0.0, x)
upside = upm(2, 0.0, x)
```

### Nonlinear Dependence

```python
import numpy as np
from pynns import nns_dep

x = np.linspace(-2.0, 2.0, 50)
y = x**2

result = nns_dep(x, y)
print(result["Dependence"], result["Correlation"])
```

### Regression

```python
import numpy as np
from pynns import nns_reg

x = np.linspace(-3.0, 3.0, 80)
y = np.sin(x) + 0.2 * x

fit = nns_reg(x, y, point_est=np.array([-1.0, 0.0, 1.0]))
print(fit["Point.est"])
```

### User-Supplied Nowcast Panel

```python
from collections import OrderedDict
from pynns import nns_nowcast_panel

panel = OrderedDict(
    {
        "series_a": [1.0, 1.2, 1.4, 1.5, 1.7],
        "series_b": [2.0, 1.9, 2.1, 2.3, 2.4],
    }
)

forecast = nns_nowcast_panel(panel, h=2, tau=1)
```

Default live `nns_nowcast()` fetching is intentionally not implemented. Use
`nns_nowcast_panel(...)` for deterministic data, or
`nns_nowcast(fetch=True, provider_backend=...)` with an explicit provider such
as `CsvNowcastProvider` or optional `FredApiNowcastProvider`.

## Main Features

- Core partial moments: `lpm`, `upm`, ratios, co-moments, and partial-moment
  matrices.
- Dependence, copula, causation, CDF, distance, and normalization helpers.
- Regression, multivariate regression, classification, stack, and boost paths.
- ARMA, VAR, seasonality, and deterministic user-panel nowcasting.
- Monte Carlo and maximum-entropy bootstrap wrappers.
- Stochastic dominance, stochastic superiority, and SD clustering.
- R parity and Python-native invariant tests for public behavior.

## Current Limitations

PyNNS is not full R parity yet. The main mathematical gap is `dy_d`:

- Scalar `dy_d(eval_points="last")`, `"obs"`, and `"apd"` have known parity
  gaps and are kept as xfail tests.
- Vectorized non-mean `dy_d` and vectorized `mixed=True` modes are guarded.
- Default hidden live `nns_nowcast()` provider fetching is guarded by design.
- Direct raw-factor `nns_m_reg(..., factor_2_dummy=True)` is guarded; use the
  public `nns_reg` factor-expansion path instead.

See the [API status table](docs/api_status.md) for the full status table.

## Testing

PyNNS tests public behavior against installed R NNS where useful. Tests compare
public keys, shapes, labels, signs, selected variables, and numerical values
within documented tolerances.

Exact random-stream parity is not expected for stochastic paths because PyNNS
uses NumPy RNG while R NNS uses R's RNG. Those paths are tested structurally and
statistically.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run mypy
```

Offline parity-cache run:

```bash
PYNNS_OFFLINE=1 uv run pytest -q -m "not benchmark and not stochastic"
```

## Attribution

NNS was created by Fred Viole as the companion R package to Viole, F. and
Nawrocki, D. (2013), *Nonlinear Nonparametric Statistics: Using Partial Moments*.

Upstream: [OVVO-Financial/NNS](https://github.com/OVVO-Financial/NNS)
