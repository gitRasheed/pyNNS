# PyNNS Examples

These examples are Python-native companions to the upstream R NNS documentation.
They are not copies of the R reports in `reference/NNS/examples/`; they are
small runnable scripts that show the PyNNS API, expected result shapes, and a few
basic identities.

The upstream R repository contains several kinds of material:

- `reference/NNS/man/`: function reference pages.
- `reference/NNS/doc/` and `reference/NNS/vignettes/`: CRAN-style tutorials.
- `reference/NNS/book/`: conceptual book chapters.
- `reference/NNS/examples/`: larger applied reports, PDFs, HTML demos, and case
  studies.

Use those as conceptual references. Use the examples here when you want a short
Python call pattern.

## Runnable Examples

| Topic | Tutorial | Script | Upstream analogue |
|---|---|---|---|
| Partial moments | [partial_moments.md](partial_moments.md) | [partial_moments.py](partial_moments.py) | `NNSvignette_Partial_Moments.Rmd` |
| Dependence | [dependence.md](dependence.md) | [dependence.py](dependence.py) | `NNSvignette_Correlation_and_Dependence.Rmd` |
| Distributions / ANOVA | [distributions_anova.md](distributions_anova.md) | [distributions_anova.py](distributions_anova.py) | `NNSvignette_Comparing_Distributions.Rmd` |
| Regression | [regression.md](regression.md) | [regression.py](regression.py) | `NNSvignette_Clustering_and_Regression.Rmd` |
| Classification | [classification.md](classification.md) | [classification.py](classification.py) | `NNSvignette_Classification.Rmd` |
| Forecasting | [forecasting.md](forecasting.md) | [forecasting.py](forecasting.py) | `NNSvignette_Forecasting.Rmd` |
| Nowcast panel | [nowcast_panel.md](nowcast_panel.md) | [nowcast_panel.py](nowcast_panel.py) | `NNS.nowcast` / `NNS.VAR` material |

Run one example:

```bash
uv run python docs/examples/partial_moments.py
```

Run all examples:

```bash
for example in docs/examples/*.py; do uv run python "$example"; done
```

The main R parity guarantees still live in `tests/parity/`. These examples are
smoke examples, not a replacement for the parity suite.
