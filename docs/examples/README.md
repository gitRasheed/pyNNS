# PyNNS Examples

These examples are Python-native companions to the upstream R NNS documentation,
not one-for-one copies of the R reports. Each script is deliberately small,
runnable, deterministic, and covered by `tests/invariants/test_examples.py`.

The upstream R repository contains several kinds of material:

- `reference/NNS/man/`: function reference pages.
- `reference/NNS/doc/` and `reference/NNS/vignettes/`: CRAN-style tutorials.
- `reference/NNS/book/`: conceptual book chapters.
- `reference/NNS/examples/`: larger applied reports, PDFs, HTML demos, and case
  studies.

Use those upstream files as conceptual references. Use the examples here when
you want short Python call patterns that are kept in sync with PyNNS.

## Runnable Examples

| Topic | Script | What it demonstrates | Upstream analogue |
|---|---|---|---|
| Partial moments | [partial_moments.py](partial_moments.py) | `lpm`, `upm`, degree-zero probability split, variance decomposition, `nns_moments` | `NNSvignette_Partial_Moments.Rmd` |
| Dependence | [dependence.py](dependence.py) | `nns_dep`, `nns_cor`, linear vs nonlinear relationships | `NNSvignette_Correlation_and_Dependence.Rmd` |
| Distributions / ANOVA | [distributions_anova.py](distributions_anova.py) | `nns_cdf`, `nns_anova`, certainty output | `NNSvignette_Comparing_Distributions.Rmd` |
| Regression | [regression.py](regression.py) | `nns_reg`, fitted values, point estimates, regression output shape | `NNSvignette_Clustering_and_Regression.Rmd` |
| Classification | [classification.py](classification.py) | `nns_reg(..., type="class")`, numeric class-code predictions | `NNSvignette_Classification.Rmd` |
| Forecasting | [forecasting.py](forecasting.py) | `nns_arma`, `nns_arma_optim`, `nns_var` | `NNSvignette_Forecasting.Rmd` |
| Nowcast panel | [nowcast_panel.py](nowcast_panel.py) | deterministic user-supplied panel, date metadata, VAR-backed forecast output | `NNS.nowcast` / `NNS.VAR` material |

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

## Why There Are No Per-Example Markdown Tutorials

The upstream R repo already has full narrative vignettes, a book, and large
application reports. Thin Python markdown files that merely restate a script add
noise without adding useful documentation. Until PyNNS has deeper Python-native
tutorials, this folder keeps the narrative in one index and makes the runnable
code the source of truth.

If deeper tutorials are added later, they should be substantial documents that
translate the R vignettes into Python workflows, including interpretation,
expected outputs, and PyNNS-specific caveats.
