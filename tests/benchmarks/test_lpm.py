from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pynns import lpm, nns_causation, nns_copula, nns_dep, nns_norm, pm_matrix, sd_efficient_set


@pytest.mark.benchmark
def test_lpm_small(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-3.0, 3.0, 1000)

    result = benchmark(lpm, 1, 0, x)

    assert result == pytest.approx(0.7507507507507507)
    assert isinstance(r_baseline["lpm_small_seconds"], float)


@pytest.mark.benchmark
@pytest.mark.parametrize("n_variables", [10, 50, 100])
def test_pm_matrix_scale(
    benchmark: Any,
    r_baseline: dict[str, object],
    n_variables: int,
) -> None:
    row = np.arange(1, 501, dtype=np.float64)[:, np.newaxis]
    col = np.arange(1, n_variables + 1, dtype=np.float64)[np.newaxis, :]
    variable = np.sin(row * col / 11.0) + np.cos((row + 1.0) / (col + 2.0))

    result = benchmark(pm_matrix, 1, 1, "mean", variable, True)

    assert set(result) == {"cupm", "dupm", "dlpm", "clpm", "cov.matrix"}
    assert isinstance(r_baseline[f"pm_matrix_{n_variables}x500_seconds"], float)


@pytest.mark.benchmark
def test_sd_efficient_set_degree_2_scale(
    benchmark: Any,
    r_baseline: dict[str, object],
) -> None:
    row = np.arange(1, 253, dtype=np.float64)[:, np.newaxis]
    col = np.arange(1, 51, dtype=np.float64)[np.newaxis, :]
    returns = np.sin(row * col / 17.0) + np.cos((row + 3.0) / (col + 5.0))

    result = benchmark(sd_efficient_set, returns, 2)

    assert all(0 <= index < 50 for index in result)
    assert isinstance(r_baseline["sd_efficient_set_50x252_degree2_seconds"], float)


@pytest.mark.benchmark
def test_nns_dep_1000(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-3.0, 3.0, 1000)
    y = np.sin(x) + 0.05 * np.cos(7.0 * x)

    result = benchmark(nns_dep, x, y)

    assert set(result) == {"Correlation", "Dependence"}
    assert isinstance(r_baseline["nns_dep_1000_seconds"], float)


@pytest.mark.benchmark
def test_nns_copula_1000(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-3.0, 3.0, 1000)
    y = np.sin(x) + 0.05 * np.cos(7.0 * x)

    result = benchmark(nns_copula, x, y)

    assert 0.0 <= result <= 1.0
    assert isinstance(r_baseline["nns_copula_1000_seconds"], float)


@pytest.mark.benchmark
def test_nns_causation_1000(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-3.0, 3.0, 1000)
    y = np.sin(x) + 0.05 * np.cos(7.0 * x)

    result = benchmark(nns_causation, x, y)

    assert "Causation.x.given.y" in result
    assert "Causation.y.given.x" in result
    assert isinstance(r_baseline["nns_causation_1000_seconds"], float)


@pytest.mark.benchmark
def test_nns_norm_1000x3(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-2.0, 2.0, 1000)
    variable = np.column_stack((x + 3.0, x**2 + 1.0, np.sin(x) + 2.0))

    result = benchmark(nns_norm, variable)

    assert result.shape == variable.shape
    assert isinstance(r_baseline["nns_norm_1000x3_seconds"], float)
