from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pynns import lpm, pm_matrix


@pytest.mark.benchmark
def test_lpm_small(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-3.0, 3.0, 1000)

    result = benchmark(lpm, 1, 0, x)

    assert result == pytest.approx(0.7507507507507507)
    assert isinstance(r_baseline["lpm_small_seconds"], float)


@pytest.mark.benchmark
def test_pm_matrix_10x500(benchmark: Any, r_baseline: dict[str, object]) -> None:
    row = np.arange(1, 501, dtype=np.float64)[:, np.newaxis]
    col = np.arange(1, 11, dtype=np.float64)[np.newaxis, :]
    variable = np.sin(row * col / 11.0) + np.cos((row + 1.0) / (col + 2.0))

    result = benchmark(pm_matrix, 1, 1, "mean", variable, True)

    assert set(result) == {"cupm", "dupm", "dlpm", "clpm", "cov.matrix"}
    assert isinstance(r_baseline["pm_matrix_10x500_seconds"], float)
