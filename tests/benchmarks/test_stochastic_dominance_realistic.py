from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pynns import nns_sd_cluster, sd_efficient_set

_FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "finance"
    / "sp500_daily_returns_2019_2023.csv"
)


@pytest.mark.benchmark
@pytest.mark.parametrize("column_count", [50, 100], ids=["n50", "n100"])
@pytest.mark.parametrize("degree", [1, 2], ids=["degree1", "degree2"])
def test_sd_efficient_set_sp500_daily_returns_252(
    benchmark: Any,
    column_count: int,
    degree: int,
) -> None:
    returns = _load_daily_returns(row_count=252, column_count=column_count)

    result = benchmark(sd_efficient_set, returns, degree)

    assert all(0 <= index < column_count for index in result)


@pytest.mark.benchmark
@pytest.mark.parametrize("column_count", [50, 100], ids=["n50", "n100"])
@pytest.mark.parametrize("degree", [1, 2], ids=["degree1", "degree2"])
def test_nns_sd_cluster_sp500_daily_returns_252(
    benchmark: Any,
    column_count: int,
    degree: int,
) -> None:
    returns = _load_daily_returns(row_count=252, column_count=column_count)

    result = benchmark(nns_sd_cluster, returns, degree=degree, min_cluster=1)

    clusters = result["Clusters"]
    assert isinstance(clusters, dict)
    members = [name for cluster in clusters.values() for name in cluster]
    assert len(members) == column_count
    assert len(set(members)) == column_count


def _load_daily_returns(*, row_count: int, column_count: int) -> np.ndarray:
    with _FIXTURE.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
    if len(header) - 1 < column_count:
        raise AssertionError(
            f"{_FIXTURE} has {len(header) - 1} return columns, expected at least {column_count}.",
        )

    data = np.loadtxt(
        _FIXTURE,
        delimiter=",",
        skiprows=1,
        max_rows=row_count,
        usecols=range(1, column_count + 1),
        dtype=np.float64,
    )
    if data.shape != (row_count, column_count):
        expected_shape = (row_count, column_count)
        raise AssertionError(f"{_FIXTURE} has shape {data.shape}, expected {expected_shape}.")
    return data
