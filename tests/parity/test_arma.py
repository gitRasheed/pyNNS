from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _r import nns
from _tolerances import COMPOUND

from pynns import nns_arma


@pytest.mark.parity
@pytest.mark.parametrize(
    ("name", "variable", "h", "seasonal_factor", "method", "training_set", "best_periods"),
    [
        ("known-linear", np.arange(1, 21, dtype=np.float64), 5, 4, "lin", None, 1),
        ("short-nonlin", np.arange(1, 7, dtype=np.float64), 3, 2, "nonlin", None, 1),
        ("multi-lag-linear", np.arange(1, 31, dtype=np.float64), 5, [3, 4], "lin", None, 1),
        (
            "explicit-nonlin",
            np.sin(np.arange(1, 41, dtype=np.float64) / 3.0) + 2.0,
            5,
            4,
            "nonlin",
            None,
            1,
        ),
        (
            "explicit-both",
            np.sin(np.arange(1, 41, dtype=np.float64) / 3.0) + 2.0,
            5,
            4,
            "both",
            None,
            1,
        ),
        ("means", np.arange(1, 21, dtype=np.float64), 5, 4, "means", None, 1),
        (
            "auto-seasonal",
            np.sin(np.arange(1, 60, dtype=np.float64) / 3.0)
            + 0.1 * np.arange(1, 60, dtype=np.float64),
            5,
            True,
            "nonlin",
            None,
            1,
        ),
        (
            "all-seasonal-best2",
            np.sin(np.arange(1, 60, dtype=np.float64) / 3.0)
            + 0.1 * np.arange(1, 60, dtype=np.float64),
            5,
            False,
            "nonlin",
            None,
            2,
        ),
        (
            "training-set",
            np.sin(np.arange(1, 50, dtype=np.float64) / 3.0) + 2.0,
            5,
            4,
            "lin",
            30,
            1,
        ),
        ("constant-auto", np.full(20, 5.0), 3, True, "nonlin", None, 1),
        ("constant-explicit", np.full(20, 5.0), 3, 4, "lin", None, 1),
        ("negative-both", np.sin(np.arange(1, 31, dtype=np.float64)), 3, 5, "both", None, 1),
    ],
)
def test_nns_arma_matches_r(
    name: str,
    variable: np.ndarray,
    h: int,
    seasonal_factor: Any,
    method: str,
    training_set: int | None,
    best_periods: int | None,
) -> None:
    del name
    expected = nns(
        "NNS.ARMA",
        variable.tolist(),
        h,
        training_set,
        seasonal_factor,
        None,
        best_periods,
        None,
        True,
        False,
        method,
        False,
        False,
        False,
        False,
        None,
    )

    actual = nns_arma(
        variable,
        h=h,
        training_set=training_set,
        seasonal_factor=seasonal_factor,
        method=method,
        best_periods=best_periods,
    )

    np.testing.assert_allclose(actual, _array(expected), atol=COMPOUND, equal_nan=True)


@pytest.mark.parity
def test_nns_arma_dynamic_means_matches_r() -> None:
    variable = np.sin(np.arange(1, 40, dtype=np.float64) / 3.0) + 2.0

    expected = nns(
        "NNS.ARMA",
        variable.tolist(),
        3,
        None,
        False,
        None,
        1,
        None,
        True,
        False,
        "means",
        True,
        False,
        False,
        False,
        None,
    )
    actual = nns_arma(variable, h=3, seasonal_factor=False, method="means", dynamic=True)

    np.testing.assert_allclose(actual, _array(expected), atol=COMPOUND, equal_nan=True)


def test_nns_arma_known_linear_check() -> None:
    result = nns_arma(np.arange(1, 21, dtype=np.float64), h=5, seasonal_factor=4, method="lin")

    np.testing.assert_allclose(result, np.array([21, 22, 23, 24, 25], dtype=np.float64))


def _array(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64)
    if isinstance(value, list):
        return np.asarray([np.nan if item == "NaN" else item for item in value], dtype=np.float64)
    raise AssertionError(f"Unexpected R value type: {type(value)!r}")
