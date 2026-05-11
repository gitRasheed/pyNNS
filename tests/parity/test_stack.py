from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _r import nns_stack_numeric
from _tolerances import COMPOUND

from pynns import nns_stack


@pytest.mark.parity
@pytest.mark.parametrize("method", [[1], [2], [1, 2]])
@pytest.mark.parametrize("stack", [True, False])
def test_nns_stack_numeric_matches_r(method: list[int], stack: bool) -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)
    point = variable[:5]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=2,
        method=method,
        order=None,
        stack=stack,
        dim_red_method="cor",
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=2,
        method=method,
        stack=stack,
        dim_red_method="cor",
    )

    _assert_stack_matches(actual, expected)


@pytest.mark.parity
def test_nns_stack_equal_dim_red_matches_r() -> None:
    x = np.linspace(-1.5, 1.5, 36)
    variable = np.column_stack((x, x**2, np.sin(x)))
    y = 0.5 * x + x**2 - 0.25 * np.sin(x)
    point = variable[::9]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=2,
        method=[2],
        order=2,
        stack=False,
        dim_red_method="equal",
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=2,
        method=2,
        order=2,
        stack=False,
        dim_red_method="equal",
    )

    _assert_stack_matches(actual, expected)


def _assert_stack_matches(actual: dict[str, Any], expected: Any) -> None:
    assert isinstance(expected, dict)
    assert set(actual) == set(expected)
    for key in actual:
        if actual[key] is None:
            assert expected[key] is None
        else:
            np.testing.assert_allclose(
                np.asarray(actual[key], dtype=np.float64),
                _numeric(expected[key]),
                atol=COMPOUND,
            )


def _numeric(value: object) -> np.ndarray:
    if isinstance(value, str):
        if value == "NA":
            return np.asarray(np.nan, dtype=np.float64)
        if value == "Inf":
            return np.asarray(np.inf, dtype=np.float64)
        if value == "-Inf":
            return np.asarray(-np.inf, dtype=np.float64)
    return np.asarray(value, dtype=np.float64)
