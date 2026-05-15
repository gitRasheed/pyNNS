from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from _r import nns

from pynns.var import _lag_mtx, _var_interpolate_and_extrapolate


def _to_matrix(result: object, names: list[str], key: str) -> np.ndarray:
    assert isinstance(result, dict)
    values = result[key]
    assert isinstance(values, dict)
    return np.column_stack([np.asarray(values[name], dtype=np.float64) for name in names])


def _json_safe_data(values: np.ndarray) -> list[list[float | None]]:
    return [
        [None if np.isnan(item) else float(item) for item in row]
        for row in values.tolist()
    ]


def test_lag_mtx_scalar_tau_matches_reference_blocks() -> None:
    x = np.column_stack(
        (
            np.array([1, 2, 3, 4, 5], dtype=np.float64),
            np.array([6, 7, 8, 9, 10], dtype=np.float64),
        )
    )
    actual, names = _lag_mtx(x, 2, names=["a", "b"])

    expected = np.array(
        [
            [3, 8, 2, 1, 7, 6],
            [4, 9, 3, 2, 8, 7],
            [5, 10, 4, 3, 9, 8],
        ],
        dtype=np.float64,
    )
    expected_names = ["a_tau_0", "b_tau_0", "a_tau_1", "a_tau_2", "b_tau_1", "b_tau_2"]

    np.testing.assert_allclose(actual, expected)
    assert names == expected_names


def test_lag_mtx_nested_tau_keeps_requested_lags_plus_tau_zero() -> None:
    x = np.column_stack(
        (
            np.array([1, 2, 3, 4, 5], dtype=np.float64),
            np.array([6, 7, 8, 9, 10], dtype=np.float64),
        )
    )
    actual, names = _lag_mtx(x, ([1, 2], [1]), names=["a", "b"])

    expected_names = ["a_tau_0", "b_tau_0", "a_tau_1", "a_tau_2", "b_tau_1"]
    expected = np.array(
        [
            [3, 8, 2, 1, 7],
            [4, 9, 3, 2, 8],
            [5, 10, 4, 3, 9],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(actual, expected)
    assert names == expected_names


def _expected_var_reference(
    variables: np.ndarray,
    h: int,
    tau: int | list[int] | list[list[int]],
) -> dict[str, object]:
    result = cast(dict[str, Any], nns("NNS.VAR", _json_safe_data(variables), h, tau))
    names = list(result["interpolated_and_extrapolated"].keys()) if h > 0 else list(result.keys())
    if h > 0:
        interpolated_and_extrapolated = _to_matrix(result, names, "interpolated_and_extrapolated")
        univariate = _to_matrix(result, names, "univariate")
    else:
        interpolated_and_extrapolated = np.column_stack(
            [np.asarray(result[name], dtype=np.float64) for name in names]
        )
        univariate = None

    expected: dict[str, object] = {
        "interpolated_and_extrapolated": interpolated_and_extrapolated,
        "names": names,
    }
    if h > 0:
        expected["univariate"] = univariate
    return expected


@pytest.mark.parametrize(
    ("name", "h"),
    [
        ("complete_finite", 3),
        ("interior_na", 3),
        ("trailing_na", 3),
        ("negative", 3),
    ],
)
def test_var_interpolate_and_extrapolate_matches_r(
    name: str,
    h: int,
) -> None:
    base = np.column_stack(
        (
            np.arange(-2.0, 18.0, 1.0, dtype=float),
            np.arange(1.0, 40.0, 2.0, dtype=float),
        )
    )
    if name == "interior_na":
        base = base.copy()
        base[4, 0] = np.nan
    elif name == "trailing_na":
        base = base.copy()
        base[19, 0] = np.nan
    elif name == "negative":
        base = -base

    expected_result = _expected_var_reference(base, h, 2)
    names = cast(list[str], expected_result["names"])
    actual_result = _var_interpolate_and_extrapolate(base, h, tau=2, names=names)
    actual_interpolated = cast(np.ndarray, actual_result["interpolated_and_extrapolated"])
    expected_interpolated = cast(
        np.ndarray,
        expected_result["interpolated_and_extrapolated"],
    )
    actual_univariate = cast(np.ndarray, actual_result["univariate"])
    expected_univariate = cast(np.ndarray, expected_result["univariate"])

    np.testing.assert_allclose(actual_interpolated, expected_interpolated, equal_nan=True)
    assert actual_result["names"] == expected_result["names"]

    assert "univariate" in actual_result
    assert "univariate" in expected_result
    assert isinstance(actual_result["univariate"], np.ndarray)
    assert isinstance(expected_result["univariate"], np.ndarray)
    np.testing.assert_allclose(actual_univariate, expected_univariate, equal_nan=True)


def test_var_interpolate_and_extrapolate_h0_matches_r() -> None:
    variables = np.column_stack(
        (
            np.arange(-2.0, 18.0, 1.0, dtype=float),
            np.arange(1.0, 40.0, 2.0, dtype=float),
        )
    )

    expected_result = _expected_var_reference(variables, 0, 2)
    actual_result = _var_interpolate_and_extrapolate(variables, 0, tau=2)
    actual_interpolated = cast(np.ndarray, actual_result["interpolated_and_extrapolated"])
    expected_interpolated = cast(
        np.ndarray,
        expected_result["interpolated_and_extrapolated"],
    )

    np.testing.assert_allclose(actual_interpolated, expected_interpolated, equal_nan=True)
    assert "univariate" not in actual_result
