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

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


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

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
@pytest.mark.parametrize(
    ("method", "ts_test"),
    [([1], 5), ([1], 10), ([2], 5), ([2], 10), ([1, 2], 10)],
)
def test_nns_stack_ts_test_matches_r(method: list[int], ts_test: int) -> None:
    x = np.linspace(-2.0, 2.0, 40)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)
    point = variable[:5]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=method,
        order=None,
        stack=True,
        dim_red_method="cor",
        ts_test=ts_test,
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=method,
        stack=True,
        dim_red_method="cor",
        ts_test=ts_test,
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
def test_nns_stack_var_like_ts_test_matches_r() -> None:
    h = 5
    x = np.linspace(-2.0, 2.0, 40)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)
    point = variable[-h:]
    ts_test = max(2 * h, int(0.2 * y.size))

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=[1, 2],
        order=None,
        stack=True,
        dim_red_method="cor",
        ts_test=ts_test,
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        stack=True,
        dim_red_method="cor",
        ts_test=ts_test,
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
@pytest.mark.parametrize("method", [[1], [2], [1, 2]])
def test_nns_stack_pred_int_matches_r(method: list[int]) -> None:
    x = np.linspace(-2.0, 2.0, 40)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)
    point = variable[:5]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=method,
        order=None,
        stack=True,
        dim_red_method="cor",
        pred_int=0.95,
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=method,
        stack=True,
        dim_red_method="cor",
        pred_int=0.95,
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
@pytest.mark.parametrize("method", [[1], [2], [1, 2]])
def test_nns_stack_binary_class_matches_r(method: list[int]) -> None:
    x = np.linspace(-2.0, 2.0, 36)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x + np.sin(x) > 0.0, 2.0, 1.0)
    point = variable[::9]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=method,
        order=None,
        stack=True,
        dim_red_method="cor",
        type="class",
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=method,
        stack=True,
        dim_red_method="cor",
        type="class",
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
@pytest.mark.parametrize("method", [[1], [2], [1, 2]])
def test_nns_stack_multiclass_matches_r(method: list[int]) -> None:
    x = np.linspace(-2.0, 2.0, 36)
    variable = np.column_stack((x, x**2, np.sin(x)))
    y = np.where(x < -0.5, 1.0, np.where(x > 0.75, 3.0, 2.0))
    point = variable[[0, 7, 18, 31]]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=method,
        order=1,
        stack=True,
        dim_red_method="cor",
        type="class",
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=method,
        order=1,
        stack=True,
        dim_red_method="cor",
        type="class",
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
def test_nns_stack_factor_like_class_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    labels = np.where(x < -0.5, "A", np.where(x > 0.75, "C", "B"))
    point = variable[::10]

    expected = nns_stack_numeric(
        variable.tolist(),
        labels.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=[1, 2],
        order=1,
        stack=True,
        dim_red_method="cor",
        type="class",
        class_levels=["A", "B", "C"],
    )
    actual = nns_stack(
        variable,
        labels,
        point,
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        order=1,
        stack=True,
        dim_red_method="cor",
        type="class",
        class_levels=["A", "B", "C"],
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


def test_nns_stack_raw_character_class_raises() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack((x, np.sin(x)))
    labels = np.where(x > 0.0, "B", "A")

    with pytest.raises(ValueError, match="class_levels"):
        nns_stack(variable, labels, variable[:3], type="class", cv_size=0.25, folds=1)


def _assert_stack_matches(
    actual: dict[str, Any],
    expected: Any,
    *,
    exact_probability_threshold: bool = True,
) -> None:
    assert isinstance(expected, dict)
    assert set(actual) == set(expected)
    for key in actual:
        if key == "probability.threshold" and not exact_probability_threshold:
            assert np.isfinite(float(actual[key]))
            assert 0.0 <= float(actual[key]) <= 1.0
            assert np.isfinite(float(_numeric(expected[key])))
            continue
        if actual[key] is None:
            assert expected[key] is None
        elif isinstance(actual[key], dict):
            assert isinstance(expected[key], dict)
            assert set(actual[key]) == set(expected[key])
            for column, values in actual[key].items():
                np.testing.assert_allclose(
                    np.asarray(values, dtype=np.float64),
                    _numeric(expected[key][column]),
                    atol=COMPOUND,
                )
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
