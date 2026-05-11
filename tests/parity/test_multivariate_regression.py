from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from _r import nns
from _tolerances import COMPOUND

from pynns import nns_m_reg, nns_reg
from pynns.part import NoiseReduction
from pynns.regression import Order


@pytest.mark.parity
@pytest.mark.parametrize("order", [None, 1, 2])
def test_nns_reg_multivariate_call_matches_r(order: int | None) -> None:
    x = np.linspace(-2.0, 2.0, 50)
    y = x * x + 0.1 * np.sin(np.arange(x.size))

    expected = nns(
        "NNS.reg",
        x.tolist(),
        y.tolist(),
        False,
        order,
        None,
        None,
        None,
        None,
        "top",
        True,
        False,
        False,
        True,
        None,
        0,
        None,
        False,
        "off",
        "L2",
        None,
        False,
        True,
    )
    actual = nns_reg(x, y, order=order, multivariate_call=True)
    expected_dict = cast(dict[str, Any], expected)

    assert set(actual) == set(expected_dict)
    np.testing.assert_allclose(actual["x"], _array(expected_dict["x"]), atol=COMPOUND)
    np.testing.assert_allclose(actual["y"], _array(expected_dict["y"]), atol=COMPOUND)


MREG_CASES = [
    (50, 2, "linear", None, None, None, False, "off"),
    (50, 3, "nonlinear", 1, 1, np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]), False, "off"),
    (200, 3, "mixed", 2, 2, None, False, "mean"),
    (200, 5, "linear", "max", None, None, False, "median"),
    (50, 2, "nonlinear", 1, 1, np.array([[0.0, 0.0], [3.0, 0.0]]), True, "off"),
]


@pytest.mark.parity
@pytest.mark.parametrize(
    ("size", "n_cols", "relationship", "order", "n_best", "point_est", "point_only", "noise"),
    MREG_CASES,
)
def test_nns_m_reg_matches_r(
    rng: np.random.Generator,
    size: int,
    n_cols: int,
    relationship: str,
    order: int | str | None,
    n_best: int | str | None,
    point_est: np.ndarray | None,
    point_only: bool,
    noise: str,
) -> None:
    x, y = _dataset(size, n_cols, relationship, rng)
    if point_est is not None and point_est.shape[1] != n_cols:
        point_est = np.pad(
            point_est[:, : min(point_est.shape[1], n_cols)],
            ((0, 0), (0, n_cols - point_est.shape[1])),
        )

    expected = _r_nns_m_reg(x, y, order, n_best, point_est, point_only, noise)
    actual = nns_m_reg(
        x,
        y,
        order=cast(Order, order),
        n_best=n_best,
        point_est=point_est,
        point_only=point_only,
        noise_reduction=cast(NoiseReduction, noise),
        ncores=1,
    )

    _assert_m_reg_matches(actual, expected)


def _r_nns_m_reg(
    x: np.ndarray,
    y: np.ndarray,
    order: int | str | None,
    n_best: int | str | None,
    point_est: np.ndarray | None,
    point_only: bool,
    noise: str,
) -> Any:
    return nns(
        "NNS.M.reg",
        x.tolist(),
        y.tolist(),
        False,
        order,
        n_best,
        None,
        None if point_est is None else point_est.tolist(),
        point_only,
        False,
        False,
        None,
        noise,
        "L2",
        False,
        False,
        1,
        None,
    )


def _assert_m_reg_matches(actual: dict[str, Any], expected: Any) -> None:
    assert isinstance(expected, dict)
    assert set(actual) == set(expected)
    for key in actual:
        if isinstance(actual[key], dict):
            assert isinstance(expected[key], dict)
            assert set(actual[key]) == set(expected[key])
            for column, values in actual[key].items():
                if column == "NNS.ID":
                    np.testing.assert_array_equal(
                        values.astype(str),
                        np.asarray(expected[key][column], dtype=str),
                    )
                else:
                    np.testing.assert_allclose(values, _array(expected[key][column]), atol=COMPOUND)
        elif actual[key] is None:
            assert _array(expected[key]).size == 0
        else:
            np.testing.assert_allclose(actual[key], _array(expected[key]), atol=COMPOUND)


def _dataset(
    size: int,
    n_cols: int,
    relationship: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.linspace(-2.0, 2.0, size)
    cols = [base]
    if n_cols > 1:
        cols.append(np.sin(base))
    for index in range(2, n_cols):
        cols.append(np.cos((index + 1) * base) + 0.01 * rng.normal(size=size))
    x = np.column_stack(cols)
    if relationship == "linear":
        beta = np.linspace(0.4, 1.0, n_cols)
        y = x @ beta + 0.01 * np.sin(np.arange(size))
    elif relationship == "nonlinear":
        y = x[:, 0] ** 2 + np.sin(x[:, 1])
    else:
        y = x[:, 0] + x[:, 1] ** 2 + 0.2 * x[:, -1]
    return x, y


def _array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)
