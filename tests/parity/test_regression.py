from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from _r import nns
from _tolerances import COMPOUND

from pynns import nns_reg
from pynns.part import NoiseReduction
from pynns.regression import Order

SIZES = [50, 200, 1000]
RELATIONSHIPS = ["linear", "quadratic", "cubic", "sin", "random"]
MODE_RELATIONSHIPS = ["linear", "quadratic", "sin", "random"]
CASES: list[tuple[int | str | None, str, np.ndarray | None]] = [
    (None, "off", None),
    (1, "mean", None),
    (2, "median", np.array([-3.0, -1.0, 0.25, 3.0])),
    ("max", "off", np.array([-3.0, 0.0, 3.0])),
]
MODE_ORDERS: list[int | None] = [None, 1, 2, 3, 5]
MEAN_POINT_EST_CASES = [
    np.array([-3.0]),
    np.array([3.0]),
    np.array([-3.0, -1.0, 0.0, 2.5]),
]
DIM_RED_METHODS: list[str | list[float]] = [
    "cor",
    "NNS.dep",
    "NNS.caus",
    "all",
    "equal",
    [1.0, 0.5, 0.25],
]


@pytest.mark.parity
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("relationship", RELATIONSHIPS)
@pytest.mark.parametrize(("order", "noise", "point_est"), CASES)
def test_nns_reg_univariate_matches_r(
    rng: np.random.Generator,
    size: int,
    relationship: str,
    order: int | str | None,
    noise: str,
    point_est: np.ndarray | None,
) -> None:
    x, y = _relationship(relationship, size, rng)

    expected = _r_nns_reg(x, y, order=order, noise=noise, point_est=point_est)
    actual = nns_reg(
        x,
        y,
        order=cast(Order, order),
        noise_reduction=cast(NoiseReduction, noise),
        point_est=point_est,
    )

    _assert_reg_matches(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("relationship", MODE_RELATIONSHIPS)
@pytest.mark.parametrize("order", MODE_ORDERS)
def test_nns_reg_mode_noise_reduction_matches_r(
    rng: np.random.Generator,
    size: int,
    relationship: str,
    order: int | None,
) -> None:
    x, y = _relationship(relationship, size, rng)

    expected = _r_nns_reg(x, y, order=order, noise="mode", point_est=None)
    actual = nns_reg(x, y, order=order, noise_reduction="mode")

    _assert_reg_matches(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("relationship", MODE_RELATIONSHIPS)
@pytest.mark.parametrize("order", MODE_ORDERS)
def test_nns_reg_mode_class_noise_reduction_matches_r(
    rng: np.random.Generator,
    size: int,
    relationship: str,
    order: int | None,
) -> None:
    x, y = _relationship(relationship, size, rng)

    expected = _r_nns_reg(x, y, order=order, noise="mode_class", point_est=None)
    actual = nns_reg(x, y, order=order, noise_reduction="mode_class")

    _assert_reg_matches(
        actual,
        expected,
        skip_standard_errors=order is None,
    )


@pytest.mark.parity
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("point_est", MEAN_POINT_EST_CASES)
def test_nns_reg_mean_out_of_range_point_est_matches_r(
    size: int,
    point_est: np.ndarray,
) -> None:
    x = np.linspace(-2.0, 2.0, size)
    y = np.sin(x)

    expected = _r_nns_reg(x, y, order=1, noise="mean", point_est=point_est)
    actual = nns_reg(x, y, order=1, noise_reduction="mean", point_est=point_est)

    _assert_reg_matches(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize("method", DIM_RED_METHODS)
def test_nns_reg_dim_red_matches_r(method: str | list[float]) -> None:
    x1 = np.linspace(-2.0, 2.0, 50)
    x = np.column_stack((x1, np.sin(x1), np.cos(x1)))
    y = x[:, 0] + x[:, 1] + 0.25 * x[:, 2]
    point_est = np.array([[0.0, 0.0, 1.0], [3.0, 0.0, 1.0]])

    expected = _r_nns_reg_dimred(
        x,
        y,
        order=2,
        dim_red_method=method,
        threshold=0.0,
        point_est=point_est,
        point_only=False,
    )
    actual = nns_reg(
        x,
        y,
        order=2,
        dim_red_method=method,
        point_est=point_est,
    )

    tolerance = 5e-2 if method == "NNS.caus" else COMPOUND
    _assert_reg_matches(actual, expected, check_dimred=True, atol=tolerance)


@pytest.mark.parity
def test_nns_reg_dim_red_point_only_matches_r() -> None:
    x1 = np.linspace(-2.0, 2.0, 50)
    x = np.column_stack((x1, np.sin(x1), np.cos(x1)))
    y = x[:, 0] + x[:, 1] + 0.25 * x[:, 2]
    point_est = np.array([[0.0, 0.0, 1.0], [3.0, 0.0, 1.0]])

    expected = _r_nns_reg_dimred(
        x,
        y,
        order=None,
        dim_red_method="equal",
        threshold=0.0,
        point_est=point_est,
        point_only=True,
    )
    actual = nns_reg(
        x,
        y,
        dim_red_method="equal",
        point_est=point_est,
        point_only=True,
    )

    _assert_reg_matches(actual, expected, check_dimred=True)


def _r_nns_reg(
    x: np.ndarray,
    y: np.ndarray,
    *,
    order: int | str | None,
    noise: str,
    point_est: np.ndarray | None,
) -> Any:
    point_arg: list[float] | None = None if point_est is None else point_est.tolist()
    return nns(
        "NNS.reg",
        x.tolist(),
        y.tolist(),
        False,
        order,
        None,
        None,
        None,
        point_arg,
        "top",
        True,
        False,
        False,
        False,
        None,
        0,
        None,
        False,
        noise,
        "L2",
        None,
        False,
        False,
    )


def _r_nns_reg_dimred(
    x: np.ndarray,
    y: np.ndarray,
    *,
    order: int | str | None,
    dim_red_method: str | list[float],
    threshold: float,
    point_est: np.ndarray | None,
    point_only: bool,
) -> Any:
    return nns(
        "NNS.reg",
        x.tolist(),
        y.tolist(),
        False,
        order,
        dim_red_method,
        None,
        None,
        None if point_est is None else point_est.tolist(),
        "top",
        True,
        False,
        False,
        False,
        None,
        threshold,
        None,
        False,
        "off",
        "L2",
        1,
        point_only,
        False,
    )


def _assert_reg_matches(
    actual: dict[str, Any],
    expected: Any,
    *,
    skip_standard_errors: bool = False,
    check_dimred: bool = False,
    atol: float = COMPOUND,
) -> None:
    assert isinstance(expected, dict)
    assert set(actual) == set(expected)
    np.testing.assert_allclose(actual["R2"], _array(expected["R2"]), atol=atol)
    np.testing.assert_allclose(actual["SE"], _array(expected["SE"]), atol=atol)
    np.testing.assert_allclose(actual["Point.est"], _array(expected["Point.est"]), atol=atol)
    if check_dimred:
        assert isinstance(expected["equation"], dict)
        assert isinstance(actual["equation"], dict)
        np.testing.assert_array_equal(
            actual["equation"]["Variable"].astype(str),
            _strings(expected["equation"]["Variable"]),
        )
        np.testing.assert_allclose(
            actual["equation"]["Coefficient"],
            _array(expected["equation"]["Coefficient"]),
            atol=atol,
        )
        assert isinstance(expected["x.star"], dict)
        assert isinstance(actual["x.star"], dict)
        np.testing.assert_allclose(
            actual["x.star"]["x"],
            _array(expected["x.star"]["x"]),
            atol=atol,
        )

    for key in ("derivative", "regression.points", "Fitted.xy"):
        assert isinstance(expected[key], dict)
        assert isinstance(actual[key], dict)
        assert set(actual[key]) == set(expected[key])
        for column in actual[key]:
            if skip_standard_errors and key == "Fitted.xy" and column == "standard.errors":
                continue
            if column == "NNS.ID":
                np.testing.assert_array_equal(
                    actual[key][column].astype(str),
                    _strings(expected[key][column]),
                )
            else:
                np.testing.assert_allclose(
                    actual[key][column],
                    _array(expected[key][column]),
                    atol=atol,
                )


def _array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _strings(value: object) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value, dtype=str)
    return np.asarray(value, dtype=str)


def _relationship(
    relationship: str,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-2.0, 2.0, size)
    if relationship == "linear":
        return x, 1.5 + 0.7 * x + 0.02 * np.sin(np.arange(size))
    if relationship == "quadratic":
        return x, x * x
    if relationship == "cubic":
        return x, x**3
    if relationship == "sin":
        return x, np.sin(x)
    return x, rng.normal(size=size)
