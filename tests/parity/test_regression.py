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
CASES: list[tuple[int | str | None, str, np.ndarray | None]] = [
    (None, "off", None),
    (1, "mean", None),
    (2, "median", np.array([-3.0, -1.0, 0.25, 3.0])),
    ("max", "off", np.array([-3.0, 0.0, 3.0])),
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
def test_nns_reg_rejects_deferred_matrix_path() -> None:
    x = np.column_stack((np.arange(10.0), np.arange(10.0) ** 2))
    y = np.arange(10.0)

    with pytest.raises(NotImplementedError, match=r"NNS\.M\.reg"):
        nns_reg(x, y)


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


def _assert_reg_matches(actual: dict[str, Any], expected: Any) -> None:
    assert isinstance(expected, dict)
    assert set(actual) == set(expected)
    np.testing.assert_allclose(actual["R2"], _array(expected["R2"]), atol=COMPOUND)
    np.testing.assert_allclose(actual["SE"], _array(expected["SE"]), atol=COMPOUND)
    np.testing.assert_allclose(actual["Point.est"], _array(expected["Point.est"]), atol=COMPOUND)

    for key in ("derivative", "regression.points", "Fitted.xy"):
        assert isinstance(expected[key], dict)
        assert isinstance(actual[key], dict)
        assert set(actual[key]) == set(expected[key])
        for column in actual[key]:
            if column == "NNS.ID":
                np.testing.assert_array_equal(
                    actual[key][column].astype(str),
                    _strings(expected[key][column]),
                )
            else:
                np.testing.assert_allclose(
                    actual[key][column],
                    _array(expected[key][column]),
                    atol=COMPOUND,
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
