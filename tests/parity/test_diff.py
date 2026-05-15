from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _r import dy_d_scalar, dy_dx_numeric, dy_dx_overall, nns_diff_custom
from _tolerances import EXACT

from pynns import dy_d, dy_dx, nns_diff

DIFF_PARITY = 1e-5
DY_D_PARITY = 1e-3


@pytest.mark.parity
@pytest.mark.parametrize(
    ("name", "func", "point"),
    [
        ("square", lambda x: x * x, 2.0),
        ("sin", np.sin, 1.0),
        ("exp", np.exp, 0.5),
        ("constant", lambda x: 5.0, 2.0),
        ("identity", lambda x: x, 3.0),
    ],
)
def test_nns_diff_derivative_matches_r(
    name: str,
    func: Any,
    point: float,
) -> None:
    expected = _r_nns_diff(name, point)
    actual = nns_diff(func, point)

    np.testing.assert_allclose(actual["DERIVATIVE"], expected["DERIVATIVE"], atol=DIFF_PARITY)
    np.testing.assert_allclose(
        actual["Value of f(x) at point"],
        expected["Value of f(x) at point"],
        atol=EXACT,
    )


@pytest.mark.parity
def test_dy_dx_overall_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 24)
    y = x + np.sin(x)

    expected = float(np.asarray(dy_dx_overall(x.tolist(), y.tolist()), dtype=np.float64))
    actual = dy_dx(x, y, eval_point="overall")

    assert actual == pytest.approx(expected, abs=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize("eval_point", [[0.0], [-1.0, 0.0, 1.0]])
def test_dy_dx_numeric_eval_points_match_r(eval_point: list[float]) -> None:
    x = np.linspace(-2.0, 2.0, 24)
    y = x + np.sin(x)

    expected = _dict_array(dy_dx_numeric(x.tolist(), y.tolist(), eval_point))
    actual = dy_dx(x, y, eval_point=np.asarray(eval_point, dtype=np.float64))
    assert isinstance(actual, dict)

    assert list(actual) == list(expected)
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=5e-3, equal_nan=True)


@pytest.mark.parity
@pytest.mark.parametrize(
    ("wrt",),
    [
        (1,),
        (2,),
    ],
)
def test_dy_d_mean_wrt_match_r(wrt: int) -> None:
    x = np.column_stack(
        (np.array([-2, -1, 0, 1, 2], dtype=float), np.array([1, 3, 5, 7, 9], dtype=float))
    )
    y = 2 * x[:, 0] + 3 * x[:, 1]

    expected = _dict_array(dy_d_scalar(x.tolist(), y.tolist(), wrt, "mean"))
    actual = dy_d(x, y, wrt=wrt, eval_points="mean")

    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=DY_D_PARITY, equal_nan=True)


@pytest.mark.parity
def test_dy_d_nonlinear_wrt1_mean_matches_r() -> None:
    x = np.column_stack(
        (np.array([-2, -1, 0, 1, 2], dtype=float), np.array([1, 3, 5, 7, 9], dtype=float))
    )
    y = x[:, 0] ** 2 + np.sin(x[:, 1])

    expected = _dict_array(dy_d_scalar(x.tolist(), y.tolist(), 1, "mean"))
    actual = dy_d(x, y, wrt=1, eval_points="mean")

    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=DY_D_PARITY, equal_nan=True)


def _r_nns_diff(name: str, point: float) -> dict[str, float]:
    result = nns_diff_custom(name, point)
    assert isinstance(result, dict)
    return {
        key: float(np.asarray(value).reshape(-1)[0])
        for key, value in result.items()
        if isinstance(value, np.ndarray)
    }


def _dict_array(value: object) -> dict[str, np.ndarray]:
    if not isinstance(value, dict):
        raise AssertionError(f"Expected dictionary, got {type(value)!r}")
    return {key: np.asarray(item, dtype=np.float64) for key, item in value.items()}
