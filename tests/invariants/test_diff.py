from __future__ import annotations

import numpy as np
import pytest

from pynns import dy_d, dy_dx, nns_diff


def test_nns_diff_constant_derivative_is_zero() -> None:
    result = nns_diff(lambda x: 12.0, 3.0)

    assert result["DERIVATIVE"] == pytest.approx(0.0)


def test_nns_diff_identity_derivative_is_one() -> None:
    result = nns_diff(lambda x: x, -2.0)

    assert result["DERIVATIVE"] == pytest.approx(1.0)


def test_nns_diff_smooth_function_derivative_has_bounded_error() -> None:
    point = 1.25
    result = nns_diff(np.sin, point)

    assert result["DERIVATIVE"] == pytest.approx(np.cos(point), abs=1e-6)


def test_dy_dx_numeric_eval_point_returns_derivative_table() -> None:
    x = np.linspace(-2.0, 2.0, 24)
    y = x + np.sin(x)

    result = dy_dx(x, y, eval_point=np.array([-1.0, 0.0, 1.0]))

    assert isinstance(result, dict)
    assert list(result) == ["eval.point", "first.derivative", "second.derivative"]
    assert all(value.shape == (3,) for value in result.values())
    assert np.all(np.isfinite(result["first.derivative"]))


def test_dy_d_remains_deferred() -> None:
    x = np.column_stack((np.linspace(-2.0, 2.0, 24), np.linspace(0.0, 1.0, 24)))
    y = x[:, 0] + x[:, 1]

    with pytest.raises(NotImplementedError, match="dy_d finite-difference derivatives"):
        dy_d(x, y, wrt=1)
