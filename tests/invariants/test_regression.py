from __future__ import annotations

import numpy as np
import pytest

from pynns import nns_reg


def test_nns_reg_shapes_and_bounds() -> None:
    x = np.linspace(-2.0, 2.0, 100)
    y = np.sin(x)

    result = nns_reg(x, y, order=3, point_est=np.array([-3.0, 0.0, 3.0]))

    assert 0.0 <= result["R2"] <= 1.0
    assert result["SE"] >= 0.0
    assert result["Fitted.xy"]["x"].shape == x.shape
    assert result["Fitted.xy"]["y.hat"].shape == x.shape
    assert result["Point.est"].shape == (3,)
    assert result["derivative"]["Coefficient"].size == result["regression.points"]["x"].size - 1


def test_nns_reg_order_max_is_perfect_fit() -> None:
    x = np.linspace(-2.0, 2.0, 50)
    y = x**2

    result = nns_reg(x, y, order="max")

    np.testing.assert_allclose(result["Fitted.xy"]["y.hat"], y, atol=1e-12)
    assert result["R2"] == pytest.approx(1.0)
    assert result["SE"] == pytest.approx(0.0)


def test_nns_reg_increasing_order_does_not_reduce_r2_for_smooth_curve() -> None:
    x = np.linspace(-2.0, 2.0, 200)
    y = np.sin(x)

    r1 = nns_reg(x, y, order=1)["R2"]
    r2 = nns_reg(x, y, order=2)["R2"]
    r3 = nns_reg(x, y, order=3)["R2"]

    assert r2 >= r1 - 1e-12
    assert r3 >= r2 - 1e-12


def test_nns_reg_dim_red_shapes_and_equation() -> None:
    x1 = np.linspace(-2.0, 2.0, 80)
    x = np.column_stack((x1, np.sin(x1), np.cos(x1)))
    y = x[:, 0] + x[:, 1] + 0.25 * x[:, 2]
    point_est = np.array([[0.0, 0.0, 1.0], [3.0, 0.0, 1.0]])

    result = nns_reg(x, y, dim_red_method="equal", point_est=point_est, point_only=True)

    assert np.isnan(result["R2"]) or 0.0 <= result["R2"] <= 1.0
    assert result["x.star"]["x"].shape == y.shape
    assert result["equation"]["Variable"].shape == (x.shape[1] + 1,)
    assert result["equation"]["Coefficient"].shape == (x.shape[1] + 1,)
    assert result["Point.est"].shape == (2,)
    assert result["Fitted.xy"]["x"].shape == y.shape


@pytest.mark.parametrize(
    "path",
    ["smooth", "confidence", "class", "point_only"],
)
def test_nns_reg_deferred_paths_raise(path: str) -> None:
    x = np.linspace(-2.0, 2.0, 20)
    y = np.sin(x)

    with pytest.raises(NotImplementedError):
        if path == "smooth":
            nns_reg(x, y, smooth=True)
        elif path == "confidence":
            nns_reg(x, y, confidence_interval=0.95)
        elif path == "class":
            nns_reg(x, y, type="CLASS")
        elif path == "point_only":
            nns_reg(x, y, point_only=True)
        else:
            nns_reg(x, y, multivariate_call=True)
