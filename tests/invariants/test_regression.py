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


def test_nns_reg_confidence_interval_shapes_and_row_drop() -> None:
    x = np.linspace(-2.0, 2.0, 50)
    y = np.sin(x)
    point_est = np.array([-3.0, -1.0, 0.0, 2.5])

    result = nns_reg(x, y, order=1, point_est=point_est, confidence_interval=0.95)

    assert result["Fitted.xy"]["conf.int.pos"].shape == x.shape
    assert result["Fitted.xy"]["conf.int.neg"].shape == x.shape
    assert result["Point.est"].shape == point_est.shape
    assert result["pred.int"] is not None
    assert set(result["pred.int"]) == {"pred.int.neg", "pred.int.pos"}
    assert result["pred.int"]["pred.int.neg"].shape == (3,)
    assert result["pred.int"]["pred.int.pos"].shape == (3,)


def test_nns_reg_confidence_interval_none_output_unchanged() -> None:
    x = np.linspace(-2.0, 2.0, 50)
    y = np.sin(x)

    result = nns_reg(x, y, order=1)

    assert "conf.int.pos" not in result["Fitted.xy"]
    assert "conf.int.neg" not in result["Fitted.xy"]
    assert result["pred.int"] is None


@pytest.mark.parametrize(
    "path",
    ["smooth", "smooth_confidence", "point_only", "dimred_tau_ts", "matrix_point_est"],
)
def test_nns_reg_deferred_paths_raise(path: str) -> None:
    x = np.linspace(-2.0, 2.0, 20)
    y = np.sin(x)

    with pytest.raises(NotImplementedError):
        if path == "smooth":
            nns_reg(x, y, smooth=True)
        elif path == "smooth_confidence":
            nns_reg(x, y, smooth=True, confidence_interval=0.95)
        elif path == "point_only":
            nns_reg(x, y, point_only=True)
        elif path == "dimred_tau_ts":
            matrix = np.column_stack((x, np.cos(x)))
            nns_reg(matrix, y, dim_red_method="NNS.caus", tau="ts")
        elif path == "matrix_point_est":
            nns_reg(x, y, point_est=np.array([[0.0], [1.0]]))
        else:
            nns_reg(x, y, multivariate_call=True)


def test_nns_reg_classification_outputs_numeric_codes() -> None:
    x = np.linspace(0.0, 5.0, 6)
    y = np.array([1, 1, 1, 2, 2, 2], dtype=np.float64)

    result = nns_reg(x, y, type="CLASS", point_est=np.array([1.5, 4.5]))

    assert result["Prediction.Accuracy"] is not None
    assert set(result["Fitted.xy"]["y.hat"]).issubset(set(y))
    assert set(result["Point.est"]).issubset(set(y))


def test_nns_reg_raw_string_class_labels_raise() -> None:
    x = np.linspace(0.0, 5.0, 6)
    y = np.array(["A", "A", "A", "B", "B", "B"])

    with pytest.raises(ValueError, match="class_levels"):
        nns_reg(x, y, type="class")
