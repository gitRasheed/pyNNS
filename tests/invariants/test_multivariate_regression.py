from __future__ import annotations

import numpy as np
import pytest

from pynns import nns_m_reg


def test_nns_m_reg_shapes_and_bounds() -> None:
    x1 = np.linspace(-2.0, 2.0, 100)
    x = np.column_stack((x1, np.sin(x1), np.cos(x1)))
    y = x1 + np.sin(x1)
    points = np.array([[0.0, 0.0, 1.0], [3.0, 0.0, 1.0]])

    result = nns_m_reg(x, y, order=2, n_best=1, point_est=points)

    assert np.isnan(result["R2"]) or 0.0 <= result["R2"] <= 1.0
    assert result["Fitted.xy"]["y"].shape == y.shape
    assert result["Fitted.xy"]["y.hat"].shape == y.shape
    assert result["Fitted.xy"]["NNS.ID"].shape == y.shape
    assert result["Point.est"].shape == (2,)
    assert result["RPM"]["y.hat"].size <= y.size
    assert all("." in item for item in result["Fitted.xy"]["NNS.ID"].astype(str))


def test_nns_m_reg_point_only_returns_point_est_and_rpm() -> None:
    x1 = np.linspace(-2.0, 2.0, 50)
    x = np.column_stack((x1, np.sin(x1)))
    y = x1 + np.sin(x1)

    result = nns_m_reg(x, y, order=1, n_best=1, point_est=np.array([[0.0, 0.0]]), point_only=True)

    assert set(result) == {"Point.est", "RPM"}
    assert result["Point.est"].shape == (1,)
    assert result["RPM"]["y.hat"].size <= y.size


def test_nns_m_reg_order_max_is_perfect_fit() -> None:
    x1 = np.linspace(-2.0, 2.0, 30)
    x = np.column_stack((x1, np.sin(x1)))
    y = x1 + np.sin(x1)

    result = nns_m_reg(x, y, order="max")

    np.testing.assert_allclose(result["Fitted.xy"]["y.hat"], y, atol=1e-12)
    assert result["R2"] == pytest.approx(1.0)


@pytest.mark.parametrize("kwargs", [{"type": "class"}, {"confidence_interval": 0.95}])
def test_nns_m_reg_deferred_paths_raise(kwargs: dict[str, object]) -> None:
    x1 = np.linspace(-2.0, 2.0, 20)
    x = np.column_stack((x1, np.sin(x1)))
    y = x1 + np.sin(x1)

    with pytest.raises(NotImplementedError):
        nns_m_reg(x, y, **kwargs)
