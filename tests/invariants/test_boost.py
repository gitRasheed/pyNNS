from __future__ import annotations

import numpy as np
import pytest

from pynns import nns_boost


def test_nns_boost_shapes_and_feature_weights() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x)

    result = nns_boost(variable, y, variable[:6], cv_size=0.25, feature_importance=False)

    assert result["results"].shape == (6,)
    assert result["pred.int"] is None
    assert np.sum(result["feature.weights"]) == pytest.approx(1.0)
    assert result["feature.frequency"].shape == result["feature.weights"].shape
    assert np.all(np.isfinite(result["results"]))


def test_nns_boost_class_shapes_and_codes() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x < -0.5, 1.0, np.where(x > 0.75, 3.0, 2.0))

    result = nns_boost(
        variable,
        y,
        variable[:6],
        cv_size=0.25,
        depth=1,
        type="class",
        feature_importance=False,
    )

    assert result["results"].shape == (6,)
    assert np.all(np.isin(result["results"], np.unique(y)))
    assert result["pred.int"] is None


@pytest.mark.parametrize("path", ["balance", "ts", "interval"])
def test_nns_boost_deferred_paths_raise(path: str) -> None:
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack((x, np.sin(x)))
    y = x + np.sin(x)

    with pytest.raises(NotImplementedError):
        if path == "balance":
            nns_boost(variable, y, balance=True)
        elif path == "ts":
            nns_boost(variable, y, ts_test=4)
        else:
            nns_boost(variable, y, pred_int=0.95)


def test_nns_boost_class_pred_int_raises() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack((x, np.sin(x)))
    y = np.where(x > 0.0, 2.0, 1.0)

    with pytest.raises(NotImplementedError, match="classification"):
        nns_boost(variable, y, type="class", pred_int=0.95)


def test_nns_boost_rejects_unported_stochastic_epoch_path() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack([np.sin((idx + 1) * x) for idx in range(11)])
    y = x + np.sin(x)

    with pytest.raises(
        NotImplementedError,
        match="n_features > 10 requires R's stochastic epoch keeper loop",
    ):
        nns_boost(variable, y, variable[:3], cv_size=0.25, feature_importance=False)
