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


@pytest.mark.parametrize("path", ["class", "balance", "ts", "interval"])
def test_nns_boost_deferred_paths_raise(path: str) -> None:
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack((x, np.sin(x)))
    y = x + np.sin(x)

    with pytest.raises(NotImplementedError):
        if path == "class":
            nns_boost(variable, y, type="class")
        elif path == "balance":
            nns_boost(variable, y, balance=True)
        elif path == "ts":
            nns_boost(variable, y, ts_test=4)
        else:
            nns_boost(variable, y, pred_int=0.95)


def test_nns_boost_rejects_unported_stochastic_epoch_path() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack([np.sin((idx + 1) * x) for idx in range(11)])
    y = x + np.sin(x)

    with pytest.raises(
        NotImplementedError,
        match="n_features > 10 requires R's stochastic epoch keeper loop",
    ):
        nns_boost(variable, y, variable[:3], cv_size=0.25, feature_importance=False)
