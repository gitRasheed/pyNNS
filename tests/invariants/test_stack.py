from __future__ import annotations

import numpy as np
import pytest

from pynns import nns_stack
from pynns.stack import _cv_split


def test_nns_stack_numeric_shapes_and_keys() -> None:
    x = np.linspace(-2.0, 2.0, 40)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x)
    point = variable[:7]

    result = nns_stack(variable, y, point, cv_size=0.25, folds=2, method=(1, 2))

    assert set(result) == {
        "OBJfn.reg",
        "NNS.reg.n.best",
        "probability.threshold",
        "OBJfn.dim.red",
        "NNS.dim.red.threshold",
        "reg",
        "reg.pred.int",
        "dim.red",
        "dim.red.pred.int",
        "stack",
        "pred.int",
    }
    assert result["reg"].shape == (7,)
    assert result["dim.red"].shape == (7,)
    assert result["stack"].shape == (7,)
    assert result["probability.threshold"] == pytest.approx(0.5)
    assert np.all(np.isfinite(result["stack"]))


@pytest.mark.parametrize("path", ["class", "balance"])
def test_nns_stack_deferred_paths_raise(path: str) -> None:
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack((x, np.sin(x)))
    y = x + np.sin(x)

    with pytest.raises(NotImplementedError):
        if path == "class":
            nns_stack(variable, y, type="class")
        else:
            nns_stack(variable, y, balance=True)


@pytest.mark.parametrize("method", [(1,), (2,), (1, 2)])
def test_nns_stack_pred_int_shapes(method: tuple[int, ...]) -> None:
    x = np.linspace(-2.0, 2.0, 40)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x)
    point = variable[:7]

    result = nns_stack(variable, y, point, cv_size=0.25, folds=1, method=method, pred_int=0.95)

    assert result["stack"].shape == (7,)
    assert result["pred.int"] is not None
    assert all(values.shape == (7,) for values in result["pred.int"].values())


@pytest.mark.parametrize("method", [(1,), (2,), (1, 2)])
def test_nns_stack_ts_test_shape_and_determinism(method: tuple[int, ...]) -> None:
    x = np.linspace(-2.0, 2.0, 40)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x)
    point = variable[:7]

    first = nns_stack(variable, y, point, cv_size=0.25, folds=1, method=method, ts_test=10)
    second = nns_stack(variable, y, point, cv_size=0.25, folds=1, method=method, ts_test=10)

    assert first["stack"].shape == (7,)
    assert set(first) == set(second)
    np.testing.assert_allclose(first["stack"], second["stack"])


def test_nns_stack_ts_test_split_matches_r_sizes() -> None:
    train_idx, test_idx = _cv_split(40, fold=1, cv_size=0.25, ts_test=10)

    assert train_idx.shape == (10,)
    assert test_idx.shape == (30,)
    np.testing.assert_array_equal(train_idx, np.arange(30, 40))
    np.testing.assert_array_equal(test_idx, np.arange(0, 30))


@pytest.mark.parametrize("ts_test", [0, 1, 41])
def test_nns_stack_invalid_ts_test_raises(ts_test: int) -> None:
    x = np.linspace(-2.0, 2.0, 40)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x)

    with pytest.raises(ValueError):
        nns_stack(variable, y, variable[:3], cv_size=0.25, folds=1, method=1, ts_test=ts_test)
