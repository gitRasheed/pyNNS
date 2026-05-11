from __future__ import annotations

import numpy as np
import pytest

from pynns import nns_stack


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


@pytest.mark.parametrize("path", ["class", "balance", "ts", "interval"])
def test_nns_stack_deferred_paths_raise(path: str) -> None:
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack((x, np.sin(x)))
    y = x + np.sin(x)

    with pytest.raises(NotImplementedError):
        if path == "class":
            nns_stack(variable, y, type="class")
        elif path == "balance":
            nns_stack(variable, y, balance=True)
        elif path == "ts":
            nns_stack(variable, y, ts_test=4)
        else:
            nns_stack(variable, y, pred_int=0.95)
