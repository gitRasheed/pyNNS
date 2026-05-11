from __future__ import annotations

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import nns_stack

finite_matrices = arrays(
    dtype=np.float64,
    shape=st.tuples(st.integers(min_value=16, max_value=45), st.integers(min_value=2, max_value=3)),
    elements=st.floats(
        min_value=-100.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)


@given(finite_matrices, st.sampled_from([[1], [2], [1, 2]]))
def test_nns_stack_numeric_bounds_hold(x: np.ndarray, method: list[int]) -> None:
    y = 0.5 * x[:, 0] - 0.25 * x[:, 1]
    assume(np.unique(y).size > 1)
    assume(all(np.unique(x[:, col]).size > 1 for col in range(x.shape[1])))

    result = nns_stack(x, y, x[:3], cv_size=0.25, folds=1, method=method)

    assert result["stack"].shape == (3,)
    assert np.all(np.isfinite(result["stack"]))
    assert result["probability.threshold"] == 0.5


@given(finite_matrices, st.integers(min_value=2, max_value=10))
def test_nns_stack_ts_test_method1_shape(x: np.ndarray, ts_test: int) -> None:
    assume(ts_test <= x.shape[0] - 2)
    y = 0.5 * x[:, 0] - 0.25 * x[:, 1]
    assume(np.unique(y).size > 1)
    assume(all(np.unique(x[:, col]).size > 1 for col in range(x.shape[1])))

    result = nns_stack(x, y, x[:3], cv_size=0.25, folds=1, method=1, ts_test=ts_test)

    assert result["stack"].shape == (3,)
    assert np.all(np.isfinite(result["stack"]))


@given(finite_matrices, st.sampled_from([[1], [2], [1, 2]]))
def test_nns_stack_pred_int_shape_invariants_hold(x: np.ndarray, method: list[int]) -> None:
    y = 0.5 * x[:, 0] - 0.25 * x[:, 1]
    assume(np.unique(y).size > 1)
    assume(all(np.unique(x[:, col]).size > 1 for col in range(x.shape[1])))

    result = nns_stack(x, y, x[:3], cv_size=0.25, folds=1, method=method, pred_int=0.95)

    assert result["stack"].shape == (3,)
    assert result["pred.int"] is not None
    assert all(values.shape == (3,) for values in result["pred.int"].values())


@given(finite_matrices, st.sampled_from([[1], [2], [1, 2]]), st.integers(min_value=2, max_value=4))
def test_nns_stack_class_shape_and_codes_hold(
    x: np.ndarray,
    method: list[int],
    n_classes: int,
) -> None:
    trend = np.linspace(-1.0, 1.0, x.shape[0])[:, np.newaxis]
    offsets = np.arange(1, x.shape[1] + 1, dtype=np.float64)[np.newaxis, :]
    x_values = x + trend * offsets
    score = x_values[:, 0] + 0.25 * x_values[:, 1]
    quantiles = np.quantile(score, np.linspace(0.0, 1.0, n_classes + 1)[1:-1])
    y = np.searchsorted(quantiles, score, side="right").astype(np.float64) + 1.0

    result = nns_stack(
        x_values,
        y,
        x_values[:3],
        cv_size=0.25,
        folds=1,
        method=method,
        type="class",
    )

    assert result["stack"].shape == (3,)
    assert np.all(np.isin(result["stack"], np.unique(y)))
    assert result["pred.int"] is None
