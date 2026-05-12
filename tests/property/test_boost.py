from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import nns_boost

finite_matrices = arrays(
    dtype=np.float64,
    shape=st.tuples(st.integers(min_value=16, max_value=35), st.integers(min_value=2, max_value=3)),
    elements=st.floats(
        min_value=-100.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)


@given(finite_matrices)
def test_nns_boost_numeric_bounds_hold(x: np.ndarray) -> None:
    row_jitter = np.arange(x.shape[0], dtype=np.float64)[:, np.newaxis] * 1e-6
    col_jitter = np.arange(x.shape[1], dtype=np.float64)[np.newaxis, :] * 1e-7
    x = x + row_jitter + col_jitter
    y = 0.5 * x[:, 0] - 0.25 * x[:, 1]

    result = nns_boost(x, y, x[:3], cv_size=0.25, feature_importance=False)

    assert result["results"].shape == (3,)
    assert np.all(np.isfinite(result["results"]))
    assert np.sum(result["feature.weights"]) > 0.0


@given(finite_matrices, st.integers(min_value=2, max_value=4), st.sampled_from([1, 2]))
def test_nns_boost_class_shape_and_codes_hold(
    x: np.ndarray,
    n_classes: int,
    depth: int,
) -> None:
    row_jitter = np.arange(x.shape[0], dtype=np.float64)[:, np.newaxis] * 1e-6
    col_jitter = np.arange(x.shape[1], dtype=np.float64)[np.newaxis, :] * 1e-7
    x = x + row_jitter + col_jitter
    score = x[:, 0] + 0.25 * x[:, 1]
    quantiles = np.quantile(score, np.linspace(0.0, 1.0, n_classes + 1)[1:-1])
    y = np.searchsorted(quantiles, score, side="right").astype(np.float64) + 1.0

    result = nns_boost(x, y, x[:3], cv_size=0.25, depth=depth, type="class")

    assert result["results"].shape == (3,)
    assert np.all(np.isin(result["results"], np.unique(y)))
    assert np.sum(result["feature.weights"]) > 0.0


@pytest.mark.stochastic
@given(finite_matrices, st.integers(min_value=2, max_value=4), st.sampled_from([1, 2]))
def test_nns_boost_balance_class_shape_and_codes_hold(
    x: np.ndarray,
    n_classes: int,
    depth: int,
) -> None:
    row_jitter = np.arange(x.shape[0], dtype=np.float64)[:, np.newaxis] * 1e-6
    col_jitter = np.arange(x.shape[1], dtype=np.float64)[np.newaxis, :] * 1e-7
    x = x + row_jitter + col_jitter
    score = x[:, 0] + 0.25 * x[:, 1]
    quantiles = np.quantile(score, np.linspace(0.0, 1.0, n_classes + 1)[1:-1])
    y = np.searchsorted(quantiles, score, side="right").astype(np.float64) + 1.0

    result = nns_boost(
        x,
        y,
        x[:3],
        cv_size=0.25,
        depth=depth,
        type="class",
        balance=True,
        random_seed=5,
    )

    assert result["results"].shape == (3,)
    assert np.all(np.isin(result["results"], np.unique(y)))
    assert np.sum(result["feature.weights"]) > 0.0
