from __future__ import annotations

import numpy as np
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
