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
