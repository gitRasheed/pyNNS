from __future__ import annotations

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import nns_arma

finite_arrays = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=10, max_value=100),
    elements=st.floats(
        min_value=-100.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)


@given(
    finite_arrays,
    st.integers(min_value=1, max_value=5),
    st.sampled_from([1, 4]),
    st.sampled_from(["lin", "nonlin", "both", "means"]),
)
def test_nns_arma_random_explicit_lag_shape(
    variable: np.ndarray,
    h: int,
    seasonal_factor: int,
    method: str,
) -> None:
    assume(np.ptp(variable) > 0.0)

    result = nns_arma(variable, h=h, seasonal_factor=seasonal_factor, method=method)

    assert result.shape == (h,)
