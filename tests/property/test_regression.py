from __future__ import annotations

from typing import cast

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import nns_reg
from pynns.part import NoiseReduction
from pynns.regression import Order

finite_arrays = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=12, max_value=100),
    elements=st.floats(
        min_value=-1e4,
        max_value=1e4,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)


@given(
    finite_arrays,
    finite_arrays,
    st.sampled_from([None, 1, 2, 3, "max"]),
    st.sampled_from(["off", "mean", "median", "mode", "mode_class"]),
)
def test_nns_reg_univariate_shape_invariants_hold(
    x: np.ndarray,
    y: np.ndarray,
    order: int | str | None,
    noise: str,
) -> None:
    size = min(x.size, y.size)
    x = x[:size]
    y = y[:size]
    assume(np.unique(x).size > 1)
    assume(np.unique(y).size > 1)

    result = nns_reg(x, y, order=cast(Order, order), noise_reduction=cast(NoiseReduction, noise))

    assert np.isnan(result["R2"]) or -1e-12 <= result["R2"] <= 1.0 + 1e-12
    assert result["SE"] >= 0.0
    assert result["Fitted.xy"]["x"].shape == (size,)
    assert result["Fitted.xy"]["y"].shape == (size,)
    assert result["Fitted.xy"]["y.hat"].shape == (size,)
    assert result["Fitted.xy"]["gradient"].shape == (size,)
    assert result["derivative"]["Coefficient"].size == result["derivative"]["X.Lower.Range"].size
    assert result["derivative"]["Coefficient"].size == result["derivative"]["X.Upper.Range"].size
    assert result["regression.points"]["x"].size == result["regression.points"]["y"].size
