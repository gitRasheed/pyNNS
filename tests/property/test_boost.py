from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import nns_boost


@settings(max_examples=10, deadline=None)
@given(
    arrays(
        np.float64,
        (20, 2),
        elements=st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        ),
    ),
    arrays(
        np.float64,
        20,
        elements=st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        ),
    ),
)
def test_nns_boost_outputs_are_finite_for_random_numeric_data(
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    result = nns_boost(
        x,
        y,
        x[:5],
        epochs=5,
        learner_trials=5,
        cv_size=0.25,
        random_seed=123,
    )

    predictions = result["results"]
    weights = result["feature.weights"]
    assert isinstance(predictions, np.ndarray)
    assert isinstance(weights, np.ndarray)
    assert predictions.shape == (5,)
    assert np.all(np.isfinite(predictions))
    assert np.all(weights >= 0.0)
    assert np.isclose(np.sum(weights), 1.0)
