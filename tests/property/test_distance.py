from __future__ import annotations

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import nns_distance, nns_distance_bulk

feature_matrices = arrays(
    dtype=np.float64,
    shape=st.tuples(st.integers(min_value=5, max_value=40), st.integers(min_value=2, max_value=6)),
    elements=st.floats(
        min_value=1e-6,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)


@given(feature_matrices)
def test_distance_predictions_are_finite(features: np.ndarray) -> None:
    assume(np.all(np.ptp(features, axis=0) > 0.0))
    y_hat = np.mean(features, axis=1)
    rpm = np.column_stack((features, y_hat))
    target = features[0] + 0.1

    assert np.isfinite(nns_distance(rpm, target, k=min(3, features.shape[0])))

    bulk = nns_distance_bulk(
        rpm,
        features[: min(4, features.shape[0])],
        k=min(3, features.shape[0]),
    )
    assert bulk.shape == (min(4, features.shape[0]),)
    assert np.all(np.isfinite(bulk))
