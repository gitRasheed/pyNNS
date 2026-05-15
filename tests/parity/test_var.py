from __future__ import annotations

import numpy as np

from pynns.var import _lag_mtx


def test_lag_mtx_scalar_tau_matches_reference_blocks() -> None:
    x = np.column_stack(
        (
            np.array([1, 2, 3, 4, 5], dtype=np.float64),
            np.array([6, 7, 8, 9, 10], dtype=np.float64),
        )
    )
    actual, names = _lag_mtx(x, 2, names=["a", "b"])

    expected = np.array(
        [
            [3, 8, 2, 1, 7, 6],
            [4, 9, 3, 2, 8, 7],
            [5, 10, 4, 3, 9, 8],
        ],
        dtype=np.float64,
    )
    expected_names = ["a_tau_0", "b_tau_0", "a_tau_1", "a_tau_2", "b_tau_1", "b_tau_2"]

    np.testing.assert_allclose(actual, expected)
    assert names == expected_names


def test_lag_mtx_nested_tau_keeps_requested_lags_plus_tau_zero() -> None:
    x = np.column_stack(
        (
            np.array([1, 2, 3, 4, 5], dtype=np.float64),
            np.array([6, 7, 8, 9, 10], dtype=np.float64),
        )
    )
    actual, names = _lag_mtx(x, ([1, 2], [1]), names=["a", "b"])

    expected_names = ["a_tau_0", "b_tau_0", "a_tau_1", "a_tau_2", "b_tau_1"]
    expected = np.array(
        [
            [3, 8, 2, 1, 7],
            [4, 9, 3, 2, 8],
            [5, 10, 4, 3, 9],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(actual, expected)
    assert names == expected_names
