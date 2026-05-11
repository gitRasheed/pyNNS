from __future__ import annotations

import numpy as np
import pytest

from pynns import nns_arma


def test_nns_arma_output_length_matches_h() -> None:
    variable = np.sin(np.arange(1, 80, dtype=np.float64) / 4.0) + 2.0

    result = nns_arma(variable, h=12, seasonal_factor=4, method="nonlin")

    assert result.shape == (12,)


def test_nns_arma_repeated_calls_are_deterministic() -> None:
    variable = np.sin(np.arange(1, 60, dtype=np.float64) / 3.0) + 2.0

    first = nns_arma(variable, h=5, seasonal_factor=4, method="both")
    second = nns_arma(variable, h=5, seasonal_factor=4, method="both")

    np.testing.assert_array_equal(first, second)


def test_nns_arma_numeric_seasonal_dynamic_raises() -> None:
    variable = np.sin(np.arange(1, 40, dtype=np.float64))

    with pytest.raises(ValueError, match="dynamic"):
        nns_arma(variable, h=3, seasonal_factor=5, dynamic=True)


def test_nns_arma_pred_int_deferred() -> None:
    variable = np.sin(np.arange(1, 40, dtype=np.float64))

    with pytest.raises(NotImplementedError, match="NNS\\.MC / NNS\\.meboot"):
        nns_arma(variable, h=3, seasonal_factor=5, pred_int=0.95)


@pytest.mark.parametrize("values", [np.array([1.0, np.nan, 3.0]), np.array([1.0, np.inf, 3.0])])
def test_nns_arma_non_finite_inputs_raise(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        nns_arma(values)


def test_nns_arma_finite_where_r_is_finite_case() -> None:
    variable = np.arange(1, 21, dtype=np.float64)

    result = nns_arma(variable, h=5, seasonal_factor=4, method="lin")

    assert np.all(np.isfinite(result))
