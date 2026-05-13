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


@pytest.mark.stochastic
def test_nns_arma_pred_int_returns_interval_dict() -> None:
    variable = np.sin(np.arange(1, 40, dtype=np.float64) / 3.0) + 2.0

    result = nns_arma(
        variable,
        h=3,
        seasonal_factor=4,
        method="nonlin",
        pred_int=0.95,
        random_seed=123,
    )

    assert isinstance(result, dict)
    assert list(result) == ["Estimates", "Lower 95% pred.int", "Upper 95% pred.int"]
    assert all(value.shape == (3,) for value in result.values())


@pytest.mark.stochastic
def test_nns_arma_pred_int_seed_reproducibility() -> None:
    variable = np.sin(np.arange(1, 45, dtype=np.float64) / 3.0) + 2.0

    first = nns_arma(variable, h=4, seasonal_factor=4, method="nonlin", pred_int=0.8, random_seed=1)
    second = nns_arma(
        variable, h=4, seasonal_factor=4, method="nonlin", pred_int=0.8, random_seed=1
    )
    third = nns_arma(variable, h=4, seasonal_factor=4, method="nonlin", pred_int=0.8, random_seed=2)

    assert isinstance(first, dict)
    assert isinstance(second, dict)
    assert isinstance(third, dict)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert not np.array_equal(first["Lower 80% pred.int"], third["Lower 80% pred.int"])


def test_nns_arma_static_linear_pred_int_matches_installed_r_error() -> None:
    variable = np.arange(1, 21, dtype=np.float64)

    with pytest.raises(TypeError, match="non-numeric argument"):
        nns_arma(variable, h=5, seasonal_factor=4, method="lin", pred_int=0.95)


def test_nns_arma_pred_int_h_one_matches_installed_r_error() -> None:
    variable = np.sin(np.arange(1, 40, dtype=np.float64) / 3.0) + 2.0

    with pytest.raises(ValueError, match="incorrect number of dimensions"):
        nns_arma(variable, h=1, seasonal_factor=4, method="nonlin", pred_int=0.95)


@pytest.mark.parametrize("values", [np.array([1.0, np.nan, 3.0]), np.array([1.0, np.inf, 3.0])])
def test_nns_arma_non_finite_inputs_raise(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        nns_arma(values)


def test_nns_arma_finite_where_r_is_finite_case() -> None:
    variable = np.arange(1, 21, dtype=np.float64)

    result = nns_arma(variable, h=5, seasonal_factor=4, method="lin")

    assert np.all(np.isfinite(result))
