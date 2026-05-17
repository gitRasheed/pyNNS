from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import pytest

from pynns import nns_nowcast, nns_nowcast_panel, nns_var


class FixtureNowcastProvider:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.requested_series: tuple[str, ...] | None = None
        self.requested_start_date: str | None = None

    def fetch(self, series: tuple[str, ...], start_date: str) -> dict[str, object]:
        self.requested_series = series
        self.requested_start_date = start_date
        return self.payload


def _panel() -> np.ndarray:
    idx = np.arange(1, 40, dtype=np.float64)
    return np.column_stack(
        (
            np.sin(idx / 3.0) + 2.0,
            np.cos(idx / 5.0) + 3.0,
        )
    )


def test_nns_nowcast_panel_array_h0_matches_var_core() -> None:
    panel = _panel()

    actual = nns_nowcast_panel(panel, h=0, tau=2)
    expected = nns_var(panel, h=0, tau=2)

    assert set(actual) == {
        "interpolated_and_extrapolated",
        "names",
        "dates",
        "metadata",
    }
    np.testing.assert_allclose(
        actual["interpolated_and_extrapolated"],
        expected["interpolated_and_extrapolated"],
    )
    assert actual["names"] == ["x1", "x2"]
    assert actual["dates"] == {
        "observed": None,
        "forecast": [],
        "interpolated_and_extrapolated": None,
    }
    assert actual["metadata"] == {
        "source": "user_panel",
        "freq": "monthly",
        "tau": 2,
        "dim_red_method": "cor",
        "naive_weights": False,
    }


def test_nns_nowcast_panel_array_h3_matches_var_core() -> None:
    panel = _panel()

    actual = nns_nowcast_panel(panel, h=3, tau=2, dim_red_method="NNS.dep")
    expected = nns_var(panel, h=3, tau=2, dim_red_method="NNS.dep", naive_weights=False)

    assert set(actual) == {
        "interpolated_and_extrapolated",
        "relevant_variables",
        "univariate",
        "multivariate",
        "ensemble",
        "names",
        "dates",
        "metadata",
    }
    for key in ("interpolated_and_extrapolated", "univariate", "multivariate", "ensemble"):
        np.testing.assert_allclose(actual[key], expected[key])
    assert np.array_equal(actual["relevant_variables"], expected["relevant_variables"])
    assert actual["names"] == ["x1", "x2"]
    assert actual["dates"]["observed"] is None
    assert actual["dates"]["forecast"] == ["t+1", "t+2", "t+3"]
    assert actual["dates"]["interpolated_and_extrapolated"] is None


def test_nns_nowcast_panel_mapping_preserves_column_order_and_names() -> None:
    panel = OrderedDict(
        (
            ("PAYEMS", [1.0, 2.0, 3.0, 4.0, 5.0]),
            ("GDPC1", [2.0, 3.0, 4.0, 5.0, 6.0]),
        )
    )

    actual = nns_nowcast_panel(panel, h=0, tau=1)

    assert actual["names"] == ["PAYEMS", "GDPC1"]
    np.testing.assert_allclose(
        actual["interpolated_and_extrapolated"],
        np.column_stack((panel["PAYEMS"], panel["GDPC1"])),
    )


def test_nns_nowcast_panel_rejects_mismatched_names() -> None:
    with pytest.raises(ValueError, match="names length"):
        nns_nowcast_panel(_panel(), h=0, names=["only_one"])


def test_nns_nowcast_panel_normalizes_dates_and_forecast_months() -> None:
    panel = _panel()
    dates = ["2020-01-15", "2020-02", np.datetime64("2020-03-31")]
    dates.extend(f"2020-{month:02d}" for month in range(4, 13))
    dates.extend(f"2021-{month:02d}" for month in range(1, 13))
    dates.extend(f"2022-{month:02d}" for month in range(1, 13))
    dates.extend(f"2023-{month:02d}" for month in range(1, 4))

    actual = nns_nowcast_panel(
        panel,
        h=2,
        tau=1,
        dates=dates,
    )

    assert actual["dates"]["observed"][:3] == ["2020-01", "2020-02", "2020-03"]
    assert actual["dates"]["forecast"] == ["2023-04", "2023-05"]
    assert actual["dates"]["interpolated_and_extrapolated"] == actual["dates"]["observed"]


def test_nns_nowcast_panel_rejects_invalid_dates() -> None:
    panel = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="dates length"):
        nns_nowcast_panel(panel, h=0, dates=["2020-01"])
    with pytest.raises(ValueError, match="duplicate"):
        nns_nowcast_panel(panel, h=0, dates=["2020-01", "2020-01", "2020-02"])
    with pytest.raises(ValueError, match="sorted"):
        nns_nowcast_panel(panel, h=0, dates=["2020-02", "2020-01", "2020-03"])


def test_nns_nowcast_panel_missing_values_delegate_to_var() -> None:
    panel = _panel()
    panel[4, 0] = np.nan
    panel[-1, 1] = np.nan

    actual = nns_nowcast_panel(panel, h=3, tau=2)

    assert np.all(np.isfinite(actual["interpolated_and_extrapolated"]))
    assert np.all(np.isfinite(actual["univariate"]))
    assert np.all(np.isfinite(actual["multivariate"]))
    assert np.all(np.isfinite(actual["ensemble"]))


def _provider_payload() -> dict[str, Any]:
    panel = _panel()
    return {
        "dates": [f"2020-{month:02d}" for month in range(1, 13)]
        + [f"2021-{month:02d}" for month in range(1, 13)]
        + [f"2022-{month:02d}" for month in range(1, 13)]
        + [f"2023-{month:02d}" for month in range(1, 4)],
        "series": OrderedDict(
            (
                ("PAYEMS", panel[:, 0].tolist()),
                ("UNRATE", panel[:, 1].tolist()),
            )
        ),
        "metadata": {"provider": "fixture"},
    }


def test_nns_nowcast_default_call_remains_guarded() -> None:
    with pytest.raises(NotImplementedError, match="default live macro data retrieval"):
        nns_nowcast(h=1)


def test_nns_nowcast_fetch_requires_provider() -> None:
    with pytest.raises(ValueError, match="provider_backend is required"):
        nns_nowcast(fetch=True)


def test_nns_nowcast_provider_path_matches_panel_core() -> None:
    payload = _provider_payload()
    provider = FixtureNowcastProvider(payload)

    actual = nns_nowcast(fetch=True, provider_backend=provider, h=2, start_date="2020-01-03")
    expected = nns_nowcast_panel(
        payload["series"],
        h=2,
        tau=12,
        dates=payload["dates"],
        naive_weights=False,
    )

    assert provider.requested_start_date == "2020-01-03"
    assert provider.requested_series is not None
    assert provider.requested_series[:2] == ("PAYEMS", "JTSJOL")
    assert actual["names"] == ["PAYEMS", "UNRATE"]
    assert actual["dates"]["forecast"] == ["2023-04", "2023-05"]
    for key in ("interpolated_and_extrapolated", "univariate", "multivariate", "ensemble"):
        np.testing.assert_allclose(actual[key], expected[key])
    assert np.array_equal(actual["relevant_variables"], expected["relevant_variables"])
    assert actual["metadata"]["source"] == "provider"
    assert actual["metadata"]["provider"] == {"provider": "fixture"}
    assert "raw_panel" not in actual


def test_nns_nowcast_keep_data_includes_raw_provider_panel() -> None:
    payload = _provider_payload()

    actual = nns_nowcast(
        fetch=True,
        provider_backend=FixtureNowcastProvider(payload),
        h=0,
        keep_data=True,
    )

    assert actual["raw_panel"] == {
        "PAYEMS": payload["series"]["PAYEMS"],
        "UNRATE": payload["series"]["UNRATE"],
    }
    assert actual["raw_dates"] == payload["dates"]


def test_nns_nowcast_provider_rejects_unsupported_r_fetch_arguments() -> None:
    provider = FixtureNowcastProvider(_provider_payload())

    with pytest.raises(ValueError, match="additional_regressors"):
        nns_nowcast(fetch=True, provider_backend=provider, additional_regressors=["SPY"])
    with pytest.raises(ValueError, match="specific_regressors"):
        nns_nowcast(fetch=True, provider_backend=provider, specific_regressors=[1])


def test_nns_nowcast_provider_payload_validation() -> None:
    with pytest.raises(ValueError, match="'series'"):
        nns_nowcast(fetch=True, provider_backend=FixtureNowcastProvider({}))
    with pytest.raises(ValueError, match="equal lengths"):
        nns_nowcast(
            fetch=True,
            provider_backend=FixtureNowcastProvider(
                {"series": OrderedDict((("PAYEMS", [1.0, 2.0]), ("UNRATE", [3.0])))}
            ),
        )
    with pytest.raises(ValueError, match="dates length"):
        nns_nowcast(
            fetch=True,
            provider_backend=FixtureNowcastProvider(
                {"dates": ["2020-01"], "series": OrderedDict((("PAYEMS", [1.0, 2.0]),))}
            ),
        )
    with pytest.raises(ValueError):
        nns_nowcast(
            fetch=True,
            provider_backend=FixtureNowcastProvider(
                {"series": OrderedDict((("PAYEMS", [1.0, "bad"]),))}
            ),
        )
