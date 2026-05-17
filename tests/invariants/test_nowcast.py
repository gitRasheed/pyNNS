from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from pynns import nns_nowcast, nns_nowcast_panel, nns_var
from pynns.providers import CsvNowcastProvider, FredApiNowcastProvider


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


def test_csv_nowcast_provider_returns_payload(tmp_path: Any) -> None:
    csv_path = tmp_path / "macro.csv"
    csv_path.write_text(
        "date,PAYEMS,UNRATE\n2020-01-15,1.0,4.0\n2020-02,2.0,5.0\n2020-03-31,3.0,6.0\n",
        encoding="utf-8",
    )

    payload = CsvNowcastProvider(csv_path).fetch(("PAYEMS",), "2020-01-01")

    assert payload["dates"] == ["2020-01", "2020-02", "2020-03"]
    series_payload = cast(Mapping[str, object], payload["series"])
    assert list(series_payload) == ["PAYEMS", "UNRATE"]
    assert payload["series"] == OrderedDict(
        (
            ("PAYEMS", [1.0, 2.0, 3.0]),
            ("UNRATE", [4.0, 5.0, 6.0]),
        )
    )
    assert payload["metadata"] == {
        "provider": "csv",
        "path": str(csv_path),
        "date_column": "date",
        "series_columns": ["PAYEMS", "UNRATE"],
    }


def test_nns_nowcast_csv_provider_matches_panel_core(tmp_path: Any) -> None:
    csv_path = tmp_path / "macro.csv"
    panel = _panel()
    rows = ["date,PAYEMS,UNRATE"]
    for index, month in enumerate(
        [f"2020-{month:02d}" for month in range(1, 13)]
        + [f"2021-{month:02d}" for month in range(1, 13)]
        + [f"2022-{month:02d}" for month in range(1, 13)]
        + [f"2023-{month:02d}" for month in range(1, 4)]
    ):
        rows.append(f"{month},{panel[index, 0]},{panel[index, 1]}")
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    actual = nns_nowcast(fetch=True, provider_backend=CsvNowcastProvider(csv_path), h=2)
    expected = nns_nowcast_panel(
        OrderedDict(
            (
                ("PAYEMS", panel[:, 0].tolist()),
                ("UNRATE", panel[:, 1].tolist()),
            )
        ),
        h=2,
        tau=12,
        dates=[row.split(",", maxsplit=1)[0] for row in rows[1:]],
    )

    assert actual["names"] == ["PAYEMS", "UNRATE"]
    assert actual["dates"]["forecast"] == ["2023-04", "2023-05"]
    for key in ("interpolated_and_extrapolated", "univariate", "multivariate", "ensemble"):
        np.testing.assert_allclose(actual[key], expected[key])
    assert np.array_equal(actual["relevant_variables"], expected["relevant_variables"])
    assert actual["metadata"]["provider"]["provider"] == "csv"


def test_csv_nowcast_provider_selects_and_orders_series_columns(tmp_path: Any) -> None:
    csv_path = tmp_path / "macro.csv"
    csv_path.write_text(
        "date,PAYEMS,UNRATE,GDPC1\n2020-01,1.0,4.0,7.0\n2020-02,2.0,5.0,8.0\n",
        encoding="utf-8",
    )

    payload = CsvNowcastProvider(csv_path, series_columns=["GDPC1", "PAYEMS"]).fetch((), "2020-01")

    series_payload = cast(Mapping[str, object], payload["series"])
    assert list(series_payload) == ["GDPC1", "PAYEMS"]
    assert payload["series"] == OrderedDict((("GDPC1", [7.0, 8.0]), ("PAYEMS", [1.0, 2.0])))


def test_csv_nowcast_provider_parses_missing_values(tmp_path: Any) -> None:
    csv_path = tmp_path / "macro.csv"
    csv_path.write_text(
        "date,PAYEMS,UNRATE\n2020-01,1.0,\n2020-02,NA,5.0\n2020-03,nan,null\n",
        encoding="utf-8",
    )

    payload = CsvNowcastProvider(csv_path).fetch((), "2020-01")

    assert payload["series"] == OrderedDict(
        (
            ("PAYEMS", [1.0, None, None]),
            ("UNRATE", [None, 5.0, None]),
        )
    )


def test_csv_nowcast_provider_filters_start_date(tmp_path: Any) -> None:
    csv_path = tmp_path / "macro.csv"
    csv_path.write_text(
        "date,PAYEMS\n2020-01,1.0\n2020-02,2.0\n2020-03,3.0\n",
        encoding="utf-8",
    )

    payload = CsvNowcastProvider(csv_path).fetch((), "2020-02-15")

    assert payload["dates"] == ["2020-02", "2020-03"]
    assert payload["series"] == OrderedDict((("PAYEMS", [2.0, 3.0]),))


def test_csv_nowcast_provider_rejects_bad_dates(tmp_path: Any) -> None:
    duplicate_path = tmp_path / "duplicate.csv"
    duplicate_path.write_text(
        "date,PAYEMS\n2020-01,1.0\n2020-01,2.0\n",
        encoding="utf-8",
    )
    unsorted_path = tmp_path / "unsorted.csv"
    unsorted_path.write_text(
        "date,PAYEMS\n2020-02,2.0\n2020-01,1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        CsvNowcastProvider(duplicate_path).fetch((), "2020-01")
    with pytest.raises(ValueError, match="sorted"):
        CsvNowcastProvider(unsorted_path).fetch((), "2020-01")


def test_csv_nowcast_provider_rejects_missing_columns(tmp_path: Any) -> None:
    csv_path = tmp_path / "macro.csv"
    csv_path.write_text(
        "month,PAYEMS\n2020-01,1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing date column"):
        CsvNowcastProvider(csv_path).fetch((), "2020-01")
    with pytest.raises(ValueError, match="missing selected series column"):
        CsvNowcastProvider(csv_path, date_column="month", series_columns=["UNRATE"]).fetch(
            (), "2020-01"
        )


def test_csv_nowcast_provider_rejects_nonnumeric_values(tmp_path: Any) -> None:
    csv_path = tmp_path / "macro.csv"
    csv_path.write_text(
        "date,PAYEMS\n2020-01,bad\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="nonnumeric"):
        CsvNowcastProvider(csv_path).fetch((), "2020-01")


def test_csv_nowcast_provider_rejects_empty_or_no_series_csv(tmp_path: Any) -> None:
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("", encoding="utf-8")
    header_only_path = tmp_path / "header_only.csv"
    header_only_path.write_text("date,PAYEMS\n", encoding="utf-8")
    no_series_path = tmp_path / "no_series.csv"
    no_series_path.write_text(
        "date\n2020-01\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="empty"):
        CsvNowcastProvider(empty_path).fetch((), "2020-01")
    with pytest.raises(ValueError, match="no data rows"):
        CsvNowcastProvider(header_only_path).fetch((), "2020-01")
    with pytest.raises(ValueError, match="at least one usable series"):
        CsvNowcastProvider(no_series_path).fetch((), "2020-01")


def test_fredapi_nowcast_provider_missing_dependency_raises_importerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(name: str) -> object:
        if name == "fredapi":
            raise ImportError("missing fredapi")
        raise AssertionError(name)

    monkeypatch.setattr("pynns.providers.nowcast.importlib.import_module", fake_import_module)

    with pytest.raises(ImportError, match="requires the optional 'fredapi' dependency"):
        FredApiNowcastProvider(api_key="key", series=["PAYEMS"]).fetch((), "2020-01")


def test_fredapi_nowcast_provider_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    with pytest.raises(ValueError, match="FRED API key is required"):
        FredApiNowcastProvider(series=["PAYEMS"]).fetch((), "2020-01")


def test_fredapi_nowcast_provider_does_not_read_dotenv(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    (tmp_path / ".env").write_text("FRED_API_KEY=secret\n", encoding="utf-8")

    with pytest.raises(ValueError, match="FRED API key is required"):
        FredApiNowcastProvider(series=["PAYEMS"]).fetch((), "2020-01")


def test_fredapi_nowcast_provider_uses_explicit_key_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_keys: list[str] = []

    class FakeFred:
        def __init__(self, api_key: str) -> None:
            captured_keys.append(api_key)

        def get_series(self, name: str, observation_start: str) -> OrderedDict[str, float]:
            return OrderedDict((("2020-01-01", 1.0),))

    monkeypatch.setenv("FRED_API_KEY", "env-key")
    monkeypatch.setattr(
        "pynns.providers.nowcast.importlib.import_module",
        lambda name: SimpleNamespace(Fred=FakeFred),
    )

    FredApiNowcastProvider(api_key="explicit-key", series=["PAYEMS"]).fetch((), "2020-01")

    assert captured_keys == ["explicit-key"]


def test_fredapi_nowcast_provider_uses_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_keys: list[str] = []

    class FakeFred:
        def __init__(self, api_key: str) -> None:
            captured_keys.append(api_key)

        def get_series(self, name: str, observation_start: str) -> OrderedDict[str, float]:
            return OrderedDict((("2020-01-01", 1.0),))

    monkeypatch.setenv("FRED_API_KEY", "env-key")
    monkeypatch.setattr(
        "pynns.providers.nowcast.importlib.import_module",
        lambda name: SimpleNamespace(Fred=FakeFred),
    )

    FredApiNowcastProvider(series=["PAYEMS"]).fetch((), "2020-01")

    assert captured_keys == ["env-key"]


def test_fredapi_nowcast_provider_monthly_series_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeFred:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def get_series(self, name: str, observation_start: str) -> OrderedDict[str, float]:
            assert observation_start == "2020-01-03"
            return OrderedDict((("2020-01-01", 1.0), ("2020-02-01", 2.0)))

    monkeypatch.setattr(
        "pynns.providers.nowcast.importlib.import_module",
        lambda name: SimpleNamespace(Fred=FakeFred),
    )

    payload = FredApiNowcastProvider(api_key="key", series=["PAYEMS"]).fetch((), "2020-01-03")

    assert payload["dates"] == ["2020-01", "2020-02"]
    assert payload["series"] == OrderedDict((("PAYEMS", [1.0, 2.0]),))
    assert payload["metadata"] == {
        "provider": "fredapi",
        "series": ["PAYEMS"],
        "api_key_env": "FRED_API_KEY",
        "start_date": "2020-01-03",
        "dependency": "fredapi",
    }


def test_fredapi_nowcast_provider_daily_values_use_monthly_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFred:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def get_series(self, name: str, observation_start: str) -> OrderedDict[str, float]:
            return OrderedDict(
                (
                    ("2020-01-02", 1.0),
                    ("2020-01-31", 2.0),
                    ("2020-02-07", 3.0),
                    ("2020-02-28", 4.0),
                )
            )

    monkeypatch.setattr(
        "pynns.providers.nowcast.importlib.import_module",
        lambda name: SimpleNamespace(Fred=FakeFred),
    )

    payload = FredApiNowcastProvider(api_key="key", series=["DGS10"]).fetch((), "2020-01")

    assert payload["dates"] == ["2020-01", "2020-02"]
    assert payload["series"] == OrderedDict((("DGS10", [2.0, 4.0]),))


def test_fredapi_nowcast_provider_quarterly_values_leave_other_months_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = {
        "GDPC1": OrderedDict((("2020-03-31", 100.0), ("2020-06-30", 110.0))),
        "PAYEMS": OrderedDict((("2020-01-01", 1.0), ("2020-02-01", 2.0), ("2020-03-01", 3.0))),
    }

    class FakeFred:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def get_series(self, name: str, observation_start: str) -> OrderedDict[str, float]:
            return data[name]

    monkeypatch.setattr(
        "pynns.providers.nowcast.importlib.import_module",
        lambda name: SimpleNamespace(Fred=FakeFred),
    )

    payload = FredApiNowcastProvider(api_key="key", series=["PAYEMS", "GDPC1"]).fetch((), "2020-01")

    assert payload["dates"] == ["2020-01", "2020-02", "2020-03", "2020-06"]
    assert payload["series"] == OrderedDict(
        (
            ("PAYEMS", [1.0, 2.0, 3.0, None]),
            ("GDPC1", [None, None, 100.0, 110.0]),
        )
    )


def test_fredapi_nowcast_provider_preserves_requested_series_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFred:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def get_series(self, name: str, observation_start: str) -> OrderedDict[str, float]:
            return OrderedDict((("2020-01-01", float(len(name))),))

    monkeypatch.setattr(
        "pynns.providers.nowcast.importlib.import_module",
        lambda name: SimpleNamespace(Fred=FakeFred),
    )

    payload = FredApiNowcastProvider(api_key="key", series=["UNRATE", "PAYEMS"]).fetch(
        (), "2020-01"
    )

    series_payload = cast(Mapping[str, object], payload["series"])
    assert list(series_payload) == ["UNRATE", "PAYEMS"]


def test_nns_nowcast_fredapi_provider_path_matches_panel_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel()
    dates = (
        [f"2020-{month:02d}-01" for month in range(1, 13)]
        + [f"2021-{month:02d}-01" for month in range(1, 13)]
        + [f"2022-{month:02d}-01" for month in range(1, 13)]
        + [f"2023-{month:02d}-01" for month in range(1, 4)]
    )
    data = {
        "PAYEMS": OrderedDict((date, float(panel[index, 0])) for index, date in enumerate(dates)),
        "UNRATE": OrderedDict((date, float(panel[index, 1])) for index, date in enumerate(dates)),
    }

    class FakeFred:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def get_series(self, name: str, observation_start: str) -> OrderedDict[str, float]:
            return data[name]

    monkeypatch.setattr(
        "pynns.providers.nowcast.importlib.import_module",
        lambda name: SimpleNamespace(Fred=FakeFred),
    )
    provider = FredApiNowcastProvider(api_key="key", series=["PAYEMS", "UNRATE"])

    actual = nns_nowcast(fetch=True, provider_backend=provider, h=2, keep_data=True)
    expected = nns_nowcast_panel(
        OrderedDict(
            (
                ("PAYEMS", panel[:, 0].tolist()),
                ("UNRATE", panel[:, 1].tolist()),
            )
        ),
        h=2,
        tau=12,
        dates=[date[:7] for date in dates],
    )

    assert actual["raw_dates"] == [date[:7] for date in dates]
    assert actual["raw_panel"] == {
        "PAYEMS": panel[:, 0].tolist(),
        "UNRATE": panel[:, 1].tolist(),
    }
    for key in ("interpolated_and_extrapolated", "univariate", "multivariate", "ensemble"):
        np.testing.assert_allclose(actual[key], expected[key])
    assert np.array_equal(actual["relevant_variables"], expected["relevant_variables"])
