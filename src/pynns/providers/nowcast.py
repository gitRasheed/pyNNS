from __future__ import annotations

import csv
import importlib
import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from pynns.nowcast import _DEFAULT_NOWCAST_SERIES, _normalize_month_label


class CsvNowcastProvider:
    """Local CSV provider for deterministic nowcast panels."""

    def __init__(
        self,
        path: str | Path,
        *,
        date_column: str = "date",
        series_columns: Sequence[str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.date_column = date_column
        self.series_columns = (
            None if series_columns is None else [str(name) for name in series_columns]
        )

    def fetch(self, series: Sequence[str], start_date: str) -> dict[str, object]:
        del series
        rows, fieldnames = self._read_rows()
        selected_columns = self._selected_columns(fieldnames)
        dates, values = self._parse_rows(rows, selected_columns, start_date)
        return {
            "dates": dates,
            "series": values,
            "metadata": {
                "provider": "csv",
                "path": str(self.path),
                "date_column": self.date_column,
                "series_columns": selected_columns,
            },
        }

    def _read_rows(self) -> tuple[list[Mapping[str, str]], list[str]]:
        if not self.path.exists():
            raise FileNotFoundError(f"CSV nowcast provider file does not exist: {self.path}")
        with self.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("CSV nowcast provider file is empty.")
            fieldnames = [str(name) for name in reader.fieldnames]
            rows = cast(list[Mapping[str, str]], list(reader))
        if not rows:
            raise ValueError("CSV nowcast provider file has no data rows.")
        if self.date_column not in fieldnames:
            raise ValueError(f"CSV nowcast provider missing date column: {self.date_column}")
        return rows, fieldnames

    def _selected_columns(self, fieldnames: Sequence[str]) -> list[str]:
        if self.series_columns is None:
            selected = [name for name in fieldnames if name != self.date_column]
        else:
            selected = list(self.series_columns)
            missing = [name for name in selected if name not in fieldnames]
            if missing:
                raise ValueError(
                    f"CSV nowcast provider missing selected series column: {missing[0]}"
                )
        if not selected:
            raise ValueError("CSV nowcast provider requires at least one usable series column.")
        return selected

    def _parse_rows(
        self,
        rows: Sequence[Mapping[str, str]],
        selected_columns: Sequence[str],
        start_date: str,
    ) -> tuple[list[str], OrderedDict[str, list[float | None]]]:
        start_month = _normalize_month_label(start_date)
        dates: list[str] = []
        values: OrderedDict[str, list[float | None]] = OrderedDict(
            (name, []) for name in selected_columns
        )
        for row_number, row in enumerate(rows, start=2):
            raw_date = row.get(self.date_column)
            if raw_date is None:
                raise ValueError(f"CSV nowcast provider row {row_number} is missing a date value.")
            month = _normalize_month_label(raw_date)
            if month < start_month:
                continue
            dates.append(month)
            for column in selected_columns:
                values[column].append(_parse_optional_float(row.get(column), column, row_number))

        if not dates:
            raise ValueError("CSV nowcast provider has no rows on or after start_date.")
        if len(set(dates)) != len(dates):
            raise ValueError("CSV nowcast provider dates must not contain duplicate months.")
        if dates != sorted(dates):
            raise ValueError(
                "CSV nowcast provider dates must be sorted in ascending monthly order."
            )
        lengths = {len(column_values) for column_values in values.values()}
        if lengths != {len(dates)}:
            raise ValueError("CSV nowcast provider series columns must have equal lengths.")
        return dates, values


class FredApiNowcastProvider:
    """Optional FRED API provider for deterministic nowcast payloads."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "FRED_API_KEY",
        series: Sequence[str] | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.series = None if series is None else [str(name) for name in series]

    def fetch(self, series: Sequence[str], start_date: str) -> dict[str, object]:
        selected_series = self.series if self.series is not None else [str(name) for name in series]
        if not selected_series:
            selected_series = list(_DEFAULT_NOWCAST_SERIES)
        api_key = self._api_key()
        fred_module = _load_fredapi()
        fred_client = fred_module.Fred(api_key=api_key)

        start_month = _normalize_month_label(start_date)
        monthly_series: OrderedDict[str, OrderedDict[str, float | None]] = OrderedDict()
        all_months: set[str] = set()
        for name in selected_series:
            raw_values = fred_client.get_series(name, observation_start=start_date)
            monthly_values = _monthly_last_observations(raw_values, start_month)
            monthly_series[name] = monthly_values
            all_months.update(monthly_values)

        dates = sorted(all_months)
        payload_series: OrderedDict[str, list[float | None]] = OrderedDict()
        for name, values in monthly_series.items():
            payload_series[name] = [values.get(month) for month in dates]

        return {
            "dates": dates,
            "series": payload_series,
            "metadata": {
                "provider": "fredapi",
                "series": selected_series,
                "api_key_env": self.api_key_env,
                "start_date": start_date,
                "dependency": "fredapi",
            },
        }

    def _api_key(self) -> str:
        api_key = self.api_key if self.api_key is not None else os.environ.get(self.api_key_env)
        if not api_key:
            raise ValueError(f"FRED API key is required; pass api_key or set {self.api_key_env}.")
        return api_key


def _load_fredapi() -> Any:
    try:
        return importlib.import_module("fredapi")
    except ImportError as exc:
        raise ImportError(
            "FredApiNowcastProvider requires the optional 'fredapi' dependency; "
            "install nns-pm[fred] or install fredapi."
        ) from exc


def _monthly_last_observations(
    raw_values: object,
    start_month: str,
) -> OrderedDict[str, float | None]:
    monthly_values: OrderedDict[str, float | None] = OrderedDict()
    for raw_date, raw_value in _iter_observations(raw_values):
        month = _normalize_month_label(raw_date)
        if month < start_month:
            continue
        monthly_values[month] = _coerce_optional_float(raw_value)
    return monthly_values


def _iter_observations(raw_values: object) -> list[tuple[object, object]]:
    items_method = getattr(raw_values, "items", None)
    if callable(items_method):
        return list(items_method())
    if isinstance(raw_values, Sequence) and not isinstance(raw_values, str | bytes):
        observations = list(raw_values)
        if all(
            isinstance(item, Sequence) and not isinstance(item, str | bytes)
            for item in observations
        ):
            return [(item[0], item[1]) for item in observations if len(item) >= 2]
    raise TypeError("fredapi get_series result must provide .items() or date/value pairs.")


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"FRED series contains a nonnumeric value: {value!r}") from exc
    if result != result:
        return None
    return result


def _parse_optional_float(value: Any, column: str, row_number: int) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"na", "nan", "null", "none"}:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            f"CSV nowcast provider column {column!r} row {row_number} "
            f"contains a nonnumeric value: {value!r}"
        ) from exc
