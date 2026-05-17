from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_TICKER_ALIASES = {"^GSPC": "GSPC"}

_SP100_SYMBOLS = [
    "AAPL",
    "ABBV",
    "ABT",
    "ACN",
    "ADBE",
    "AIG",
    "AMD",
    "AMGN",
    "AMT",
    "AMZN",
    "AVGO",
    "AXP",
    "BA",
    "BAC",
    "BK",
    "BKNG",
    "BLK",
    "BMY",
    "BRK-B",
    "C",
    "CAT",
    "CHTR",
    "CL",
    "CMCSA",
    "COF",
    "COP",
    "COST",
    "CRM",
    "CSCO",
    "CVS",
    "CVX",
    "DE",
    "DHR",
    "DIS",
    "DUK",
    "EMR",
    "F",
    "FDX",
    "GD",
    "GE",
    "GILD",
    "GM",
    "GOOG",
    "GOOGL",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "INTU",
    "JNJ",
    "JPM",
    "KO",
    "LIN",
    "LLY",
    "LMT",
    "LOW",
    "MA",
    "MCD",
    "MDLZ",
    "MDT",
    "MET",
    "META",
    "MMM",
    "MO",
    "MRK",
    "MS",
    "MSFT",
    "NEE",
    "NFLX",
    "NKE",
    "NVDA",
    "ORCL",
    "PEP",
    "PFE",
    "PG",
    "PM",
    "PYPL",
    "QCOM",
    "RTX",
    "SBUX",
    "SCHW",
    "SO",
    "SPG",
    "T",
    "TGT",
    "TMO",
    "TMUS",
    "TSLA",
    "TXN",
    "UNH",
    "UNP",
    "UPS",
    "USB",
    "V",
    "VZ",
    "APD",
    "WFC",
    "WMT",
    "XOM",
]


def main() -> None:
    args = _parse_args()
    output = args.output
    metadata_output = output.with_name(f"{output.stem}_metadata.json")

    yf = _import_yfinance()
    tickers = _universe_symbols(args.universe, args.ticker_source_csv)
    for symbol in args.include_symbol:
        if symbol not in tickers:
            tickers.append(symbol)
    if args.max_symbols is not None:
        tickers = tickers[: args.max_symbols]

    prices = yf.download(
        tickers=tickers,
        start=args.start,
        end=args.end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    adjusted_close = _adjusted_close_frame(prices)
    adjusted_close = adjusted_close.rename(columns=_TICKER_ALIASES)
    tickers_for_metadata = [_TICKER_ALIASES.get(ticker, ticker) for ticker in tickers]
    all_missing_tickers = sorted(
        set(tickers_for_metadata) - set(str(column) for column in adjusted_close.columns),
    )
    returns = adjusted_close.pct_change(fill_method=None).iloc[1:]
    returns = returns.replace([np.inf, -np.inf], np.nan)

    row_count_before_drop = int(returns.shape[0])
    dropped: dict[str, str] = {
        ticker: "missing adjusted close data" for ticker in all_missing_tickers
    }
    for column in returns.columns:
        column_name = str(column)
        column_values = returns[column].to_numpy()
        if returns[column].isna().any():
            dropped[column_name] = "missing returns after pct_change"
        elif not np.isfinite(column_values).all():
            dropped[column_name] = "non-finite returns after pct_change"

    bad_tickers = sorted(dropped)
    returns = returns.drop(columns=[ticker for ticker in bad_tickers if ticker in returns.columns])
    returns = returns.dropna(axis=0, how="any")
    returns = returns.astype(float)

    output.parent.mkdir(parents=True, exist_ok=True)
    returns.to_csv(output, float_format="%.12g", index_label="Date")

    metadata = {
        "source": "Yahoo Finance via yfinance",
        "fetch_date": datetime.now(UTC).isoformat(),
        "universe": args.universe,
        "requested_universe": args.universe,
        "start": args.start,
        "end": args.end,
        "tickers_requested": tickers,
        "tickers_included": [str(column) for column in returns.columns],
        "tickers_dropped": bad_tickers,
        "ticker_aliases": _TICKER_ALIASES,
        "dropped_tickers": dropped,
        "row_count_before_drop": row_count_before_drop,
        "row_count": int(returns.shape[0]),
        "column_count": int(returns.shape[1]),
        "adjusted_close_convention": (
            "Yahoo Finance adjusted close column when available; close column fallback only if "
            "adjusted close is absent from the download payload."
        ),
        "cleaning_rules": [
            "download requested ticker adjusted close series",
            "compute simple daily returns with pct_change(fill_method=None).iloc[1:]",
            "replace positive and negative infinity with NaN",
            "drop tickers with missing or non-finite return observations",
            "drop rows with any remaining missing return observations",
        ],
        "full_sp500_or_cleaned_subset": (
            "full requested universe" if not bad_tickers else "cleaned subset of requested universe"
        ),
        "return_calculation_convention": (
            "simple daily returns from adjusted close: price.pct_change().iloc[1:]"
        ),
    }
    _add_benchmark_column_metadata(metadata, returns)
    metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"wrote {output} ({returns.shape[0]} rows x {returns.shape[1]} columns)")
    print(f"wrote {metadata_output}")
    if bad_tickers:
        print(f"dropped {len(bad_tickers)} tickers: {', '.join(bad_tickers)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manually fetch S&P-style adjusted close data with yfinance and write a static "
            "daily-return fixture. Example: uv run --with yfinance python "
            "scripts/fetch_sp500_fixture.py --universe sp100 --start 2019-01-01 "
            "--end 2024-01-01 --max-symbols 100 --output "
            "tests/fixtures/finance/sp500_daily_returns_2019_2023.csv"
        ),
    )
    parser.add_argument(
        "--universe",
        choices=["sp50", "sp100", "sp500", "fixture"],
        default="sp100",
    )
    parser.add_argument(
        "--ticker-source-csv",
        type=Path,
        default=None,
        help=(
            "Existing fixture CSV whose header supplies the ticker universe when "
            "`--universe fixture` is used."
        ),
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument(
        "--include-symbol",
        action="append",
        default=[],
        help="Extra ticker to append to the selected universe, e.g. --include-symbol SPY.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _import_yfinance() -> Any:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit(
            "yfinance is intentionally not a project dependency. Run this script with "
            "`uv run --with yfinance python scripts/fetch_sp500_fixture.py ...`."
        ) from exc
    return yf


def _universe_symbols(universe: str, ticker_source_csv: Path | None) -> list[str]:
    if universe == "fixture":
        if ticker_source_csv is None:
            raise SystemExit("--universe fixture requires --ticker-source-csv.")
        return _symbols_from_fixture_csv(ticker_source_csv)
    if universe == "sp50":
        return _SP100_SYMBOLS[:50]
    if universe == "sp100":
        return list(_SP100_SYMBOLS)
    return _sp500_symbols_from_wikipedia()


def _symbols_from_fixture_csv(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(",")
    if len(header) < 2 or header[0] != "Date":
        raise SystemExit(f"{path} does not look like a finance fixture CSV.")
    return [symbol for symbol in header[1:] if symbol not in {"GSPC", "^GSPC"}]


def _sp500_symbols_from_wikipedia() -> list[str]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "Fetching --universe sp500 requires pandas from the yfinance environment.",
        ) from exc

    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    if not tables:
        raise SystemExit("Could not read S&P 500 constituents from Wikipedia.")
    symbols = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    return symbols


def _adjusted_close_frame(prices: Any) -> Any:
    if "Adj Close" in prices:
        adjusted_close = prices["Adj Close"]
    elif "Close" in prices:
        adjusted_close = prices["Close"]
    else:
        raise SystemExit("Downloaded data did not contain adjusted close or close prices.")
    if getattr(adjusted_close, "ndim", 0) == 1:
        adjusted_close = adjusted_close.to_frame()
    adjusted_close = adjusted_close.dropna(axis=1, how="all")
    if adjusted_close.empty:
        raise SystemExit("No usable adjusted close prices were downloaded.")
    return adjusted_close


def _add_benchmark_column_metadata(metadata: dict[str, Any], returns: Any) -> None:
    included = {str(column) for column in returns.columns}
    benchmark_columns = {
        "market_index": "GSPC" if "GSPC" in included else None,
        "tradable_proxy": "SPY" if "SPY" in included else None,
        "excluded_from_constituents": [
            column for column in ("SPY", "GSPC") if column in included
        ],
        "yfinance_aliases": _TICKER_ALIASES,
    }
    metadata["benchmark_columns"] = benchmark_columns

    if {"SPY", "GSPC"}.issubset(included):
        diff = returns["SPY"] - returns["GSPC"]
        metadata["benchmark_column_sanity"] = {
            "spy_gspc_correlation": float(returns["SPY"].corr(returns["GSPC"])),
            "mean_abs_daily_return_difference": float(diff.abs().mean()),
            "max_abs_daily_return_difference": float(diff.abs().max()),
        }


if __name__ == "__main__":
    main()
