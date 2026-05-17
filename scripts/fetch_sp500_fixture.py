from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

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
    tickers = _universe_symbols(args.universe)
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
    all_missing_tickers = sorted(
        set(tickers) - set(str(column) for column in adjusted_close.columns),
    )
    returns = adjusted_close.pct_change(fill_method=None).iloc[1:]
    returns = returns.replace([np.inf, -np.inf], np.nan)

    row_count_before_drop = int(returns.shape[0])
    bad_tickers = sorted(
        all_missing_tickers
        + [
            str(column)
            for column in returns.columns
            if returns[column].isna().any() or not np.isfinite(returns[column].to_numpy()).all()
        ],
    )
    returns = returns.drop(columns=[ticker for ticker in bad_tickers if ticker in returns.columns])
    returns = returns.dropna(axis=0, how="any")
    returns = returns.astype(float)

    output.parent.mkdir(parents=True, exist_ok=True)
    returns.to_csv(output, float_format="%.12g", index_label="Date")

    metadata = {
        "source": "Yahoo Finance via yfinance",
        "fetch_date": datetime.now(UTC).isoformat(),
        "universe": args.universe,
        "start": args.start,
        "end": args.end,
        "tickers_requested": tickers,
        "tickers_included": [str(column) for column in returns.columns],
        "tickers_dropped": bad_tickers,
        "row_count_before_drop": row_count_before_drop,
        "row_count": int(returns.shape[0]),
        "column_count": int(returns.shape[1]),
        "return_calculation_convention": (
            "simple daily returns from adjusted close: price.pct_change().iloc[1:]"
        ),
    }
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
    parser.add_argument("--universe", choices=["sp50", "sp100", "sp500"], default="sp100")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--max-symbols", type=int, default=None)
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


def _universe_symbols(universe: str) -> list[str]:
    if universe == "sp50":
        return _SP100_SYMBOLS[:50]
    if universe == "sp100":
        return list(_SP100_SYMBOLS)
    return _sp500_symbols_from_wikipedia()


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


if __name__ == "__main__":
    main()
