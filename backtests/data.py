"""Market data retrieval."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close")


def get_stock_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV history for ``ticker`` and return it cleaned and ready to backtest.

    yfinance includes a row with NaN OHLC values for a session that is still
    in progress (or otherwise unsettled); such rows are dropped since a
    strategy cannot trade on a price that doesn't exist yet.
    """
    stock_data = yf.Ticker(ticker).history(period=period, interval=interval)

    if stock_data.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' (period={period!r}, interval={interval!r}). "
            "Check the ticker symbol and your network connection."
        )

    stock_data.index = stock_data.index.tz_localize(None)

    incomplete = stock_data[list(REQUIRED_COLUMNS)].isna().any(axis=1)
    if incomplete.any():
        stock_data = stock_data.loc[~incomplete]

    if len(stock_data) < 2:
        raise ValueError(f"Not enough clean data points for ticker '{ticker}' to run a backtest.")

    return stock_data
