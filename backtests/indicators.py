"""Technical indicators used by the trading strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return close.rolling(window=window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return close.ewm(span=span).mean()


def slope(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Rate of change of ``series`` per second, using the bar timestamps in ``index``."""
    seconds = index.to_series().diff().dt.total_seconds()
    return series.diff() / seconds


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    The seed average (at the first fully-formed window) is a plain rolling
    mean; every value after that recursively blends in the new gain/loss,
    which is what makes this "Wilder's" RSI rather than a plain EMA of
    gains/losses (those use a different, non-equivalent seed).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    for i in range(period + 1, len(close)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        result = 100 - (100 / (1 + rs))

    result = result.where(avg_loss != 0, 100.0)
    result = result.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return result
