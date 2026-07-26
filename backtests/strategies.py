"""Trading strategies evaluated against a shared Portfolio and stock history."""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from .indicators import ema, rsi, sma, slope
from .portfolio import Portfolio


def resets_portfolio(strategy_method):
    """Reset the shared portfolio to its initial state before running a strategy."""

    @functools.wraps(strategy_method)
    def wrapper(self, *args, **kwargs):
        self.portfolio.reset()
        return strategy_method(self, *args, **kwargs)

    return wrapper


class Strategy:
    """Bundles a stock history, a portfolio, and RSI-scaled trading rules."""

    def __init__(
        self,
        stock_data: pd.DataFrame,
        portfolio: Portfolio,
        ticker: str,
        short_roll: int,
        long_roll: int,
        first_shares: int,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
    ):
        self.stock_data = stock_data
        self.portfolio = portfolio
        self.ticker = ticker
        self.short_roll = short_roll
        self.long_roll = long_roll
        self.first_shares = first_shares
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

        self.n = len(stock_data)
        self.close = stock_data["Close"]
        self.short_sma_slope = slope(sma(self.close, short_roll), stock_data.index)
        self.long_sma_slope = slope(sma(self.close, long_roll), stock_data.index)
        self.rsi_values = rsi(self.close, period=rsi_period)

    def _trade_sizes(self, price: float, rsi_value: float) -> tuple[int, int]:
        """Return (buy_shares, sell_shares), scaled down as RSI nears overbought/oversold."""
        max_shares_buy = int(self.portfolio.money // price)
        max_shares_sell = self.portfolio.portfolio[self.ticker]

        if pd.isna(rsi_value):
            return max_shares_buy, max_shares_sell

        buy_scale = max(0.1, (self.rsi_overbought - rsi_value) / self.rsi_overbought)
        sell_scale = max(0.1, (rsi_value - self.rsi_oversold) / self.rsi_oversold)

        buy_shares = min(max_shares_buy, max(1, int(max_shares_buy * buy_scale))) if max_shares_buy > 0 else 0
        sell_shares = (
            min(max_shares_sell, max(1, int(max_shares_sell * sell_scale))) if max_shares_sell > 0 else 0
        )
        return buy_shares, sell_shares

    # -- strategies -----------------------------------------------------------

    @resets_portfolio
    def buy_hold(self) -> list[float]:
        """Buy as many shares as possible at the open and hold for the whole period."""
        open_price = self.stock_data["Open"].iloc[0]
        shares = int(self.portfolio.money // open_price)
        self.portfolio.buy(ticker=self.ticker, shares=shares, price=open_price)

        values = [np.nan] * self.n
        for i in range(self.n):
            price = self.stock_data["Open"].iloc[i] if i == 0 else self.close.iloc[i]
            values[i] = self.portfolio.get_value(ticker=self.ticker, price=price)
        return values

    @resets_portfolio
    def momentum(self) -> list[float]:
        """Buy when both the short- and long-term SMA slopes are rising, sell when both fall."""
        values = [np.nan] * self.n
        for i in range(self.n):
            price = self.close.iloc[i]
            buy_shares, sell_shares = self._trade_sizes(price, self.rsi_values.iloc[i])

            if i <= 50 and i % 10 == 0:
                self.portfolio.buy(ticker=self.ticker, shares=self.first_shares, price=price)

            long_slope = self.long_sma_slope.iloc[i]
            short_slope = self.short_sma_slope.iloc[i]

            if pd.notna(long_slope) and pd.notna(short_slope):
                if long_slope > 0 and short_slope > 0:
                    self.portfolio.buy(ticker=self.ticker, shares=buy_shares, price=price)
                elif long_slope <= 0 and short_slope <= 0:
                    self.portfolio.sell(ticker=self.ticker, shares=sell_shares, price=price)

            values[i] = self.portfolio.get_value(ticker=self.ticker, price=price)
        return values

    @resets_portfolio
    def cross(self) -> list[float]:
        """Buy when the short SMA is above the long SMA, sell when it's below."""
        return self._crossover(sma(self.close, self.short_roll), sma(self.close, self.long_roll))

    @resets_portfolio
    def cross_ema(self) -> list[float]:
        """Buy when the short EMA is above the long EMA, sell when it's below."""
        return self._crossover(ema(self.close, self.short_roll), ema(self.close, self.long_roll))

    def _crossover(self, fast: pd.Series, slow: pd.Series) -> list[float]:
        values = [np.nan] * self.n
        for i in range(self.n):
            price = self.close.iloc[i]
            buy_shares, sell_shares = self._trade_sizes(price, self.rsi_values.iloc[i])
            fast_i, slow_i = fast.iloc[i], slow.iloc[i]

            if pd.notna(fast_i) and pd.notna(slow_i):
                if fast_i > slow_i:
                    self.portfolio.buy(ticker=self.ticker, shares=buy_shares, price=price)
                elif fast_i < slow_i:
                    self.portfolio.sell(ticker=self.ticker, shares=sell_shares, price=price)

            values[i] = self.portfolio.get_value(ticker=self.ticker, price=price)
        return values

    @resets_portfolio
    def price_short(self) -> list[float]:
        """Buy when price is above the short SMA, sell when it's below."""
        short_sma = sma(self.close, self.short_roll)
        values = [np.nan] * self.n
        for i in range(self.n):
            price = self.close.iloc[i]
            buy_shares, sell_shares = self._trade_sizes(price, self.rsi_values.iloc[i])
            short_i = short_sma.iloc[i]

            if pd.notna(short_i):
                if price > short_i:
                    self.portfolio.buy(ticker=self.ticker, shares=buy_shares, price=price)
                elif price < short_i:
                    self.portfolio.sell(ticker=self.ticker, shares=sell_shares, price=price)

            values[i] = self.portfolio.get_value(ticker=self.ticker, price=price)
        return values


#: Canonical display-name -> method-name mapping, shared by the CLI, the web
#: app, and strategy blending so they always agree on what strategies exist.
STRATEGY_METHODS: dict[str, str] = {
    "Buy and Hold": "buy_hold",
    "Momentum": "momentum",
    "Cross (SMA)": "cross",
    "Cross (EMA)": "cross_ema",
    "Price over Short": "price_short",
}


def run_strategy(strategy: Strategy, label: str) -> list[float]:
    """Run the strategy method registered under ``label`` (see STRATEGY_METHODS)."""
    return getattr(strategy, STRATEGY_METHODS[label])()
