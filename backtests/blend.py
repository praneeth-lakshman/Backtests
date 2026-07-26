"""Run several strategies at once by splitting capital equally between them."""

from __future__ import annotations

import pandas as pd

from .portfolio import Portfolio
from .strategies import STRATEGY_METHODS, Strategy, run_strategy


def blended_portfolio_values(
    stock_data: pd.DataFrame,
    ticker: str,
    labels: list[str],
    initial_money: float,
    short_roll: int,
    long_roll: int,
    first_shares: int,
    rsi_period: int = 14,
    rsi_oversold: float = 30,
    rsi_overbought: float = 70,
) -> list[float]:
    """Value, over time, of splitting ``initial_money`` equally across ``labels``
    and running each strategy independently on its own slice.

    This models running the strategies simultaneously with a fixed allocation,
    as opposed to a single portfolio that switches between rules.
    """
    if not labels:
        raise ValueError("Select at least one strategy to blend.")
    unknown = set(labels) - STRATEGY_METHODS.keys()
    if unknown:
        raise ValueError(f"Unknown strategies: {sorted(unknown)}")

    share = initial_money / len(labels)
    total = [0.0] * len(stock_data)
    for label in labels:
        portfolio = Portfolio(initial_money=share, share_holdings={ticker: 0})
        strategy = Strategy(
            stock_data=stock_data,
            portfolio=portfolio,
            ticker=ticker,
            short_roll=short_roll,
            long_roll=long_roll,
            first_shares=first_shares,
            rsi_period=rsi_period,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
        )
        values = run_strategy(strategy, label)
        total = [t + v for t, v in zip(total, values)]
    return total
