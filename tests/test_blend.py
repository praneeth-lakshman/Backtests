import pandas as pd
import pytest

from backtests.blend import blended_portfolio_values
from backtests.portfolio import Portfolio
from backtests.strategies import Strategy, run_strategy

STRATEGY_KWARGS = dict(
    short_roll=3,
    long_roll=5,
    first_shares=1,
    rsi_period=3,
    rsi_oversold=30,
    rsi_overbought=70,
)


def _sample_stock_data(n=20):
    index = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.Series([100 + i + (5 if i % 4 == 0 else 0) for i in range(n)], dtype=float)
    close.index = index
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close}, index=index)


def _run_alone(stock_data, ticker, label, money):
    portfolio = Portfolio(initial_money=money, share_holdings={ticker: 0})
    strategy = Strategy(stock_data=stock_data, portfolio=portfolio, ticker=ticker, **STRATEGY_KWARGS)
    return run_strategy(strategy, label)


def test_blend_of_one_strategy_matches_running_it_alone_with_full_money():
    stock_data = _sample_stock_data()
    blended = blended_portfolio_values(
        stock_data=stock_data,
        ticker="AAA",
        labels=["Cross (SMA)"],
        initial_money=1000,
        **STRATEGY_KWARGS,
    )
    alone = _run_alone(stock_data, "AAA", "Cross (SMA)", money=1000)
    assert blended == pytest.approx(alone)


def test_blend_of_two_strategies_equals_sum_of_independent_half_money_runs():
    stock_data = _sample_stock_data()
    labels = ["Buy and Hold", "Cross (SMA)"]

    blended = blended_portfolio_values(
        stock_data=stock_data,
        ticker="AAA",
        labels=labels,
        initial_money=1000,
        **STRATEGY_KWARGS,
    )

    expected = [0.0] * len(stock_data)
    for label in labels:
        values = _run_alone(stock_data, "AAA", label, money=500)
        expected = [a + b for a, b in zip(expected, values)]

    assert blended == pytest.approx(expected)


def test_blend_rejects_no_labels():
    with pytest.raises(ValueError):
        blended_portfolio_values(
            stock_data=_sample_stock_data(),
            ticker="AAA",
            labels=[],
            initial_money=1000,
            **STRATEGY_KWARGS,
        )


def test_blend_rejects_unknown_label():
    with pytest.raises(ValueError):
        blended_portfolio_values(
            stock_data=_sample_stock_data(),
            ticker="AAA",
            labels=["Not A Real Strategy"],
            initial_money=1000,
            **STRATEGY_KWARGS,
        )
