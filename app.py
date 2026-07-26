"""Interactive web app: see what an investment would be worth today under a chosen strategy."""

from __future__ import annotations

import streamlit as st

from backtests.blend import blended_portfolio_values
from backtests.data import get_stock_data
from backtests.metrics import best_strategy, evaluate
from backtests.portfolio import Portfolio
from backtests.report import annotate_best, build_comparison_figure, results_dataframe
from backtests.strategies import STRATEGY_METHODS, Strategy, run_strategy

_NUMERIC_DECIMALS = {
    "Total return %": 2,
    "Excess return %": 2,
    "Annualized volatility": 4,
    "Max drawdown %": 2,
    "Beta": 4,
    "Sharpe ratio": 4,
    "Final value $": 2,
}

st.set_page_config(page_title="Backtests", page_icon="📈", layout="wide")
st.title("Strategy Backtester")
st.caption("See how much money you'd have today if you'd invested and followed a given strategy.")

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker", value="COIN").strip().upper()
    initial_money = st.number_input("Initial investment ($)", min_value=1.0, value=1000.0, step=100.0)
    period = st.selectbox("History period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)
    interval = st.selectbox("Bar interval", ["1d", "1h", "1wk"], index=0)
    strategy_labels = st.multiselect(
        "Strategies to compare", list(STRATEGY_METHODS.keys()), default=list(STRATEGY_METHODS.keys())
    )
    blend = st.checkbox(
        "Also run the selected strategies simultaneously",
        help=(
            "Splits your initial investment equally across the strategies selected above and runs each "
            "one independently on its own slice, then adds the combined total as an extra line. "
            "Needs at least 2 strategies selected."
        ),
    )

    with st.expander("Advanced settings"):
        short_window = st.number_input("Short SMA/EMA window", min_value=2, value=10)
        long_window = st.number_input("Long SMA/EMA window", min_value=2, value=30)
        rsi_period = st.number_input("RSI period", min_value=2, value=14)
        rsi_oversold = st.number_input("RSI oversold threshold", min_value=1.0, max_value=99.0, value=30.0)
        rsi_overbought = st.number_input("RSI overbought threshold", min_value=1.0, max_value=99.0, value=70.0)
        risk_free_rate = st.number_input("Annual risk-free rate", min_value=0.0, max_value=1.0, value=0.05)
        first_shares = st.number_input("Momentum warm-up shares", min_value=0, value=1)

    run = st.button("Run backtest", type="primary")

if not run:
    st.info("Set your inputs in the sidebar and click **Run backtest**.")
    st.stop()

if not ticker:
    st.error("Enter a ticker symbol.")
    st.stop()

if not strategy_labels:
    st.error("Select at least one strategy.")
    st.stop()

if long_window <= short_window:
    st.error("Long window must be greater than short window.")
    st.stop()

try:
    with st.spinner(f"Fetching {ticker} data..."):
        stock_data = get_stock_data(ticker=ticker, period=period, interval=interval)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

strategy_kwargs = dict(
    short_roll=short_window,
    long_roll=long_window,
    first_shares=first_shares,
    rsi_period=rsi_period,
    rsi_oversold=rsi_oversold,
    rsi_overbought=rsi_overbought,
)

portfolio = Portfolio(initial_money=initial_money, share_holdings={ticker: 0})
strategy = Strategy(stock_data=stock_data, portfolio=portfolio, ticker=ticker, **strategy_kwargs)

# Buy and Hold is always computed as the benchmark for excess return / beta,
# even if the user didn't select it for display.
buy_hold_values = strategy.buy_hold()

series: dict[str, list[float]] = {}
results = []
for label in strategy_labels:
    values = buy_hold_values if label == "Buy and Hold" else run_strategy(strategy, label)
    series[label] = values
    results.append(evaluate(label, values, buy_hold_values, risk_free_rate, interval))

if blend:
    if len(strategy_labels) < 2:
        st.warning("Select at least two strategies to blend them together.")
    else:
        blended_label = "Blended (" + " + ".join(strategy_labels) + ")"
        blended_values = blended_portfolio_values(
            stock_data=stock_data,
            ticker=ticker,
            labels=strategy_labels,
            initial_money=initial_money,
            **strategy_kwargs,
        )
        series[blended_label] = blended_values
        results.append(evaluate(blended_label, blended_values, buy_hold_values, risk_free_rate, interval))

winner = best_strategy(results)

st.subheader(f"Results for {ticker}")

cols = st.columns(len(results))
for col, result in zip(cols, results):
    label = result["strategy"]
    is_best = label == winner
    col.metric(
        label=f"\U0001f3c6 {label}" if is_best else label,
        value=f"${result['final_value']:,.2f}",
        delta=f"{result['total_return_pct']:+.2f}%",
    )

fig = build_comparison_figure(stock_data.index, series, ticker, interval, highlight=winner)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Metrics")
table = annotate_best(results_dataframe(results), winner).round(_NUMERIC_DECIMALS)
st.dataframe(
    table.style.apply(
        lambda row: ["background-color: rgba(255, 190, 0, 0.18)" if row["Strategy"] == f"\U0001f3c6 {winner}" else "" for _ in row],
        axis=1,
    ),
    use_container_width=True,
    hide_index=True,
)
