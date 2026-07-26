"""CLI entry point: run every strategy for a ticker and report the results."""

from __future__ import annotations

import argparse

from backtests.data import get_stock_data
from backtests.metrics import evaluate
from backtests.portfolio import Portfolio
from backtests.report import plot_comparison, print_rankings, print_summary, save_results_csv
from backtests.strategies import Strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest a handful of simple trading strategies against buy-and-hold."
    )
    parser.add_argument("--ticker", default="COIN", help="Ticker symbol to backtest (default: %(default)s)")
    parser.add_argument(
        "--period", default="1y", help="yfinance history period, e.g. 1y, 6mo, 5d (default: %(default)s)"
    )
    parser.add_argument("--interval", default="1d", help="Bar interval, e.g. 1d, 1h, 5m (default: %(default)s)")
    parser.add_argument("--initial-money", type=float, default=1000, help="Starting cash (default: %(default)s)")
    parser.add_argument("--short-window", type=int, default=10, help="Short SMA/EMA window (default: %(default)s)")
    parser.add_argument("--long-window", type=int, default=30, help="Long SMA/EMA window (default: %(default)s)")
    parser.add_argument("--rsi-period", type=int, default=14, help="RSI lookback period (default: %(default)s)")
    parser.add_argument("--rsi-oversold", type=float, default=30, help="RSI oversold threshold (default: %(default)s)")
    parser.add_argument(
        "--rsi-overbought", type=float, default=70, help="RSI overbought threshold (default: %(default)s)"
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.05,
        help="Annual risk-free rate for Sharpe ratio (default: %(default)s)",
    )
    parser.add_argument(
        "--first-shares",
        type=int,
        default=1,
        help="Shares bought per warm-up tick in the momentum strategy (default: %(default)s)",
    )
    parser.add_argument(
        "--output-html",
        default="Strategy_Comparison.html",
        help="Path to write the interactive comparison chart (default: %(default)s)",
    )
    parser.add_argument("--output-csv", default=None, help="Optional path to write a CSV summary of all metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stock_data = get_stock_data(ticker=args.ticker, period=args.period, interval=args.interval)
    portfolio = Portfolio(initial_money=args.initial_money, share_holdings={args.ticker: 0})
    strategy = Strategy(
        stock_data=stock_data,
        portfolio=portfolio,
        ticker=args.ticker,
        short_roll=args.short_window,
        long_roll=args.long_window,
        first_shares=args.first_shares,
        rsi_period=args.rsi_period,
        rsi_oversold=args.rsi_oversold,
        rsi_overbought=args.rsi_overbought,
    )

    runs = {
        "Buy and Hold": strategy.buy_hold,
        "Momentum": strategy.momentum,
        "Cross (SMA)": strategy.cross,
        "Price over Short": strategy.price_short,
        "Cross (EMA)": strategy.cross_ema,
    }

    print("=== STRATEGY PERFORMANCE ANALYSIS ===\n")

    series: dict[str, list[float]] = {}
    results: list[dict] = []
    benchmark: list[float] | None = None
    for name, run in runs.items():
        values = run()
        series[name] = values
        if name == "Buy and Hold":
            benchmark = values
        result = evaluate(name, values, benchmark, args.risk_free_rate, args.interval)
        results.append(result)
        print_summary(result)

    plot_comparison(stock_data.index, series, args.ticker, args.interval, args.output_html)
    print(f"Interactive plot saved as '{args.output_html}'")

    if args.output_csv:
        save_results_csv(results, args.output_csv)
        print(f"Metrics summary saved as '{args.output_csv}'")

    print()
    print_rankings(results)


if __name__ == "__main__":
    main()
