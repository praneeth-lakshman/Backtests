# Backtests

A lightweight backtesting engine for comparing simple technical-analysis trading
strategies against a buy-and-hold baseline, using historical price data from
[Yahoo Finance](https://finance.yahoo.com/).

Given a ticker, it runs five strategies over the same price history and
reports total return, excess return over buy-and-hold, annualized
volatility, max drawdown, beta, and Sharpe ratio for each — plus an
interactive HTML chart comparing portfolio value over time.

## Features

- **Five built-in strategies**: buy-and-hold, RSI-scaled momentum, SMA
  crossover, EMA crossover, and price-vs-short-SMA.
- **RSI-aware position sizing**: trend-following strategies scale trade size
  down as RSI approaches overbought/oversold thresholds instead of trading a
  fixed size on every signal.
- **Standard performance metrics**: total return, excess return, annualized
  volatility, max drawdown, beta, and Sharpe ratio, computed consistently
  across every strategy and interval.
- **Interactive web app**: enter a ticker, an investment amount, and a
  strategy, and see the resulting portfolio value and chart live in the
  browser — see [Web app](#web-app) below.
- **Configurable via CLI flags**: ticker, history period, bar interval,
  starting cash, SMA/EMA windows, RSI period and thresholds, and the
  risk-free rate — no code editing required.
- **Interactive chart output**: an HTML file (via Plotly) plotting every
  strategy's portfolio value over the backtest period.
- **Optional CSV export**: write the full metrics table to disk for further
  analysis.
- **Automated test suite**: unit tests for the portfolio, indicators, report,
  and metrics modules that run without network access.

## Installation

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For running the test suite, install the dev dependencies instead:

```bash
pip install -r requirements-dev.txt
```

## Usage

Run with the defaults (COIN, 1 year of daily bars):

```bash
python main.py
```

Customize the run with flags:

```bash
python main.py --ticker AAPL --period 6mo --interval 1d \
    --short-window 5 --long-window 20 \
    --initial-money 5000 --output-csv results.csv
```

Run `python main.py --help` for the full list of options, including RSI
period/thresholds, risk-free rate, and output paths.

Each run prints a metrics block per strategy to the console, saves an
interactive comparison chart (`Strategy_Comparison.html` by default), and
optionally writes a CSV summary if `--output-csv` is given.

## Web app

An interactive [Streamlit](https://streamlit.io/) page lets you set a ticker,
an investment amount, and which strategies to compare, then see the
resulting portfolio value, an interactive chart, and the full metrics table
— all without touching the command line.

```bash
streamlit run app.py
```

This opens the app at `http://localhost:8501`. Set your inputs in the
sidebar (ticker, initial investment, history period, bar interval,
strategies to compare, and an "Advanced settings" panel for SMA/EMA windows,
RSI parameters, and the risk-free rate), then click **Run backtest**.

## Strategies

| Strategy | Signal |
|---|---|
| **Buy and Hold** | Buy the maximum affordable shares at the open, hold for the entire period. Serves as the benchmark for excess return and beta. |
| **Momentum** | Buy when both the short- and long-window SMA slopes are positive (accelerating uptrend); sell when both are negative. Seeds a small position during the first 50 bars. |
| **Cross (SMA)** | Buy when the short SMA is above the long SMA, sell when it crosses below. |
| **Cross (EMA)** | Same as above, using exponential instead of simple moving averages. |
| **Price over Short** | Buy when price is above the short SMA, sell when it's below. |

All trend-following strategies scale each trade's size by how far RSI is
from its overbought/oversold thresholds, down to a floor of 10% of the
maximum affordable/sellable size.

## Metrics

- **Total return** — percentage change in portfolio value from start to end.
- **Excess return** — total return minus the buy-and-hold total return.
- **Annualized volatility** — standard deviation of period returns, scaled
  to a yearly figure based on the bar interval.
- **Max drawdown** — largest peak-to-trough decline in portfolio value.
- **Beta** — sensitivity of the strategy's returns to buy-and-hold's returns.
- **Sharpe ratio** — mean excess return over the risk-free rate, per unit of
  volatility.

## Project structure

```
main.py                  CLI entry point
app.py                    Streamlit web app
backtests/
  data.py                 Fetches and cleans price history from yfinance
  portfolio.py             Cash/share bookkeeping for the backtest
  indicators.py            SMA, EMA, slope, and Wilder's RSI
  strategies.py             The five strategies, sharing one Portfolio/RSI state
  metrics.py                Return, volatility, drawdown, beta, Sharpe ratio
  report.py                  Console summaries, HTML/Plotly chart, CSV export
tests/                    Unit tests (no network access required)
```

## Testing

```bash
pytest
```

## Disclaimer

This project is for educational and research purposes only. It does not
constitute financial advice, and past backtest performance is not indicative
of future results. Do not use it to make real trading decisions without your
own independent research and risk assessment.

## License

[MIT](LICENSE)
