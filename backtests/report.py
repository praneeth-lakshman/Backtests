"""Console, chart, and CSV reporting for backtest results."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

_LINE_COLORS = ["blue", "red", "green", "orange", "purple", "brown", "teal", "magenta"]


def print_summary(result: dict) -> None:
    """Print one strategy's metrics block to the console."""
    print(f"Strategy: {result['strategy']}")
    print(f"Total return: {result['total_return_pct']:.2f}%")
    print(f"Excess return: {result['excess_return_pct']:.2f}%")
    print(f"Volatility (annualized): {result['annualized_volatility']:.4f}")
    print(f"Max drawdown: {result['max_drawdown_pct']:.2f}%")
    print(f"Beta: {result['beta']:.4f}")
    print(f"Sharpe ratio: {result['sharpe_ratio']:.4f}")
    print()


def print_rankings(results: list[dict]) -> None:
    """Print final portfolio values, best strategy first."""
    print("=== FINAL PORTFOLIO VALUES ===")
    for result in sorted(results, key=lambda r: r["final_value"], reverse=True):
        print(f"{result['strategy']:<25}: ${result['final_value']:,.2f}")


def plot_comparison(index, series: dict[str, list[float]], ticker: str, interval: str, output_path: str) -> None:
    """Save an interactive HTML chart comparing every strategy's portfolio value."""
    fig = go.Figure()
    for (name, values), color in zip(series.items(), _LINE_COLORS):
        fig.add_trace(go.Scatter(x=index, y=values, name=name, line=dict(width=2, color=color)))

    fig.update_layout(
        title=f"{ticker} Strategy Comparison ({interval} intervals)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_white",
    )
    fig.write_html(output_path)


def save_results_csv(results: list[dict], output_path: str) -> None:
    """Write the full metrics table to a CSV file."""
    pd.DataFrame(results).to_csv(output_path, index=False)
