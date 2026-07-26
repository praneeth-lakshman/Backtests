"""Console, chart, and CSV reporting for backtest results."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

_LINE_COLORS = ["blue", "red", "green", "orange", "purple", "brown", "teal", "magenta"]

_COLUMN_LABELS = {
    "strategy": "Strategy",
    "total_return_pct": "Total return %",
    "excess_return_pct": "Excess return %",
    "annualized_volatility": "Annualized volatility",
    "max_drawdown_pct": "Max drawdown %",
    "beta": "Beta",
    "sharpe_ratio": "Sharpe ratio",
    "final_value": "Final value $",
}


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
    """Print final portfolio values, best strategy first and marked with a trophy."""
    print("=== FINAL PORTFOLIO VALUES ===")
    for i, result in enumerate(sorted(results, key=lambda r: r["final_value"], reverse=True)):
        marker = "\U0001f3c6 " if i == 0 else "   "
        print(f"{marker}{result['strategy']:<25}: ${result['final_value']:,.2f}")


def build_comparison_figure(
    index,
    series: dict[str, list[float]],
    ticker: str,
    interval: str,
    highlight: str | None = None,
) -> go.Figure:
    """Build an interactive chart comparing every strategy's portfolio value over time.

    If ``highlight`` names one of the series, its line is drawn bold and full
    opacity while the others are dimmed, so the best strategy stands out.
    """
    fig = go.Figure()
    for (name, values), color in zip(series.items(), _LINE_COLORS):
        is_highlighted = name == highlight
        fig.add_trace(
            go.Scatter(
                x=index,
                y=values,
                name=f"\U0001f3c6 {name}" if is_highlighted else name,
                line=dict(width=4 if is_highlighted else 2, color=color),
                opacity=1.0 if (highlight is None or is_highlighted) else 0.35,
            )
        )

    fig.update_layout(
        title=f"{ticker} Strategy Comparison ({interval} intervals)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_white",
    )
    return fig


def plot_comparison(
    index,
    series: dict[str, list[float]],
    ticker: str,
    interval: str,
    output_path: str,
    highlight: str | None = None,
) -> None:
    """Save an interactive HTML chart comparing every strategy's portfolio value."""
    build_comparison_figure(index, series, ticker, interval, highlight=highlight).write_html(output_path)


def results_dataframe(results: list[dict]) -> pd.DataFrame:
    """Turn a list of metric dicts into a table with friendly column names."""
    return pd.DataFrame(results).rename(columns=_COLUMN_LABELS)


def annotate_best(df: pd.DataFrame, best_label: str) -> pd.DataFrame:
    """Return a copy of a results table with the best strategy's row visually marked."""
    df = df.copy()
    is_best = df["Strategy"] == best_label
    df.loc[is_best, "Strategy"] = "\U0001f3c6 " + best_label
    return df


def save_results_csv(results: list[dict], output_path: str) -> None:
    """Write the full metrics table to a CSV file."""
    results_dataframe(results).to_csv(output_path, index=False)
