"""Performance-evaluation metrics for a portfolio value series."""

from __future__ import annotations

import numpy as np


def total_return(values: list[float]) -> float:
    """Total return over the period, in percent."""
    return (values[-1] - values[0]) / values[0] * 100


def excess_return(values: list[float], benchmark: list[float]) -> float:
    """Total return in excess of a benchmark, in percent."""
    return total_return(values) - total_return(benchmark)


def daily_returns(values: list[float]) -> np.ndarray:
    """Period-over-period returns, with the first entry fixed at 0."""
    arr = np.asarray(values, dtype=float)
    returns = np.zeros_like(arr)
    returns[1:] = arr[1:] / arr[:-1] - 1
    return returns


def volatility(values: list[float]) -> float:
    """Standard deviation of period returns."""
    return float(np.std(daily_returns(values)))


def max_drawdown(values: list[float]) -> float:
    """Largest peak-to-trough decline over the period, in percent."""
    arr = np.asarray(values, dtype=float)
    running_max = np.maximum.accumulate(arr)
    drawdowns = (arr - running_max) / running_max
    return abs(drawdowns.min()) * 100


def beta(values: list[float], benchmark: list[float]) -> float:
    """Sensitivity of returns to the benchmark's returns."""
    returns = daily_returns(values)
    benchmark_returns = daily_returns(benchmark)
    # ddof=0 on both sides so covariance and variance use the same
    # (population) normalization; np.cov's default ddof=1 would otherwise
    # silently mismatch np.var's default ddof=0 and skew the result.
    covariance = np.cov(returns, benchmark_returns, ddof=0)[0, 1]
    return float(covariance / np.var(benchmark_returns, ddof=0))


def sharpe_ratio(values: list[float], periodic_risk_free_rate: float) -> float:
    """Mean excess return per unit of volatility."""
    returns = daily_returns(values)
    return float((np.mean(returns) - periodic_risk_free_rate) / volatility(values))


def intervals_per_year(interval: str, trading_days: int = 252, trading_hours: float = 6.5) -> int:
    """Number of bars of ``interval`` size in a trading year, for annualizing metrics."""
    if interval.endswith("m"):
        minutes = int(interval[:-1])
        intervals_per_day = (trading_hours * 60) // minutes
    elif interval.endswith("h"):
        hours = int(interval[:-1])
        intervals_per_day = trading_hours // hours
    elif interval == "1d":
        intervals_per_day = 1
    else:
        raise ValueError(f"Unsupported interval: {interval}")
    return int(intervals_per_day * trading_days)


def evaluate(name: str, values: list[float], benchmark: list[float], annual_rf: float, interval: str) -> dict:
    """Compute the full metrics set for a strategy against a benchmark."""
    n = intervals_per_year(interval)
    periodic_rf = (1 + annual_rf) ** (1 / n) - 1
    return {
        "strategy": name,
        "total_return_pct": total_return(values),
        "excess_return_pct": excess_return(values, benchmark),
        "annualized_volatility": volatility(values) * np.sqrt(n),
        "max_drawdown_pct": max_drawdown(values),
        "beta": beta(values, benchmark),
        "sharpe_ratio": sharpe_ratio(values, periodic_rf),
        "final_value": values[-1],
    }
