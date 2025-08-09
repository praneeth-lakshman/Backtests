import numpy as np

def total_return(tested_portfolio: list):
    return (tested_portfolio[-1] - tested_portfolio[0])/tested_portfolio[0] * 100

def excess_return(tested_portfolio: list, buy_hold: list):
    return (total_return(tested_portfolio=tested_portfolio) - total_return(tested_portfolio=buy_hold))

def get_daily_returns(tested_portfolio: list) -> list:
    returns = []
    for i in range(len(tested_portfolio)):
        if i == 0:
            returns.append(0)
        else:
            returns.append(tested_portfolio[i]/tested_portfolio[i-1] - 1)
    return np.array(returns)

def get_volatility(tested_portfolio: list):
    return np.std(get_daily_returns(tested_portfolio=tested_portfolio))

def max_drawdown(tested_portfolio:list):
    portfolio_values = np.array(tested_portfolio)
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max
    return abs(drawdowns.min()) * 100

def beta(tested_portfolio: list, buy_hold: list):
    return np.cov(get_daily_returns(tested_portfolio), get_daily_returns(buy_hold))[0,1] / np.var(get_daily_returns(buy_hold))

def sharpe(tested_portfolio, adjusted_rf):
    return (np.mean(get_daily_returns(tested_portfolio=tested_portfolio)) - adjusted_rf) / get_volatility(tested_portfolio=tested_portfolio)

def intervals_per_year(interval: str, trading_days=252, trading_hours=6.5):
    """
    Calculate number of intervals per trading year.

    Args:
        interval (str): yfinance interval string, e.g. "1m", "5m", "1h", "1d"
        trading_days (int): number of trading days per year
        trading_hours (float): trading hours per day (e.g., 6.5 for US markets)

    Returns:
        int: estimated number of intervals per year
    """

    if interval.endswith('m'):
        minutes = int(interval[:-1])
        intervals_per_day = (trading_hours * 60) // minutes
    elif interval.endswith('h'):
        hours = int(interval[:-1])
        intervals_per_day = trading_hours // hours
    elif interval == "1d":
        intervals_per_day = 1
    else:
        raise ValueError(f"Unsupported interval: {interval}")

    return int(intervals_per_day * trading_days)


def generate_metrics(name, tested_portfolio, annual_rf, buy_hold, interval):
    n = intervals_per_year(interval=interval)
    adjusted_rf = (1+annual_rf) ** (1/n) - 1
    print(f"Strategy: {name}")
    print(f"Total return: {total_return(tested_portfolio=tested_portfolio)}")
    print(f"Excess return: {excess_return(tested_portfolio=tested_portfolio, buy_hold=buy_hold)}")
    print(f"Volatility (adjusted): {get_volatility(tested_portfolio=tested_portfolio) * n}")
    print(f"Max drawdown: {max_drawdown(tested_portfolio=tested_portfolio)}")
    print(f"Beta: {beta(tested_portfolio=tested_portfolio, buy_hold=buy_hold)}")
    print(f"Sharpe ratio: {sharpe(tested_portfolio=tested_portfolio, adjusted_rf=adjusted_rf)} \n")