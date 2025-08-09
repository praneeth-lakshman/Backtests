import plotly.graph_objects as go
import yfinance as yf
import numpy as np
import pandas as pd
import copy

def get_stock_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    stock_data = stock.history(period=period, interval=interval)
    stock_data.index = stock_data.index.tz_localize(None)
    return stock_data

class Portfolio:
    def __init__(self, initial_money: float, share_holdings: dict):
        self.money = self.initial_money = initial_money
        self.portfolio = copy.deepcopy(share_holdings)
        self.initial_holdings = copy.deepcopy(share_holdings)
    def sell(self, ticker: str, shares: int, price: float):
        if self.portfolio[ticker] >= shares and shares != 0:
            self.portfolio[ticker] -= shares
            self.money += price * shares
    def buy(self, ticker: str, shares: int, price: float):
        if price * shares <= self.money and shares != 0:
            self.portfolio[ticker] += shares
            self.money -= price * shares
    def add(self, ticker: str, shares: int):
        self.portfolio[ticker] += shares
    def subtract(self, ticker: str, shares: int):
        self.portfolio[ticker] -= shares
    def get_value(self, ticker, price):
        return self.portfolio[ticker] * price + self.money
    def reset(self):
        self.money = self.initial_money
        self.portfolio = copy.deepcopy(self.initial_holdings)


class Strategy:
    def __init__(self, stock_data: pd.DataFrame, portfolio: Portfolio, ticker: str, short_roll: int, long_roll: int, first_shares: int, rsi_period=14, rsi_oversold=30, rsi_overbought=70):
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
        self.short_roll_values = self._generate_derivative_cols(roll=short_roll)
        self.long_roll_values = self._generate_derivative_cols(roll=long_roll)
        self.rsi_values = self._calculate_RSI(period=rsi_period)
        
    @staticmethod
    def reset(func):
        def wrapper(self, *args, **kwargs):
            self.portfolio.reset()
            result = func(self, *args, **kwargs)
            return result
        return wrapper
    
    # == Helper Functions ==
    def _calculate_RSI(self, period) -> pd.Series:
        delta = self.stock_data['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # Use Wilder's smoothing method
        for i in range(period, self.n):
            if i == period:
                # first average calculated already by rolling mean
                continue
            else:
                avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
                avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _generate_SMA(self, roll: int):
        return self.stock_data.Close.rolling(window=roll).mean()

    @staticmethod
    def _differentiate(current_value, previous_value, interval):
        return (current_value - previous_value) / interval

    @staticmethod
    def _subtract(time1, time2):
        return (time1.timestamp() - time2.timestamp())

    def _generate_derivative_cols(self, roll: int):
        vals = []
        roll_values = self._generate_SMA(roll=roll)
        for i in range(len(roll_values)):
            if i == 0:
                val = np.nan
            else:
                val = self._differentiate(roll_values.iloc[i], roll_values.iloc[i-1], self._subtract(self.stock_data.index[i], self.stock_data.index[i-1]))
            vals.append(val)
        return vals


    def _generate_EMA(self, roll: int):
        return self.stock_data.Close.ewm(span=roll).mean()
    
    # == Strategies ==

    @reset    
    def buy_hold(self):
        max_num_of_shares = self.portfolio.money // self.stock_data.Open.iloc[0]
        price = self.stock_data.Open.iloc[0]
        self.portfolio.buy(ticker=self.ticker, shares=max_num_of_shares, price=price)

        baseline = [np.nan] * self.n
        for i in range(self.n):
            if i == 0:
                price = self.stock_data.Open.iloc[i]
                baseline[i] = (self.portfolio.get_value(ticker=self.ticker, price=price))
            else:
                price = self.stock_data.Close.iloc[i]
                baseline[i] = (self.portfolio.get_value(ticker=self.ticker, price=price))
        return baseline
    
    @reset
    def momentum(self):
        value_of_portfolio = [np.nan] * self.n
        for i in range(self.n):
            price = self.stock_data.Close.iloc[i]
            max_shares_buy = int(self.portfolio.money // price)
            max_shares_sell = self.portfolio.portfolio[self.ticker]
            rsi = self.rsi_values.iloc[i]

            buy_scale = max(0.1, (self.rsi_overbought - rsi) / self.rsi_overbought)
            sell_scale = max(0.1, (rsi - self.rsi_oversold) / self.rsi_oversold)

            scaled_buy_shares = max(1, int(max_shares_buy * buy_scale))
            scaled_sell_shares = max(1, int(max_shares_sell * sell_scale))

            if i <= 50 and i % 10 == 0:
                self.portfolio.buy(ticker=self.ticker, shares=self.first_shares, price=price)

            if self.long_roll_values[i] > 0:
                if self.short_roll_values[i] > 0:
                    if scaled_buy_shares > 0:
                        self.portfolio.buy(ticker=self.ticker, shares=scaled_buy_shares, price=price)
                    else:
                        self.portfolio.add(ticker=self.ticker, shares=0)
                else:
                    self.portfolio.add(ticker=self.ticker, shares=0)

            if self.long_roll_values[i] <= 0:
                if self.short_roll_values[i] > 0:
                    self.portfolio.add(ticker=self.ticker, shares=0)
                else:
                    if scaled_sell_shares > 0:
                        self.portfolio.sell(ticker=self.ticker, shares=scaled_sell_shares, price=price)
                    else:
                        self.portfolio.add(ticker=self.ticker, shares=0)

            value_of_portfolio[i] = self.portfolio.get_value(ticker=self.ticker, price=price)

        return value_of_portfolio
    
    @reset
    def cross(self):
        value_of_portfolio = [np.nan] * self.n
        short_sma = self._generate_SMA(self.short_roll)
        long_sma  = self._generate_SMA(self.long_roll)
        for i in range(self.n):
            price = self.stock_data.Close.iloc[i]
            max_shares_buy = int(self.portfolio.money // price)
            max_shares_sell = self.portfolio.portfolio[self.ticker]
            rsi = self.rsi_values.iloc[i]

            buy_scale = max(0.1, (self.rsi_overbought - rsi) / self.rsi_overbought)
            sell_scale = max(0.1, (rsi - self.rsi_oversold) / self.rsi_oversold)

            scaled_buy_shares = max(1, int(max_shares_buy * buy_scale))
            scaled_sell_shares = max(1, int(max_shares_sell * sell_scale))
            
            if short_sma.iloc[i] > long_sma.iloc[i] and buy_scale > 0:
                self.portfolio.buy(ticker=self.ticker, shares=scaled_buy_shares, price=price)
            elif short_sma.iloc[i] < long_sma.iloc[i] and sell_scale > 0:
                self.portfolio.sell(ticker=self.ticker, shares=scaled_sell_shares, price=price)
            else:
                self.portfolio.add(ticker=self.ticker, shares=0)
            value_of_portfolio[i] = self.portfolio.get_value(self.ticker, price=price)
        return value_of_portfolio
    
    @reset
    def price_short(self):
        value_of_portfolio = [np.nan] * self.n
        short_sma = self._generate_SMA(self.short_roll)
        for i in range(self.n):
            price = self.stock_data.Close.iloc[i]
            max_shares_buy = int(self.portfolio.money // price)
            max_shares_sell = self.portfolio.portfolio[self.ticker]
            rsi = self.rsi_values.iloc[i]

            buy_scale = max(0.1, (self.rsi_overbought - rsi) / self.rsi_overbought)
            sell_scale = max(0.1, (rsi - self.rsi_oversold) / self.rsi_oversold)

            scaled_buy_shares = max(1, int(max_shares_buy * buy_scale))
            scaled_sell_shares = max(1, int(max_shares_sell * sell_scale))
            
            if price > short_sma.iloc[i] and buy_scale > 0:
                self.portfolio.buy(ticker=self.ticker, shares=scaled_buy_shares, price=price)
            elif price < short_sma.iloc[i] and sell_scale > 0:
                self.portfolio.sell(ticker=self.ticker, shares=scaled_sell_shares, price=price)
            else:
                self.portfolio.add(self.ticker, 0)
            value_of_portfolio[i] = self.portfolio.get_value(ticker=self.ticker, price=price)
        return value_of_portfolio
        
    @reset
    def cross_ema(self):
        value_of_portfolio = [np.nan] * self.n
        short_ema = self._generate_EMA(self.short_roll)
        long_ema = self._generate_EMA(self.long_roll)
        for i in range(self.n):
            price = self.stock_data.Close.iloc[i]
            max_shares_buy = int(self.portfolio.money // price)
            max_shares_sell = self.portfolio.portfolio[self.ticker]
            rsi = self.rsi_values.iloc[i]

            buy_scale = max(0.1, (self.rsi_overbought - rsi) / self.rsi_overbought)
            sell_scale = max(0.1, (rsi - self.rsi_oversold) / self.rsi_oversold)

            scaled_buy_shares = max(1, int(max_shares_buy * buy_scale))
            scaled_sell_shares = max(1, int(max_shares_sell * sell_scale))
            
            if short_ema.iloc[i] > long_ema.iloc[i] and buy_scale > 0:
                self.portfolio.buy(ticker=self.ticker, shares=scaled_buy_shares, price=price)
            elif short_ema.iloc[i] < long_ema.iloc[i] and sell_scale > 0:
                self.portfolio.sell(ticker=self.ticker, shares=scaled_sell_shares, price=price)
            else:
                self.portfolio.add(ticker=self.ticker, shares=0)
            value_of_portfolio[i] = self.portfolio.get_value(self.ticker, price=price)
        return value_of_portfolio         
    
    # === EVALUATION METHODS ===
    
    @staticmethod
    def _total_return(tested_portfolio: list):
        """Calculate total return percentage"""
        return (tested_portfolio[-1] - tested_portfolio[0]) / tested_portfolio[0] * 100
    
    @staticmethod
    def _excess_return(tested_portfolio: list, buy_hold: list):
        """Calculate excess return vs buy and hold"""
        return (Strategy._total_return(tested_portfolio) - Strategy._total_return(buy_hold))
    
    @staticmethod
    def _get_daily_returns(tested_portfolio: list) -> np.array:
        """Calculate daily returns"""
        returns = []
        for i in range(len(tested_portfolio)):
            if i == 0:
                returns.append(0)
            else:
                returns.append(tested_portfolio[i] / tested_portfolio[i-1] - 1)
        return np.array(returns)
    
    @staticmethod
    def _get_volatility(tested_portfolio: list):
        """Calculate volatility (standard deviation of returns)"""
        return np.std(Strategy._get_daily_returns(tested_portfolio))
    
    @staticmethod
    def _max_drawdown(tested_portfolio: list):
        """Calculate maximum drawdown percentage"""
        portfolio_values = np.array(tested_portfolio)
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        return abs(drawdowns.min()) * 100
    
    @staticmethod
    def _beta(tested_portfolio: list, buy_hold: list):
        """Calculate beta vs buy and hold"""
        tested_returns = Strategy._get_daily_returns(tested_portfolio)
        buy_hold_returns = Strategy._get_daily_returns(buy_hold)
        return np.cov(tested_returns, buy_hold_returns)[0, 1] / np.var(buy_hold_returns)
    
    @staticmethod
    def _sharpe_ratio(tested_portfolio: list, adjusted_rf: float):
        """Calculate Sharpe ratio"""
        returns = Strategy._get_daily_returns(tested_portfolio)
        return (np.mean(returns) - adjusted_rf) / Strategy._get_volatility(tested_portfolio)
    
    @staticmethod
    def _intervals_per_year(interval: str, trading_days=252, trading_hours=6.5):
        """Calculate number of intervals per trading year"""
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
    
    def evaluate(self, strategy_name: str, tested_portfolio: list, annual_rf: float, interval: str):
        """
        Evaluate strategy performance vs buy and hold
        
        Args:
            strategy_name: Name of the strategy for display
            tested_portfolio: Portfolio values from strategy
            annual_rf: Annual risk-free rate (e.g., 0.05 for 5%)
            interval: Time interval used (e.g., "1d", "1h", "5m")
        """
        # Get buy and hold benchmark
        buy_hold_portfolio = self.buy_hold()
        
        # Calculate adjusted risk-free rate
        n = self._intervals_per_year(interval=interval)
        adjusted_rf = (1 + annual_rf) ** (1/n) - 1
        
        # Calculate and display metrics
        print(f"Strategy: {strategy_name}")
        print(f"Total return: {self._total_return(tested_portfolio):.2f}%")
        print(f"Excess return: {self._excess_return(tested_portfolio, buy_hold_portfolio):.2f}%")
        print(f"Volatility (annualized): {self._get_volatility(tested_portfolio) * np.sqrt(n):.4f}")
        print(f"Max drawdown: {self._max_drawdown(tested_portfolio):.2f}%")
        print(f"Beta: {self._beta(tested_portfolio, buy_hold_portfolio):.4f}")
        print(f"Sharpe ratio: {self._sharpe_ratio(tested_portfolio, adjusted_rf):.4f}")
        print(f"Buy & Hold return: {self._total_return(buy_hold_portfolio):.2f}%\n")

# Parameters
short_roll = 10
long_roll = 30
ticker = "COIN"
annual_rf = 0.05
interval = "1d"

# Get data and create portfolio
stock_data = get_stock_data(ticker=ticker, period="1y", interval=interval)

portfolio = Portfolio(
    initial_money=1000, 
    share_holdings={ticker: 0}
)

# Create strategy instance
strategy = Strategy(
    stock_data=stock_data,
    portfolio=portfolio,
    ticker=ticker,
    short_roll=short_roll,
    long_roll=long_roll,
    first_shares=1,
    rsi_oversold=30,
    rsi_overbought=70
)

# Run all strategies and evaluate them
print("=== STRATEGY PERFORMANCE ANALYSIS ===\n")

buy_hold_values = strategy.buy_hold()
strategy.evaluate("Buy and Hold", buy_hold_values, annual_rf, interval)

momentum_values = strategy.momentum()
strategy.evaluate("Momentum", momentum_values, annual_rf, interval)

cross_values = strategy.cross()
strategy.evaluate("Cross (SMA)", cross_values, annual_rf, interval)

price_short_values = strategy.price_short()
strategy.evaluate("Price relative to short trend", price_short_values, annual_rf, interval)

cross_ema_values = strategy.cross_ema()
strategy.evaluate("Cross (EMA)", cross_ema_values, annual_rf, interval)

# Create interactive plot
fig = go.Figure()

# Add all strategy traces
fig.add_trace(go.Scatter(
    x=stock_data.index,
    y=buy_hold_values,
    name="Buy and Hold",
    line=dict(width=3, color='blue')
))

fig.add_trace(go.Scatter(
    x=stock_data.index,
    y=momentum_values,
    name=f"Momentum SMA{short_roll}/{long_roll}",
    line=dict(width=2, color='red')
))

fig.add_trace(go.Scatter(
    x=stock_data.index,
    y=cross_values,
    name="Cross (SMA)",
    line=dict(width=2, color='green')
))

fig.add_trace(go.Scatter(
    x=stock_data.index,
    y=price_short_values,
    name="Price over Short",
    line=dict(width=2, color='orange')
))

fig.add_trace(go.Scatter(
    x=stock_data.index,
    y=cross_ema_values,
    name="Cross (EMA)",
    line=dict(width=2, color='purple')
))

# Update layout for better visualization
fig.update_layout(
    title=f"{ticker} Strategy Comparison ({interval} intervals)",
    xaxis_title="Date",
    yaxis_title="Portfolio Value ($)",
    hovermode='x unified',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ),
    template="plotly_white"
)

# Save the plot
fig.write_html("Strategy_Comparison.html")
print("📊 Interactive plot saved as 'Strategy_Comparison.html'")

# Optional: Show final portfolio values summary
print("\n=== FINAL PORTFOLIO VALUES ===")
strategies = {
    "Buy and Hold": buy_hold_values[-1],
    "Momentum": momentum_values[-1],
    "Cross (SMA)": cross_values[-1],
    "Price over Short": price_short_values[-1],
    "Cross (EMA)": cross_ema_values[-1]
}

for name, value in sorted(strategies.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:<25}: ${value:,.2f}")