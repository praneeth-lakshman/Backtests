import yfinance as yf
import numpy as np
import pandas as pd
import copy

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
        self.short_roll_values = generate_derivative_cols(stock_data=stock_data, roll=short_roll)
        self.long_roll_values = generate_derivative_cols(stock_data=stock_data, roll=long_roll)
        self.rsi_values = calculate_RSI(stock_data, period=rsi_period)
        self.value_of_portfolio = [np.nan] * self.n
        
    @staticmethod
    def reset(func):
        def wrapper(self, *args, **kwargs):
            self.portfolio.reset()
            result = func(self, *args, **kwargs)
            return result
        return wrapper
    
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
        for i in range(self.n):
            price = self.stock_data.Close.iloc[i]
            max_shares_buy = int(self.portfolio.money // price)
            max_shares_sell = self.portfolio.portfolio[self.ticker]
            rsi = self.rsi_values.iloc[i]

            buy_scale = 0
            sell_scale = 0

            if rsi < self.rsi_oversold:
                buy_scale = (self.rsi_oversold - rsi) / self.rsi_oversold
            elif rsi > self.rsi_overbought:
                sell_scale = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)

            scaled_buy_shares = int(max_shares_buy * buy_scale)
            scaled_sell_shares = int(max_shares_sell * sell_scale)

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

            self.value_of_portfolio[i] = self.portfolio.get_value(ticker=self.ticker, price=price)

        return self.value_of_portfolio
    
    @reset
    def cross(self):
        for i in range(self.n):
            price = self.stock_data.Close.iloc[i]
            max_shares_buy = int(self.portfolio.money // price)
            max_shares_sell = self.portfolio.portfolio[self.ticker]
            rsi = self.rsi_values.iloc[i]

            buy_scale = 0
            sell_scale = 0

            if rsi < self.rsi_oversold:
                buy_scale = (self.rsi_oversold - rsi) / self.rsi_oversold
            elif rsi > self.rsi_overbought:
                sell_scale = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)

            scaled_buy_shares = int(max_shares_buy * buy_scale)
            scaled_sell_shares = int(max_shares_sell * sell_scale)
            
            if self.short_roll_values.iloc[i] > self.long_roll_values.iloc[i] and buy_scale > 0:
                self.portfolio.buy(ticker=self.ticker, shares=scaled_buy_shares, price=price)
            elif self.short_roll_values.iloc[i] < self.long_roll_values.iloc[i] and sell_scale > 0:
                self.portfolio.sell(ticker=self.ticker, shares=scaled_sell_shares, price=price)
            else:
                self.portfolio.add(ticker=self.ticker, shares=0)
            self.value_of_portfolio[i] = self.portfolio.get_value(self.ticker, price=price)
        return self.value_of_portfolio
    
    @reset
    def price_short(self):
        for i in range(self.n):
            price = self.stock_data.Close.iloc[i]
            max_shares_buy = int(self.portfolio.money // price)
            max_shares_sell = self.portfolio.portfolio[self.ticker]
            rsi = self.rsi_values.iloc[i]

            buy_scale = 0
            sell_scale = 0

            if rsi < self.rsi_oversold:
                buy_scale = (self.rsi_oversold - rsi) / self.rsi_oversold
            elif rsi > self.rsi_overbought:
                sell_scale = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)

            scaled_buy_shares = int(max_shares_buy * buy_scale)
            scaled_sell_shares = int(max_shares_sell * sell_scale)
            
            if price > self.short_roll_values.iloc[i] and buy_scale > 0:
                self.portfolio.buy(ticker=self.ticker, shares=scaled_buy_shares, price=price)
            elif price < self.short_roll_values.iloc[i] and sell_scale > 0:
                self.portfolio.sell(ticker=self.ticker, shares=scaled_sell_shares, price=price)
            else:
                self.portfolio.add(self.ticker, 0)
            self.value_of_portfolio[i] = self.portfolio.get_value(ticker=self.ticker, price=price)
        return self.value_of_portfolio
        
    @reset
    def cross_ema(self):
        for i in range(self.n):
            price = self.stock_data.Close.iloc[i]
            max_shares_buy = int(self.portfolio.money // price)
            max_shares_sell = self.portfolio.portfolio[self.ticker]
            rsi = self.rsi_values.iloc[i]

            buy_scale = 0
            sell_scale = 0

            if rsi < self.rsi_oversold:
                buy_scale = (self.rsi_oversold - rsi) / self.rsi_oversold
            elif rsi > self.rsi_overbought:
                sell_scale = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)

            scaled_buy_shares = int(max_shares_buy * buy_scale)
            scaled_sell_shares = int(max_shares_sell * sell_scale)
            
            if self.short_roll_values.iloc[i] > self.long_roll_values.iloc[i] and buy_scale > 0:
                self.portfolio.buy(ticker=self.ticker, shares=scaled_buy_shares, price=price)
            elif self.short_roll_values.iloc[i] < self.long_roll_values.iloc[i] and sell_scale > 0:
                self.portfolio.sell(ticker=self.ticker, shares=scaled_sell_shares, price=price)
            else:
                self.portfolio.add(ticker=self.ticker, shares=0)
            self.value_of_portfolio[i] = self.portfolio.get_value(self.ticker, price=price)
        return self.value_of_portfolio         


def calculate_RSI(stock_data: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = stock_data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Use Wilder's smoothing method
    for i in range(period, len(stock_data)):
        if i == period:
            # first average calculated already by rolling mean
            continue
        else:
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_stock_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    stock_data = stock.history(period=period, interval=interval)
    stock_data.index = stock_data.index.tz_localize(None)
    return stock_data

def generate_SMA(stock_data: pd.DataFrame, roll: int):
    return stock_data.Close.rolling(window=roll).mean()

# finding where the market is trending up or down

# define derivative
def differentiate(current_value, previous_value, interval):
    return (current_value - previous_value) / interval

def subtract(time1, time2):
    return (time1.timestamp() - time2.timestamp())

def generate_derivative_cols(stock_data, roll):
    vals = []
    roll_values = generate_SMA(stock_data=stock_data, roll=roll)
    for i in range(len(roll_values)):
        if i == 0:
            val = np.nan
        else:
            val = differentiate(roll_values.iloc[i], roll_values.iloc[i-1], subtract(stock_data.index[i], stock_data.index[i-1]))
        vals.append(val)
    return vals

def buy_hold(stock_data: pd.DataFrame, portfolio: Portfolio, ticker: str) -> list[float]:
    max_num_of_shares = portfolio.money // stock_data.Open.iloc[0]
    price = stock_data.Open.iloc[0]
    portfolio.buy(ticker=ticker, shares=max_num_of_shares, price=price)

    baseline = [np.nan] * len(stock_data.index)
    for i in range(len(stock_data.index)):
        if i == 0:
            price = stock_data.Open.iloc[i]
            baseline[i] = (portfolio.get_value(ticker=ticker, price=price))
        else:
            price = stock_data.Close.iloc[i]
            baseline[i] = (portfolio.get_value(ticker=ticker, price=price))
    return baseline

def momentum(stock_data: pd.DataFrame, portfolio: Portfolio, ticker: str, short_roll: int, long_roll: int, first_shares: int, rsi_period=14, rsi_oversold=30, rsi_overbought=70) -> list[float]:
    n = len(stock_data)
    short_roll_values = generate_derivative_cols(stock_data=stock_data, roll=short_roll)
    long_roll_values = generate_derivative_cols(stock_data=stock_data, roll=long_roll)
    rsi_values = calculate_RSI(stock_data, period=rsi_period)
    value_of_portfolio = [np.nan] * n

    for i in range(n):
        price = stock_data.Close.iloc[i]
        max_shares_buy = int(portfolio.money // price)
        max_shares_sell = portfolio.portfolio[ticker]
        rsi = rsi_values.iloc[i]

        buy_scale = 0
        sell_scale = 0

        if rsi < rsi_oversold:
            buy_scale = (rsi_oversold - rsi) / rsi_oversold
        elif rsi > rsi_overbought:
            sell_scale = (rsi - rsi_overbought) / (100 - rsi_overbought)

        scaled_buy_shares = int(max_shares_buy * buy_scale)
        scaled_sell_shares = int(max_shares_sell * sell_scale)

        if i <= 50 and i % 10 == 0:
            portfolio.buy(ticker=ticker, shares=first_shares, price=price)

        if long_roll_values[i] > 0:
            if short_roll_values[i] > 0:
                if scaled_buy_shares > 0:
                    portfolio.buy(ticker=ticker, shares=scaled_buy_shares, price=price)
                else:
                    portfolio.add(ticker=ticker, shares=0)
            else:
                portfolio.add(ticker=ticker, shares=0)

        if long_roll_values[i] <= 0:
            if short_roll_values[i] > 0:
                portfolio.add(ticker=ticker, shares=0)
            else:
                if scaled_sell_shares > 0:
                    portfolio.sell(ticker=ticker, shares=scaled_sell_shares, price=price)
                else:
                    portfolio.add(ticker=ticker, shares=0)

        value_of_portfolio[i] = portfolio.get_value(ticker=ticker, price=price)

    return value_of_portfolio


def cross(stock_data: pd.DataFrame, portfolio: Portfolio, ticker: str, short_roll: int, long_roll: int):
    n = len(stock_data)
    short_roll_values = generate_SMA(stock_data=stock_data, roll=short_roll)
    long_roll_values = generate_SMA(stock_data=stock_data, roll=long_roll)
    rsi_values = calculate_RSI(stock_data=stock_data)
    value_of_portfolio = [np.nan] * n

    for i in range(n):
        price = stock_data.Close.iloc[i]
        max_shares_buy = int(portfolio.money // price)
        max_shares_sell = portfolio.portfolio[ticker]
        rsi = rsi_values.iloc[i]

        buy_scale = 0
        sell_scale = 0

        if rsi < 30:
            buy_scale = (30 - rsi) / 30
        elif rsi > 70:
            sell_scale = (rsi - 70) / (100 - 70)

        scaled_buy_shares = int(max_shares_buy * buy_scale)
        scaled_sell_shares = int(max_shares_sell * sell_scale)
        if short_roll_values.iloc[i] > long_roll_values.iloc[i] and buy_scale > 0:
            portfolio.buy(ticker=ticker, shares=scaled_buy_shares, price=price)
        elif short_roll_values.iloc[i] < long_roll_values.iloc[i] and sell_scale > 0:
            portfolio.sell(ticker=ticker, shares=scaled_sell_shares, price=price)
        else:
            portfolio.add(ticker=ticker, shares=0)
        value_of_portfolio[i] = portfolio.get_value(ticker, price=price)
    return value_of_portfolio


def price_vs_short(stock_data: pd.DataFrame, portfolio: Portfolio, ticker: str, short_roll: int) -> list[float]:
    n = len(stock_data)
    short_roll_values = generate_SMA(stock_data=stock_data, roll=short_roll)
    value_of_portfolio = [np.nan] * n
    rsi_values = calculate_RSI(stock_data=stock_data)

    for i in range(n):
        price = stock_data.Close.iloc[i]
        max_shares_buy = int(portfolio.money // price)
        max_shares_sell = portfolio.portfolio[ticker]
        rsi = rsi_values.iloc[i]

        buy_scale = 0
        sell_scale = 0

        if rsi < 30:
            buy_scale = (30 - rsi) / 30
        elif rsi > 70:
            sell_scale = (rsi - 70) / (100 - 70)

        scaled_buy_shares = int(max_shares_buy * buy_scale)
        scaled_sell_shares = int(max_shares_sell * sell_scale)
        if price > short_roll_values.iloc[i] and buy_scale > 0:
            portfolio.buy(ticker=ticker, shares=scaled_buy_shares, price=price)
        elif price < short_roll_values.iloc[i] and sell_scale > 0:
            portfolio.sell(ticker=ticker, shares=scaled_sell_shares, price=price)
        else:
            portfolio.add(ticker, 0)
        value_of_portfolio[i] = portfolio.get_value(ticker=ticker, price=price)
    return value_of_portfolio

def generate_EMA(stock_data: pd.DataFrame, roll: int):
    return stock_data.Close.ewm(span=roll).mean()

def cross_EMA(stock_data: pd.DataFrame, portfolio: Portfolio, short_roll: int, long_roll: int, ticker: str):
    n = len(stock_data)
    short_roll_values = generate_EMA(stock_data=stock_data, roll=short_roll)
    long_roll_values = generate_EMA(stock_data=stock_data, roll=long_roll)
    value_of_portfolio = [np.nan] * n
    rsi_values = calculate_RSI(stock_data=stock_data)

    for i in range(n):
        price = stock_data.Close.iloc[i]
        max_shares_buy = int(portfolio.money // price)
        max_shares_sell = portfolio.portfolio[ticker]
        rsi = rsi_values.iloc[i]

        buy_scale = 0
        sell_scale = 0

        if rsi < 30:
            buy_scale = (30 - rsi) / 30
        elif rsi > 70:
            sell_scale = (rsi - 70) / (100 - 70)

        scaled_buy_shares = int(max_shares_buy * buy_scale)
        scaled_sell_shares = int(max_shares_sell * sell_scale)
        if short_roll_values.iloc[i] > long_roll_values.iloc[i] and buy_scale > 0:
            portfolio.buy(ticker=ticker, shares=scaled_buy_shares, price=price)
        elif short_roll_values.iloc[i] < long_roll_values.iloc[i] and sell_scale > 0:
            portfolio.sell(ticker=ticker, shares=scaled_sell_shares, price=price)
        else:
            portfolio.add(ticker=ticker, shares=0)
        value_of_portfolio[i] = portfolio.get_value(ticker, price=price)
    return value_of_portfolio    


