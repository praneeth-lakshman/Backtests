import yfinance as yf
import numpy as np
import pandas as pd

class Portfolio:
    def __init__(self, initial_money: float, share_holdings: dict):
        self.money = self.initial_money = initial_money
        self.portfolio = share_holdings

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

def get_stock_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    stock_data = stock.history(period=period, interval=interval)
    stock_data.index = stock_data.index.tz_localize(None)
    return stock_data

def generate_SMA(stock_data: pd.DataFrame, roll: int):
    return stock_data.Close.rolling(window=roll).mean()