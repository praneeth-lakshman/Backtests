import pandas as pd

class Strategy:
    def __init__(self, ticker, data, portfolio):
        self.ticker = ticker
        self.data = data
        self.portfolio = portfolio

    def generate_signals(self):
        """Returns a pd.Series of +1 (buy), -1 (sell), 0 (hold)"""
        raise NotImplementedError

class EMACross(Strategy):
    def __init__(self, ticker, data, portfolio, short_window, long_window):
        super().__init__(ticker, data, portfolio)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self):
        df = self.data.copy()
        df['ema_short'] = df['Close'].ewm(span=self.short_window, adjust=False).mean()
        df['ema_long'] = df['Close'].ewm(span=self.long_window, adjust=False).mean()
        df['signal'] = 0
        df.loc[df['ema_short'] > df['ema_long'], 'signal'] = 1
        df.loc[df['ema_short'] < df['ema_long'], 'signal'] = -1
        return df['signal']

