"""A minimal single- or multi-asset paper-trading portfolio."""

from __future__ import annotations

import copy


class Portfolio:
    """Tracks cash and share holdings for one or more tickers."""

    def __init__(self, initial_money: float, share_holdings: dict[str, int]):
        self.money = self.initial_money = initial_money
        self.portfolio = copy.deepcopy(share_holdings)
        self.initial_holdings = copy.deepcopy(share_holdings)

    def sell(self, ticker: str, shares: int, price: float) -> None:
        if shares > 0 and self.portfolio[ticker] >= shares:
            self.portfolio[ticker] -= shares
            self.money += price * shares

    def buy(self, ticker: str, shares: int, price: float) -> None:
        if shares > 0 and price * shares <= self.money:
            self.portfolio[ticker] += shares
            self.money -= price * shares

    def get_value(self, ticker: str, price: float) -> float:
        return self.portfolio[ticker] * price + self.money

    def reset(self) -> None:
        self.money = self.initial_money
        self.portfolio = copy.deepcopy(self.initial_holdings)
