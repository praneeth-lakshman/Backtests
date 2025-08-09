from dataclasses import dataclass
from typing import Dict

@dataclass
class Position:
    shares: int = 0
    avg_cost: float = 0.0

class Portfolio:
    def __init__(self, initial_cash: float):
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}

    def buy(self, ticker: str, price: float, shares: int):
        if shares <= 0:
            raise ValueError("Shares to buy must be positive")
        cost = price * shares
        if cost > self.cash:
            raise ValueError(f"Not enough cash to buy {shares} shares of {ticker} at {price}")
        
        self.cash -= cost
        pos = self.positions.get(ticker, Position())
        total_shares = pos.shares + shares
        new_avg_cost = ((pos.avg_cost * pos.shares) + (price * shares)) / total_shares
        self.positions[ticker] = Position(shares=total_shares, avg_cost=new_avg_cost)

    def sell(self, ticker: str, price: float, shares: int):
        if shares <= 0:
            raise ValueError("Shares to sell must be positive")
        pos = self.positions.get(ticker)
        if not pos or shares > pos.shares:
            raise ValueError(f"Not enough shares to sell: have {pos.shares if pos else 0}, tried to sell {shares}")
        
        self.cash += price * shares
        remaining_shares = pos.shares - shares
        if remaining_shares == 0:
            del self.positions[ticker]
        else:
            # avg_cost stays the same on partial sale
            self.positions[ticker] = Position(shares=remaining_shares, avg_cost=pos.avg_cost)

    def get_value(self, price_lookup: Dict[str, float]) -> float:
        """Calculate total portfolio value (cash + market value of holdings)."""
        total = self.cash
        for ticker, pos in self.positions.items():
            if ticker in price_lookup:
                total += pos.shares * price_lookup[ticker]
            else:
                # No price available, skip or raise error
                raise ValueError(f"No price for ticker {ticker}")
        return total

    def __repr__(self):
        return f"Portfolio(cash={self.cash}, positions={self.positions})"
