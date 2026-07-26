from backtests.portfolio import Portfolio


def test_buy_deducts_cash_and_adds_shares():
    portfolio = Portfolio(initial_money=1000, share_holdings={"AAA": 0})
    portfolio.buy(ticker="AAA", shares=10, price=50)
    assert portfolio.portfolio["AAA"] == 10
    assert portfolio.money == 500


def test_buy_is_a_no_op_when_unaffordable():
    portfolio = Portfolio(initial_money=100, share_holdings={"AAA": 0})
    portfolio.buy(ticker="AAA", shares=10, price=50)
    assert portfolio.portfolio["AAA"] == 0
    assert portfolio.money == 100


def test_buy_is_a_no_op_for_zero_or_negative_shares():
    portfolio = Portfolio(initial_money=100, share_holdings={"AAA": 0})
    portfolio.buy(ticker="AAA", shares=0, price=10)
    portfolio.buy(ticker="AAA", shares=-5, price=10)
    assert portfolio.portfolio["AAA"] == 0
    assert portfolio.money == 100


def test_sell_adds_cash_and_removes_shares():
    portfolio = Portfolio(initial_money=0, share_holdings={"AAA": 10})
    portfolio.sell(ticker="AAA", shares=4, price=25)
    assert portfolio.portfolio["AAA"] == 6
    assert portfolio.money == 100


def test_sell_is_a_no_op_when_holdings_insufficient():
    portfolio = Portfolio(initial_money=0, share_holdings={"AAA": 2})
    portfolio.sell(ticker="AAA", shares=10, price=25)
    assert portfolio.portfolio["AAA"] == 2
    assert portfolio.money == 0


def test_get_value_combines_cash_and_holdings():
    portfolio = Portfolio(initial_money=100, share_holdings={"AAA": 3})
    assert portfolio.get_value(ticker="AAA", price=10) == 130


def test_reset_restores_initial_state():
    portfolio = Portfolio(initial_money=100, share_holdings={"AAA": 0})
    portfolio.buy(ticker="AAA", shares=5, price=10)
    portfolio.reset()
    assert portfolio.money == 100
    assert portfolio.portfolio["AAA"] == 0
