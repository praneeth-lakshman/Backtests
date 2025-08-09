import plotly.graph_objects as go
from Helper import *
from Metrics import *

short_roll = 10
long_roll = 30
ticker = "AAPL"
annual_rf = 0.05
interval = "1d"

stock_data = get_stock_data(ticker=ticker, period="2y", interval=interval)

portfolio = Portfolio(initial_money=1000, share_holdings={
    ticker: 0
})


buy_hold_values = buy_hold(ticker=ticker, stock_data=stock_data, portfolio=portfolio)
generate_metrics("Buying and holding", buy_hold_values, annual_rf, buy_hold_values, interval)
portfolio.reset()

momentum_values = momentum(ticker=ticker, portfolio=portfolio, stock_data=stock_data, short_roll=short_roll, long_roll=long_roll, first_shares=1)
generate_metrics("Momentum", momentum_values, annual_rf, buy_hold_values, interval)
portfolio.reset()

short_long_values = cross(stock_data=stock_data, portfolio=portfolio, ticker=ticker, short_roll=short_roll, long_roll=long_roll)
generate_metrics("Cross (SMA)", short_long_values, annual_rf, buy_hold_values, interval)
portfolio.reset()

price_short_values = price_vs_short(stock_data=stock_data, portfolio=portfolio, ticker=ticker, short_roll=short_roll)
generate_metrics("Price relative to short trend", price_short_values, annual_rf, buy_hold_values, interval)
portfolio.reset()

cross_ema_values = cross_EMA(stock_data=stock_data, ticker=ticker, portfolio=portfolio, short_roll=short_roll, long_roll=long_roll)
generate_metrics("Cross (EMA)", cross_ema_values, annual_rf, buy_hold_values, interval)
portfolio.reset()

fig = go.Figure(
    data=[
        go.Scatter(
            x=stock_data.index,
            y=buy_hold_values,
            name="buying and holding"
        )
    ]
)
fig.add_trace(
    go.Scatter(
        x=stock_data.index,
        y=momentum_values,
        name=f"Both SMA{short_roll}/{long_roll} Up"
    )
)

fig.add_trace(
    go.Scatter(
        x=stock_data.index,
        y=short_long_values,
        name="Cross method"
    )
)

fig.add_trace(
    go.Scatter(
        x=stock_data.index,
        y=price_short_values,
        name="Price over short"
    )
)

fig.add_trace(
    go.Scatter(
        x=stock_data.index,
        y=cross_ema_values,
        name="EMA"
    )
)


fig.write_html("Test.html")