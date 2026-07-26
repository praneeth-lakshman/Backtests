import pytest

from backtests import metrics


def test_total_return():
    assert metrics.total_return([100, 150]) == pytest.approx(50.0)


def test_excess_return():
    assert metrics.excess_return([100, 150], [100, 120]) == pytest.approx(30.0)


def test_daily_returns():
    result = metrics.daily_returns([100, 200, 100])
    assert result == pytest.approx([0.0, 1.0, -0.5])


def test_max_drawdown():
    assert metrics.max_drawdown([100, 120, 90, 110]) == pytest.approx(25.0)


def test_beta_of_a_series_against_itself_is_one():
    values = [100, 110, 121, 90, 130]
    assert metrics.beta(values, values) == pytest.approx(1.0)


def test_beta_scales_with_amplitude():
    strategy = [100, 110, 121]  # +10% each step
    benchmark = [100, 105, 110.25]  # +5% each step
    assert metrics.beta(strategy, benchmark) == pytest.approx(2.0)


def test_sharpe_ratio_zero_rf():
    values = [100, 110, 121]
    assert metrics.sharpe_ratio(values, periodic_risk_free_rate=0) == pytest.approx(2**0.5)


@pytest.mark.parametrize(
    "interval,expected",
    [
        ("1d", 252),
        ("1h", 1512),
        ("5m", 19656),
    ],
)
def test_intervals_per_year(interval, expected):
    assert metrics.intervals_per_year(interval) == expected


def test_intervals_per_year_rejects_unknown_interval():
    with pytest.raises(ValueError):
        metrics.intervals_per_year("1wk")


def test_evaluate_returns_all_expected_keys():
    values = [100, 110, 121]
    result = metrics.evaluate("Test Strategy", values, values, annual_rf=0.05, interval="1d")
    assert result["strategy"] == "Test Strategy"
    assert result["beta"] == pytest.approx(1.0)
    assert result["excess_return_pct"] == pytest.approx(0.0)
    assert result["final_value"] == 121


def test_best_strategy_picks_highest_final_value():
    results = [
        {"strategy": "A", "final_value": 900},
        {"strategy": "B", "final_value": 1200},
        {"strategy": "C", "final_value": 1100},
    ]
    assert metrics.best_strategy(results) == "B"
