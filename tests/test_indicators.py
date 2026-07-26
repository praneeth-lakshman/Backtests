import pandas as pd
import pytest

from backtests.indicators import rsi, sma


def test_sma_basic_window():
    close = pd.Series([1, 2, 3, 4, 5], dtype=float)
    result = sma(close, window=3)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].tolist() == pytest.approx([2.0, 3.0, 4.0])


def test_rsi_matches_hand_computed_wilder_values():
    # delta:  nan, 1, -2, 1, 2
    close = pd.Series([10, 11, 9, 10, 12], dtype=float)
    result = rsi(close, period=2)

    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].tolist() == pytest.approx([33.333333, 60.0, 84.615385], rel=1e-5)


def test_rsi_is_100_when_there_are_no_losses():
    close = pd.Series([10, 11, 12, 13, 14], dtype=float)
    result = rsi(close, period=2)
    assert result.iloc[2:].tolist() == pytest.approx([100.0, 100.0, 100.0])


def test_rsi_is_neutral_when_price_is_flat():
    close = pd.Series([10.0] * 6)
    result = rsi(close, period=2)
    assert result.iloc[2:].tolist() == pytest.approx([50.0, 50.0, 50.0, 50.0])
