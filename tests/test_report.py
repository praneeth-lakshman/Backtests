import pandas as pd
import plotly.graph_objects as go

from backtests.report import annotate_best, build_comparison_figure, results_dataframe


def test_build_comparison_figure_has_one_trace_per_series():
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    series = {
        "Buy and Hold": [100, 110, 120],
        "Momentum": [100, 105, 130],
    }
    fig = build_comparison_figure(index, series, ticker="AAA", interval="1d")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert [trace.name for trace in fig.data] == ["Buy and Hold", "Momentum"]
    assert list(fig.data[0].y) == [100, 110, 120]
    assert "AAA" in fig.layout.title.text


def test_build_comparison_figure_marks_the_highlighted_trace():
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    series = {
        "Buy and Hold": [100, 110, 120],
        "Momentum": [100, 105, 130],
    }
    fig = build_comparison_figure(index, series, ticker="AAA", interval="1d", highlight="Momentum")

    by_name = {trace.name: trace for trace in fig.data}
    assert "Momentum" in list(by_name)[1]  # trophy-prefixed name still contains "Momentum"
    momentum_trace = [t for t in fig.data if "Momentum" in t.name][0]
    buy_hold_trace = [t for t in fig.data if t.name == "Buy and Hold"][0]

    assert momentum_trace.line.width > buy_hold_trace.line.width
    assert momentum_trace.opacity == 1.0
    assert buy_hold_trace.opacity < 1.0


def test_results_dataframe_renames_columns():
    results = [{"strategy": "A", "total_return_pct": 5.0, "final_value": 1050.0}]
    df = results_dataframe(results)
    assert list(df.columns) == ["Strategy", "Total return %", "Final value $"]


def test_annotate_best_marks_only_the_winning_row():
    df = pd.DataFrame({"Strategy": ["A", "B"], "Final value $": [1000, 1200]})
    marked = annotate_best(df, "B")
    assert marked.loc[marked["Strategy"].str.endswith("B"), "Strategy"].iloc[0] != "B"
    assert marked.loc[0, "Strategy"] == "A"
