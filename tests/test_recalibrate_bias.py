"""Tests for the bias re-measurement tool.

recalibrate_bias.py exists so the forecast bias does not silently rot: as the
model vendors push new ensemble versions, the +2.0 F correction becomes wrong.
This tool re-derives it from resolved trades.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import recalibrate_bias as rb


def _trade(city, unit, forecast, outcome_bin, resolved=True, pnl=-10.0):
    return {
        "city": city, "unit": unit, "forecast_high": forecast,
        "outcome_bin": outcome_bin, "resolved": resolved, "pnl": pnl,
    }


def test_outcome_low_parses_a_plain_range_label():
    assert rb.outcome_low("102-103°F") == 102.0
    assert rb.outcome_low("98-99°F") == 98.0


def test_outcome_low_parses_open_ended_labels():
    """'or higher' / 'or below' bins are real Polymarket labels."""
    assert rb.outcome_low("37°C or higher") == 37.0
    assert rb.outcome_low("30°C or below") == 30.0


def test_outcome_low_returns_none_for_unparseable_input():
    """A missing label must be None, never 0.0 — 0.0 would masquerade as data."""
    assert rb.outcome_low("") is None
    assert rb.outcome_low(None) is None


def test_bias_is_forecast_minus_actual():
    """Positive bias means the forecast ran hot."""
    trades = [
        _trade("Dallas", "F", 102.0, "98-99°F"),   # +4
        _trade("Miami", "F", 92.0, "90-91°F"),     # +2
    ]
    assert rb.measure_bias(trades, "F") == pytest.approx(3.0)


def test_measure_bias_separates_units():
    """F and C cities must be measured on their own scales, not pooled —
    pooling silently compares Fahrenheit degrees to Celsius degrees."""
    trades = [
        _trade("Dallas", "F", 102.0, "98-99°F"),   # +4 F
        _trade("Paris", "C", 33.0, "30-31°C"),     # +3 C
    ]
    assert rb.measure_bias(trades, "F") == pytest.approx(4.0)
    assert rb.measure_bias(trades, "C") == pytest.approx(3.0)


def test_measure_bias_ignores_unresolved_trades():
    """An open trade has no outcome, so it cannot contribute a bias."""
    trades = [
        _trade("Dallas", "F", 102.0, "98-99°F"),
        _trade("Dallas", "F", 102.0, None, resolved=False),
    ]
    assert rb.measure_bias(trades, "F") == pytest.approx(4.0)


def test_measure_bias_returns_none_when_there_is_no_data():
    """No usable trades must be None, not 0.0 — 0.0 would read as
    'perfectly calibrated' and silently disable the correction."""
    assert rb.measure_bias([], "F") is None
    assert rb.measure_bias([_trade("X", "F", 100.0, "garbage")], "F") is None


def test_suggest_rounds_to_something_a_human_can_paste_into_config():
    trades = [_trade("Dallas", "F", 102.0, "98-99°F") for _ in range(3)]
    s = rb.suggest(trades)
    assert s["F"] == pytest.approx(4.0)
    assert s["F"] == round(s["F"], 1)
