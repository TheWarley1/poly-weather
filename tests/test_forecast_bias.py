"""Tests for the forecast bias correction.

Context: 58 resolved paper trades (Sep 2026) showed the ensemble forecast ran
+1.98 F hot (median +1.90), and the miss was one-directional — the actual high
landed BELOW the traded bin 54 times out of 58. The probability model centres
its distribution on the raw forecast, so a hot forecast shifts the whole
distribution above reality and the model confidently buys bins the temperature
never reaches.

These tests pin the correction behaviour. Run: pytest tests/ -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paper_trader as pt


def test_bias_correction_lowers_a_hot_forecast_fahrenheit():
    """A +2.0 F hot forecast must come down by the F correction."""
    out = pt.apply_forecast_bias(102.0, "F", bias_f=2.0, bias_c=1.1)
    assert out == 100.0


def test_bias_correction_lowers_a_hot_forecast_celsius():
    """Celsius cities use their own correction, not the Fahrenheit one."""
    out = pt.apply_forecast_bias(34.0, "C", bias_f=2.0, bias_c=1.1)
    assert out == pytest.approx(32.9)


def test_bias_correction_is_a_no_op_when_bias_is_zero():
    """Bias 0.0 must leave the forecast untouched (the revert path)."""
    assert pt.apply_forecast_bias(102.0, "F", bias_f=0.0, bias_c=0.0) == 102.0


def test_model_peak_bin_moves_down_when_bias_is_applied():
    """The whole point: correcting the forecast must move the model's
    most-likely bin downward, so it stops buying bins the high never reaches.

    This is the behaviour that failed in production — the model liked the bin
    centred on a forecast that ran 2F hot.
    """
    bins = [f"{lo}-{lo+1}" for lo in range(95, 106)]

    raw = pt.build_model_probabilities(102.0, bins, "F", lead_days=1)
    corrected_input = pt.apply_forecast_bias(102.0, "F", bias_f=2.0, bias_c=1.1)
    corrected = pt.build_model_probabilities(corrected_input, bins, "F", lead_days=1)

    def peak(d):
        return max(d, key=d.get)

    def low(label):
        return int(label.split("-")[0])

    assert low(peak(corrected)) < low(peak(raw)), (
        "bias correction did not move the peak bin down: "
        f"raw={peak(raw)} corrected={peak(corrected)}"
    )
    # Bins span [lo-0.5, hi+0.5], so the bin containing a value v is the one
    # whose label straddles it. Allow the peak to sit on either straddling bin
    # rather than pinning an exact label.
    assert abs(low(peak(corrected)) - corrected_input) <= 1.0, (
        f"peak bin {peak(corrected)} is not near the corrected forecast "
        f"{corrected_input}"
    )


def test_correction_uses_celsius_scale_for_celsius_cities():
    """A Celsius city must not be corrected with the Fahrenheit value."""
    bins = [f"{lo}-{lo+1}" for lo in range(28, 38)]

    f_bias_input = pt.apply_forecast_bias(34.0, "C", bias_f=2.0, bias_c=1.1)
    assert f_bias_input == 32.9, "Celsius city was corrected on the F scale"

    probs = pt.build_model_probabilities(f_bias_input, bins, "C", lead_days=1)
    peak = max(probs, key=probs.get)
    assert int(peak.split("-")[0]) in (32, 33), (
        f"peak bin {peak} is not centred on the corrected 32.9 C"
    )


def test_config_declares_both_bias_keys():
    """Both units need a key. A missing one silently means 'no correction'
    for that unit, which is how the bug would come back unnoticed."""
    assert "forecast_bias_f" in pt.CONFIG
    assert "forecast_bias_c" in pt.CONFIG
    assert pt.CONFIG["forecast_bias_f"] > 0
    assert pt.CONFIG["forecast_bias_c"] > 0


def test_config_bias_is_below_the_absurd_threshold():
    """Guard rail: a correction larger than a bin is a sign of a units mixup,
    not a calibration. Bins are ~2 F / ~1 C wide."""
    assert pt.CONFIG["forecast_bias_f"] < 5.0, "F bias implausibly large"
    assert pt.CONFIG["forecast_bias_c"] < 3.0, "C bias implausibly large"


def test_default_bias_comes_from_config_when_not_passed():
    """The pipeline calls apply_forecast_bias without explicit values, so the
    CONFIG default must actually take effect."""
    expected = round(100.0 - pt.CONFIG["forecast_bias_f"], 1)
    assert pt.apply_forecast_bias(100.0, "F") == expected
