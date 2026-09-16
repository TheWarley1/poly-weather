#!/usr/bin/env python3
"""Re-measure the ensemble forecast bias from resolved paper trades.

Why this exists
---------------
paper_trader.py applies a fixed bias correction (CONFIG["forecast_bias_f" /
"forecast_bias_c"], PATCH N) to remove a systematic hot-forecast error.
That number is empirical. It was +1.98 F / +1.52 C when measured on the
first 58 resolved trades. Model vendors push new ensemble versions, so the
number will drift — a stale correction is worse than none, because it looks
deliberate.

Run this periodically and paste the suggested values into CONFIG:

    python recalibrate_bias.py               # human-readable summary
    python recalibrate_bias.py --suggest     # just the CONFIG lines to paste
    python recalibrate_bias.py --json        # machine-readable

Read-only: this never writes to paper_trades.json or CONFIG.
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

TRADES_FILE = Path(__file__).parent / "paper_trades.json"

# Minimum resolved trades before a suggested value is trustworthy. Below this
# the sample is too small to tell a bias from a bad week.
MIN_SAMPLE = 20


def outcome_low(label):
    """Low bound of an outcome bin label, in the unit's own scale.

    Returns None when the label carries no number. Never returns 0.0 for
    unparseable input — 0.0 would enter the mean as if it were a real
    temperature and drag the bias somewhere meaningless.
    """
    if label is None:
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)", str(label))
    return float(m.group(1)) if m else None


def measure_bias(trades, unit):
    """Mean (forecast - actual) for resolved trades of one unit.

    Positive means the forecast ran hot. Returns None when there is nothing
    usable, so callers can distinguish "no data" from "no bias".
    """
    diffs = []
    for t in trades:
        if not t.get("resolved"):
            continue
        if t.get("unit") != unit:
            continue
        fc = t.get("forecast_high")
        actual = outcome_low(t.get("outcome_bin"))
        if fc is None or actual is None:
            continue
        diffs.append(fc - actual)
    if not diffs:
        return None
    return statistics.mean(diffs)


def suggest(trades):
    """Bias suggestion per unit, rounded to one decimal for pasting into CONFIG."""
    out = {}
    for unit in ("F", "C"):
        b = measure_bias(trades, unit)
        if b is not None:
            out[unit] = round(b, 1)
    return out


def _report(trades):
    print("=" * 60)
    print("  FORECAST BIAS RE-MEASUREMENT")
    print("=" * 60)
    print(f"  trades file : {TRADES_FILE}")
    total = len(trades)
    resolved = [t for t in trades if t.get("resolved")]
    print(f"  trades      : {total} total, {len(resolved)} resolved")
    print()

    for unit in ("F", "C"):
        b = measure_bias(trades, unit)
        n = sum(
            1 for t in resolved
            if t.get("unit") == unit
            and t.get("forecast_high") is not None
            and outcome_low(t.get("outcome_bin")) is not None
        )
        label = "Fahrenheit" if unit == "F" else "Celsius"
        if b is None:
            print(f"  {label:<11}: no usable trades")
            continue
        flag = "" if n >= MIN_SAMPLE else f"  <- only n={n}, treat as provisional"
        print(f"  {label:<11}: {b:+.2f} over {n} trades{flag}")

        # one-directional check: a bias should not be a coin flip
        diffs = [
            t["forecast_high"] - outcome_low(t["outcome_bin"])
            for t in resolved
            if t.get("unit") == unit
            and t.get("forecast_high") is not None
            and outcome_low(t.get("outcome_bin")) is not None
        ]
        if diffs:
            hot = sum(1 for d in diffs if d > 0)
            print(f"  {'':<11}  {hot}/{len(diffs)} ran hot "
                  f"({hot / len(diffs) * 100:.0f}%)")
    print()
    s = suggest(trades)
    if s:
        print("  Current CONFIG (paper_trader.py):")
        print(f"    \"forecast_bias_f\": {s.get('F', 'n/a')},")
        print(f"    \"forecast_bias_c\": {s.get('C', 'n/a')},")
    else:
        print("  Nothing to suggest — no resolved trades with usable outcomes.")
    print()


def main():
    ap = argparse.ArgumentParser(description="Re-measure ensemble forecast bias")
    ap.add_argument("--suggest", action="store_true",
                    help="print only the CONFIG lines to paste")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--file", default=str(TRADES_FILE), help="path to trades json")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"error: trades file not found: {path}", file=sys.stderr)
        return 1
    try:
        trades = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"error: could not parse {path}: {e}", file=sys.stderr)
        return 1
    if not isinstance(trades, list):
        print(f"error: expected a list of trades in {path}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"bias": suggest(trades), "n": len(trades)}, indent=2))
        return 0

    if args.suggest:
        s = suggest(trades)
        for unit, val in s.items():
            print(f'"forecast_bias_{unit.lower()}": {val},')
        return 0

    _report(trades)
    return 0


if __name__ == "__main__":
    sys.exit(main())
