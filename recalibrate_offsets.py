"""
Recalibrate WU station offsets from forecast_logs.

Each city's forecast log now has 155+ paired (forecast, actual) observations.
Use these to compute an updated wu_offset per city. Cities with METAR-sourced
actuals have higher-confidence offsets because METAR matches what Weather
Underground (Polymarket's resolver) republishes.

Usage:
  python recalibrate_offsets.py             # print new offsets
  python recalibrate_offsets.py --apply      # update forecast_logger.py CITIES
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from forecast_logger import CITIES, load_forecast_log, is_sane_temp

LOG_DIR = Path(__file__).parent / "forecast_logs"


def compute_offset(slug, unit, min_pairs=20):
    """Compute per-city bias (mean actual - forecast) and return (offset, n, source)."""
    log = load_forecast_log(slug)

    # Prefer METAR-sourced actuals (match resolution source)
    metar_errors = []
    om_errors = []

    for entry in log:
        forecast = entry.get("forecast")
        actual = entry.get("actual")
        source = entry.get("actual_source")
        if forecast is None or actual is None:
            continue
        if not is_sane_temp(actual, unit):
            continue
        # Filter corrupt entries
        max_err = 15.0 if unit == "F" else 10.0
        if abs(actual - forecast) > max_err:
            continue

        if source == "metar_iem":
            metar_errors.append(actual - forecast)
        elif source == "open_meteo_archive":
            om_errors.append(actual - forecast)
        else:
            om_errors.append(actual - forecast)  # unknown source

    # Use METAR if available and sufficient, else fall back to Open-Meteo
    if len(metar_errors) >= min_pairs:
        offset = round(float(np.mean(metar_errors)), 1)
        source_flag = "metar"
        n = len(metar_errors)
    elif len(om_errors) >= min_pairs:
        offset = round(float(np.mean(om_errors)), 1)
        source_flag = "open-meteo"
        n = len(om_errors)
    else:
        return None, 0, "insufficient"

    return offset, n, source_flag


def main():
    dry_run = "--apply" not in sys.argv
    if not dry_run:
        print("=== APPLYING — updating forecast_logger.py ===\n")

    results = []
    for city_name, city in CITIES.items():
        offset, n, source = compute_offset(city["slug"], city["unit"])
        old_offset = city.get("wu_offset", 0)
        results.append((city_name, city["slug"], city["unit"],
                        old_offset, offset, n, source))

    # Print table
    print(f"{'City':<18s} {'Unit':>4s} {'Old':>8s} {'New':>8s} {'N':>5s} {'Source':>12s}")
    print("-" * 65)
    for name, slug, unit, old, new, n, src in results:
        old_str = f"{old:+.1f}" if old is not None else "N/A"
        new_str = f"{new:+.1f}" if new is not None else "N/A"
        change = ""
        if old is not None and new is not None and abs(new - old) >= 0.5:
            change = " ⚡"
        print(f"  {name:<16s} {unit:>4s} {old_str:>8s} {new_str:>8s} {n:>5d} {src:>12s}{change}")

    print()
    print(f"{len([r for r in results if r[4] is not None])} cities have offsets, "
          f"{len([r for r in results if r[4] is None])} insufficient data")
    print(f"⚡ = changed by ≥0.5°")

    if dry_run:
        print("\nRe-run with --apply to update forecast_logger.py CITIES dict.")
    else:
        # Read forecast_logger.py and update the CITIES dict
        logger_path = Path(__file__).parent / "forecast_logger.py"
        content = logger_path.read_text()

        for name, slug, unit, old, new, n, src in results:
            if new is None:
                continue
            # Replace the wu_offset value for this city
            line_prefix = f'    "{name}":'
            old_pattern = f'"wu_offset": {old}'
            new_pattern = f'"wu_offset": {new}'
            new_py = f'"wu_offset": {new:+}'
            if new >= 0:
                new_py = f'"wu_offset": +{new}'
            else:
                new_py = f'"wu_offset": {new}'

        # Easier: reconstruct the full CITIES block
        # Actually, let's do individual replacements
        for name, slug, unit, old, new, n, src in results:
            if new is None:
                continue
            # Find the line for this city and update wu_offset
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if f'"{name}":' in line and '"slug"' in line:
                    # The wu_offset is on this line
                    import re
                    lines[i] = re.sub(
                        r'"wu_offset":\s*[-+]?\d+\.?\d*',
                        f'"wu_offset": {new:+}',
                        lines[i]
                    )
            content = '\n'.join(lines)

        logger_path.write_text(content)
        print("Updated forecast_logger.py CITIES dict.")

        # Also update the cache so compute_ensemble_high picks up new offsets
        # (wu_offset isn't used in compute_ensemble_high — it's applied in
        # paper_trader.py after the ensemble is computed. So no cache flush needed.)


if __name__ == "__main__":
    main()
