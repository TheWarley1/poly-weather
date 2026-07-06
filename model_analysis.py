"""
Per-Model Forecast Accuracy Analysis
=====================================
Analyzes all forecast_logs/{city}.json files to compute per-model and per-city
error metrics. Answers:

  1. Which models systematically over/under-estimate high temps?
  2. How accurate is each model (MAE, RMSE, bias)?
  3. How often does the actual temp land within the resolution bin?
  4. How does model disagreement (spread) correlate with actual error?
  5. Which cities are the hardest to forecast?

Outputs a detailed table and, optionally, CSV for further exploration.

Usage:
  python model_analysis.py               # print summary
  python model_analysis.py --csv out.csv # also write CSV
  python model_analysis.py --city nyc    # drill into one city
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import re

from forecast_logger import CITIES, load_forecast_log, is_sane_temp

LOG_DIR = Path(__file__).parent / "forecast_logs"


def parse_bin_center(label, unit="F"):
    """Extract the numeric center of a Polymarket bin label. Used to check
    whether the actual temp landed inside the winning bin."""
    label = label.strip()
    nums = re.findall(r'-?\d+', label)
    if not nums:
        return None, None
    nums = [int(n) for n in nums]

    if "or higher" in label.lower() or "or above" in label.lower():
        return nums[0], nums[0] + 30
    if "or below" in label.lower() or "or lower" in label.lower():
        return nums[0] - 30, nums[0]
    if len(nums) >= 2:
        return nums[0], nums[1]
    return nums[0], nums[0]


def actual_in_bin(actual, bin_label, unit="F"):
    """Check if the actual temperature falls within the bin range."""
    lo, hi = parse_bin_center(bin_label, unit)
    if lo is None:
        return None
    return lo <= actual <= hi


def is_entry_plausible(forecast, actual, unit):
    """Reject entries where |error| is astronomically large — almost certainly
    a corrupt actual, not a genuine forecast miss."""
    max_err = 15.0 if unit == "F" else 10.0
    return abs(actual - forecast) < max_err


def compute_model_stats(city_name, city):
    """For one city, compute per-model error metrics across all paired entries."""
    log = load_forecast_log(city["slug"])
    unit = city["unit"]

    # Collect per-model errors and ensemble errors
    model_errors = defaultdict(list)  # model_name -> [error, error, ...]
    ensemble_errors = []
    bin_hits = 0
    bin_total = 0
    actual_in_bounds = 0

    for entry in log:
        forecast = entry.get("forecast")
        actual = entry.get("actual")
        model_highs = entry.get("model_highs", {})
        resolution_bin = entry.get("resolution_bin")

        if forecast is None or actual is None:
            continue
        if not is_sane_temp(actual, unit):
            continue
        if not is_entry_plausible(forecast, actual, unit):
            continue

        # Ensemble error
        ensemble_errors.append(actual - forecast)
        if abs(actual - forecast) <= 3.0 if unit == "F" else 1.5:
            actual_in_bounds += 1

        # Per-model errors
        for model, high in model_highs.items():
            if high is not None:
                model_errors[model].append(actual - high)

        # Resolution bin hit
        if resolution_bin:
            landed = actual_in_bin(actual, resolution_bin, unit)
            if landed is not None:
                bin_total += 1
                if landed:
                    bin_hits += 1

    n = len(ensemble_errors)

    # Build stats rows
    rows = []

    # Ensemble row first
    errs = np.array(ensemble_errors)
    rows.append({
        "source": "ensemble",
        "n": n,
        "bias": round(float(np.mean(errs)), 2) if n else None,
        "mae": round(float(np.mean(np.abs(errs))), 2) if n else None,
        "rmse": round(float(np.sqrt(np.mean(errs ** 2))), 2) if n else None,
        "over_pct": round(float(np.mean(errs > 0) * 100), 1) if n else None,  # actual > forecast = underestimated by model
        "within_3deg_pct": round(float(actual_in_bounds / n * 100), 1) if n else None,
    })

    # Per-model rows
    for model in sorted(model_errors.keys()):
        errs = np.array(model_errors[model])
        m_n = len(errs)
        rows.append({
            "source": f"  {model}",
            "n": m_n,
            "bias": round(float(np.mean(errs)), 2) if m_n else None,
            "mae": round(float(np.mean(np.abs(errs))), 2) if m_n else None,
            "rmse": round(float(np.sqrt(np.mean(errs ** 2))), 2) if m_n else None,
            "over_pct": round(float(np.mean(errs > 0) * 100), 1) if m_n else None,
            "within_3deg_pct": None,
        })

    # Bin hit rate
    bin_hit_pct = round(bin_hits / bin_total * 100, 1) if bin_total else None

    return rows, bin_hit_pct, bin_hits, bin_total, n


def print_all_cities():
    """Print per-model stats for every city, then a global aggregate."""

    all_errors = defaultdict(list)
    all_ensemble = []
    total_bin_hits = 0
    total_bin_total = 0
    grand_n = 0

    print()
    print("=" * 110)
    print("PER-MODEL FORECAST ACCURACY ANALYSIS")
    print("=" * 110)

    for city_name, city in CITIES.items():
        rows, bin_hit_pct, bin_hits, bin_total, n = compute_model_stats(city_name, city)
        if n == 0:
            continue

        unit = city["unit"]
        station = city["station"]
        slug = city["slug"]

        print(f"\n{'─' * 110}")
        print(f"  {city_name} ({station}, {slug}, °{unit}) — {n} paired observations")
        print(f"{'─' * 110}")
        print(f"{'Source':<28s} {'N':>6s} {'Bias':>8s} {'MAE':>8s} {'RMSE':>8s} {'OverEst%':>9s} {'Within3°%':>10s}")
        print("-" * 110)

        for row in rows:
            bias_str = f"{row['bias']:+.2f}" if row['bias'] is not None else "N/A"
            mae_str = f"{row['mae']:.2f}" if row['mae'] is not None else "N/A"
            rmse_str = f"{row['rmse']:.2f}" if row['rmse'] is not None else "N/A"
            over_str = f"{row['over_pct']:.1f}%" if row['over_pct'] is not None else "N/A"
            within_str = f"{row['within_3deg_pct']:.1f}%" if row['within_3deg_pct'] is not None else "—"
            print(f"  {row['source']:<26s} {row['n']:>6d} {bias_str:>8s} {mae_str:>8s} {rmse_str:>8s} {over_str:>9s} {within_str:>10s}")

        # Bin hit rate
        bin_str = f"{bin_hit_pct:.1f}% ({bin_hits}/{bin_total})" if bin_hit_pct is not None else "N/A"
        print(f"  {'Resolution bin hit rate:':<26s} {'':>6s} {'':>8s} {'':>8s} {'':>8s} {'':>9s} {bin_str:>10s}")

        # Accumulate for global
        log = load_forecast_log(city["slug"])
        unit = city["unit"]
        for entry in log:
            forecast = entry.get("forecast")
            actual = entry.get("actual")
            model_highs = entry.get("model_highs", {})
            if forecast is None or actual is None or not is_sane_temp(actual, unit):
                continue
            if not is_entry_plausible(forecast, actual, unit):
                continue
            all_ensemble.append(actual - forecast)
            for model, high in model_highs.items():
                if high is not None:
                    all_errors[model].append(actual - high)

        total_bin_hits += bin_hits
        total_bin_total += bin_total
        grand_n += n

    # ─── GLOBAL AGGREGATE ───
    print(f"\n{'═' * 110}")
    print(f"  GLOBAL AGGREGATE — {grand_n} paired observations across {len(CITIES)} cities")
    print(f"{'═' * 110}")
    print(f"{'Source':<28s} {'N':>6s} {'Bias':>8s} {'MAE':>8s} {'RMSE':>8s} {'OverEst%':>9s} {'Within3°%':>10s}")
    print("-" * 110)

    ens = np.array(all_ensemble)
    within_ens = np.mean(np.abs(ens) <= 3.0) * 100
    print(f"  {'ensemble':<26s} {len(ens):>6d} "
          f"{float(np.mean(ens)):>+8.2f} {float(np.mean(np.abs(ens))):>8.2f} "
          f"{float(np.sqrt(np.mean(ens**2))):>8.2f} "
          f"{float(np.mean(ens > 0) * 100):>9.1f}% {within_ens:>10.1f}%")

    for model in sorted(all_errors.keys()):
        errs = np.array(all_errors[model])
        print(f"  {'  ' + model:<26s} {len(errs):>6d} "
              f"{float(np.mean(errs)):>+8.2f} {float(np.mean(np.abs(errs))):>8.2f} "
              f"{float(np.sqrt(np.mean(errs**2))):>8.2f} "
              f"{float(np.mean(errs > 0) * 100):>9.1f}% {'—':>10s}")

    bin_global = f"{total_bin_hits / total_bin_total * 100:.1f}% ({total_bin_hits}/{total_bin_total})" if total_bin_total else "N/A"
    print(f"  {'Resolution bin hit rate:':<26s} {'':>6s} {'':>8s} {'':>8s} {'':>8s} {'':>9s} {bin_global:>10s}")

    # ─── INTERPRETATION ───
    print(f"\n{'═' * 110}")
    print("  INTERPRETATION GUIDE")
    print(f"{'═' * 110}")
    print("  Bias  : +° = model UNDER-estimates (actual was hotter than forecast)")
    print("          −° = model OVER-estimates  (actual was cooler than forecast)")
    print("  OverEst% : % of days the model underestimated (actual > forecast)")
    print("          >50% = systematic cold bias, <50% = systematic warm bias")
    print("  Within3°% : % of actuals within 3°F (1.5°C) of the ensemble forecast")
    print("  Resolution bin hit rate : % of days actual landed in Polymarket's winning bin")

    # ─── CITY RANKING ───
    print(f"\n{'═' * 110}")
    print("  CITY RANKING BY ENSEMBLE RMSE (hardest → easiest)")
    print(f"{'═' * 110}")

    city_rankings = []
    for city_name, city in CITIES.items():
        log = load_forecast_log(city["slug"])
        errors = []
        unit = city["unit"]
        for e in log:
            f = e.get("forecast"); a = e.get("actual")
            if f is None or a is None or not is_sane_temp(a, unit):
                continue
            if not is_entry_plausible(f, a, unit):
                continue
            errors.append(a - f)
        if len(errors) >= 5:
            city_rankings.append((city_name, city["unit"], len(errors),
                                   float(np.std(errors)),
                                   float(np.sqrt(np.mean(np.array(errors)**2))),
                                   float(np.mean(errors))))

    city_rankings.sort(key=lambda x: x[4], reverse=True)  # by RMSE
    print(f"{'City':<20s} {'Unit':>4s} {'N':>5s} {'RMSE':>8s} {'StdDev':>8s} {'Bias':>8s}")
    print("-" * 60)
    for name, unit, n, std, rmse, bias in city_rankings:
        print(f"  {name:<18s} {unit:>4s} {n:>5d} {rmse:>8.2f} {std:>8.2f} {bias:>+8.2f}")

    print()


def print_single_city(slug):
    """Deep-dive on one city: show every entry with per-model breakdown."""
    city_entry = None
    for name, c in CITIES.items():
        if c["slug"] == slug:
            city_entry = (name, c)
            break
    if city_entry is None:
        print(f"City slug '{slug}' not found. Available: {', '.join(c['slug'] for c in CITIES.values())}")
        return

    city_name, city = city_entry
    log = load_forecast_log(city["slug"])
    unit = city["unit"]

    rows, bin_hit_pct, bin_hits, bin_total, n = compute_model_stats(city_name, city)

    print(f"\n{'=' * 80}")
    print(f"  {city_name} ({city['station']}, °{unit}) — Per-Model Deep Dive")
    print(f"  {n} paired observations, bin hit rate: {bin_hit_pct}% ({bin_hits}/{bin_total})")
    print(f"{'=' * 80}")

    # Print model stats
    print(f"\n{'Source':<24s} {'N':>5s} {'Bias':>8s} {'MAE':>8s} {'RMSE':>8s} {'OverEst%':>9s}")
    print("-" * 80)
    for row in rows:
        print(f"  {row['source']:<22s} {row['n']:>5d} "
              f"{row['bias']:>+8.2f} {row['mae']:>8.2f} {row['rmse']:>8.2f} "
              f"{row['over_pct']:>9.1f}%")

    # Show worst 5 and best 5 individual forecasts
    print(f"\n{'─' * 80}")
    print("  Worst 5 ensemble misses (largest absolute error):")
    print(f"{'─' * 80}")
    print(f"{'Date':<12s} {'Forecast':>10s} {'Actual':>10s} {'Error':>10s} {'ResBin':>20s}")
    print("-" * 80)

    misses = []
    for e in log:
        f = e.get("forecast")
        a = e.get("actual")
        if f is None or a is None:
            continue
        if not is_sane_temp(a, unit):
            continue
        if not is_entry_plausible(f, a, unit):
            continue
        misses.append((e["date"], f, a, a - f, e.get("resolution_bin") or ""))
    misses.sort(key=lambda x: abs(x[3]), reverse=True)

    for date, f, a, err, rbin in misses[:10]:
        print(f"  {date:<12s} {f:>10.1f}° {a:>10.1f}° {err:>+10.1f}° {rbin:>20s}")

    # Model-by-model scatter
    print(f"\n{'─' * 80}")
    print("  Per-entry per-model errors (last 20 paired days):")
    print(f"{'─' * 80}")

    paired = [e for e in log if e.get("forecast") and e.get("actual") and e.get("model_highs")]
    models_seen = set()
    for e in paired:
        models_seen.update(e["model_highs"].keys())
    models_seen = sorted(models_seen)

    header = f"{'Date':<12s} {'EnsErr':>8s}"
    for m in models_seen:
        header += f" {m[:10]:>10s}"
    print(header)
    print("-" * len(header))

    for e in paired[-20:]:
        ens_err = e["actual"] - e["forecast"]
        line = f"  {e['date']:<10s} {ens_err:>+8.1f}"
        for m in models_seen:
            v = e["model_highs"].get(m)
            if v is not None and e["actual"] is not None:
                line += f" {e['actual'] - v:>+10.1f}"
            else:
                line += f" {'—':>10s}"
        print(line)

    print()


def export_csv(path):
    """Write per-model rows for all cities to a CSV."""
    rows_out = []
    for city_name, city in CITIES.items():
        rows, bin_hit_pct, bin_hits, bin_total, n = compute_model_stats(city_name, city)
        for row in rows:
            rows_out.append({
                "city": city_name,
                "slug": city["slug"],
                "station": city["station"],
                "unit": city["unit"],
                "source": row["source"].strip(),
                "n": row["n"],
                "bias": row["bias"],
                "mae": row["mae"],
                "rmse": row["rmse"],
                "over_pct": row["over_pct"],
                "within_3deg_pct": row.get("within_3deg_pct"),
            })
        if bin_hit_pct is not None:
            rows_out.append({
                "city": city_name,
                "slug": city["slug"],
                "station": city["station"],
                "unit": city["unit"],
                "source": "bin_hit_rate",
                "n": bin_total,
                "bias": None,
                "mae": bin_hit_pct,
                "rmse": None,
                "over_pct": None,
                "within_3deg_pct": None,
            })

    with open(path, "w", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=[
            "city", "slug", "station", "unit", "source", "n",
            "bias", "mae", "rmse", "over_pct", "within_3deg_pct"
        ])
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Wrote {len(rows_out)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description="Per-model forecast accuracy analysis")
    parser.add_argument("--city", type=str, help="Deep-dive on a single city slug")
    parser.add_argument("--csv", type=str, help="Export to CSV file")
    parser.add_argument("--top", type=int, default=100, help="Show top N worst misses")
    args = parser.parse_args()

    if args.city:
        print_single_city(args.city)
    else:
        print_all_cities()

    if args.csv:
        export_csv(args.csv)


if __name__ == "__main__":
    main()
