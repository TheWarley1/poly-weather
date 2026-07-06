"""
Brier Backtest — Per-City Trust Score
======================================
Replays the probability model across every paired (forecast, actual) observation
we have logged, and scores how well the model's predicted distribution matched
reality. Produces per-city metrics you can use to decide which cities to trade.

Metrics computed per city:
  - RMSE         Point-forecast accuracy (°F or °C)
  - Bias         Systematic forecast error (mean signed error)
  - σ_emp        Empirical std of forecast errors (the width of the real dist)
  - Brier        Multi-class Brier score over 1°F/°C bins near the forecast.
                 Range 0-2. Lower is better. Baseline (uniform guess) ≈ 0.94.
  - BSS          Brier Skill Score vs climatology baseline.
                 Range (-∞, 1]. >0 = better than climatology, 1 = perfect.
  - PIT_σ        Standard deviation of Probability Integral Transform values.
                 Well-calibrated model → ~0.289 (uniform on [0,1]).
                 <0.289 = model too wide (underconfident)
                 >0.289 = model too narrow (overconfident)
  - OutOfRange   Fraction of actuals that fell >3σ from the corrected forecast.
                 Calibrated gaussian expects ~0.3%. High values (>5%) flag that
                 the σ-estimate is too tight and we'll size trades too aggressively.
  - Top3Hit      Fraction of observations where actual landed in the top-3
                 predicted bins. Simple "did we aim right?" check.

Usage:
  python brier_backtest.py               # all cities, print table
  python brier_backtest.py --csv out.csv # also write CSV
  python brier_backtest.py --city nyc    # debug a single city with details
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import skewnorm

from forecast_logger import CITIES, load_forecast_log, is_sane_temp


# ─────────────────────────────────────────────
# MODEL REPLAY
# ─────────────────────────────────────────────
# This duplicates paper_trader.build_model_probabilities but takes a continuous
# distribution rather than a list of bins — so we can compute PIT and bin
# probabilities on a synthetic grid without caring about what Polymarket bins
# were actually traded that day.

def build_distribution(forecast_high, unit, model_highs=None, empirical_std=None,
                        lead_days=0):
    """Return a scipy skewnorm frozen distribution centered on forecast_high.

    This mirrors paper_trader.build_model_probabilities in spirit: it uses
    empirical_std if provided, else falls back to model spread or a unit default.
    skew_param is left at 0 (unskewed) — the backtest doesn't have access to
    the historical climatology fetch the live path uses, and a symmetric
    distribution is the right null for calibration testing anyway.
    """
    # σ selection — same priority order as the live model
    model_spread = None
    if model_highs:
        vals = [v for v in model_highs.values() if v is not None]
        if len(vals) >= 2:
            model_spread = float(np.std(vals))

    if empirical_std is not None and empirical_std > 0:
        base_std = empirical_std
    elif model_spread is not None and model_spread > 1.0:
        base_std = model_spread * 1.5
    else:
        base_std = 2.5 if unit == "F" else 1.2  # PATCH M: raised from 1.5/0.8

    # PATCH M: scaler widened to 0.8 to match paper_trader.py.
    effective_std = base_std * (1.0 + 0.8 * max(0, lead_days))
    effective_std = max(effective_std, 2.0 if unit == "F" else 1.0)  # PATCH M: raised from 1.0/0.6

    return skewnorm(a=0, loc=forecast_high, scale=effective_std), effective_std


def bins_around(forecast_high, unit, half_width=8):
    """Return list of (lo, hi) integer-bin ranges centered on forecast_high.

    Mirrors Polymarket's typical 1°-wide bins stretching ±8° from center —
    gives 17 bins total. The endpoint bins are open-ended (<lo and >hi) to
    catch tails, matching Polymarket's "X° or lower" / "X° or higher" bins.
    """
    center = int(round(forecast_high))
    bins = []
    # Open-ended low
    bins.append((float("-inf"), center - half_width - 0.5))
    # Closed bins (integer labels)
    for t in range(center - half_width, center + half_width + 1):
        bins.append((t - 0.5, t + 0.5))
    # Open-ended high
    bins.append((center + half_width + 0.5, float("inf")))
    return bins


def bin_index_for_actual(bins, actual):
    for i, (lo, hi) in enumerate(bins):
        if lo <= actual < hi:
            return i
    return len(bins) - 1  # shouldn't happen, but default to last


# ─────────────────────────────────────────────
# PER-CITY SCORING
# ─────────────────────────────────────────────

def leave_one_out_stats(errors, idx):
    """Given a list of signed errors and the index to hold out, return
    (mean, std) of the remainder. Guarantees no data leakage when scoring
    an observation using calibration derived from its own city."""
    pool = [e for i, e in enumerate(errors) if i != idx]
    if len(pool) < 3:
        return None, None
    return float(np.mean(pool)), float(np.std(pool))


def score_city(city_name, city, verbose=False):
    """Walk every paired obs in the log and produce metrics."""
    slug = city["slug"]
    unit = city["unit"]
    log = load_forecast_log(slug)

    # Same filter as paper_trader.scan_city_edges
    max_err = 10.0 if unit == "F" else 6.0
    pairs = []
    for e in log:
        f, a, mh = e.get("forecast"), e.get("actual"), e.get("model_highs")
        if f is None or a is None:
            continue
        if not is_sane_temp(a, unit):
            continue
        if abs(a - f) > max_err:
            continue
        pairs.append({"forecast": float(f), "actual": float(a), "model_highs": mh or {}})

    n = len(pairs)
    if n < 10:
        return {"city": city_name, "n": n, "insufficient": True}

    raw_errors = np.array([p["actual"] - p["forecast"] for p in pairs])

    # Empirical bias and σ for the whole city (for summary). Leave-one-out
    # values are used during scoring to avoid leakage.
    bias = float(raw_errors.mean())
    sigma = float(raw_errors.std())

    # Pre-compute climatology baseline: the distribution of actuals themselves,
    # used as the "naive forecaster" benchmark for BSS.
    actuals = np.array([p["actual"] for p in pairs])
    clim_mean, clim_std = float(actuals.mean()), float(actuals.std())
    clim_dist = skewnorm(a=0, loc=clim_mean, scale=max(clim_std, 1.0 if unit == "F" else 0.6))

    brier_scores = []
    brier_baseline = []
    pit_values = []
    out_of_range = 0
    top3_hits = 0
    rmse_sq = []

    for i, p in enumerate(pairs):
        # LOO calibration — remove this obs when computing bias/σ
        loo_bias, loo_std = leave_one_out_stats(raw_errors, i)
        if loo_std is None:
            continue

        # Apply bias correction to the forecast, same as paper_trader does
        corrected = p["forecast"] + loo_bias

        # Build the predicted distribution
        dist, used_std = build_distribution(
            corrected, unit, model_highs=p["model_highs"], empirical_std=loo_std, lead_days=1
        )

        # Set up bins centered on corrected forecast
        bins = bins_around(corrected, unit, half_width=8)
        probs = np.array([dist.cdf(hi) - dist.cdf(lo) for lo, hi in bins])
        probs = probs / probs.sum() if probs.sum() > 0 else probs

        # Find which bin the actual landed in
        actual_idx = bin_index_for_actual(bins, p["actual"])

        # Multi-class Brier: Σ (p_i - o_i)²
        outcomes = np.zeros(len(bins))
        outcomes[actual_idx] = 1.0
        brier_scores.append(float(np.sum((probs - outcomes) ** 2)))

        # Climatology baseline Brier
        clim_probs = np.array([clim_dist.cdf(hi) - clim_dist.cdf(lo) for lo, hi in bins])
        clim_probs = clim_probs / clim_probs.sum() if clim_probs.sum() > 0 else clim_probs
        brier_baseline.append(float(np.sum((clim_probs - outcomes) ** 2)))

        # PIT: CDF value of the actual under predicted distribution
        pit_values.append(float(dist.cdf(p["actual"])))

        # Out-of-range check: >3σ from forecast
        if abs(p["actual"] - corrected) > 3 * used_std:
            out_of_range += 1

        # Top-3 bin hit
        top3_bins = np.argsort(probs)[-3:]
        if actual_idx in top3_bins:
            top3_hits += 1

        rmse_sq.append((p["actual"] - corrected) ** 2)

    n_scored = len(brier_scores)
    mean_brier = float(np.mean(brier_scores))
    mean_brier_baseline = float(np.mean(brier_baseline))
    bss = 1.0 - (mean_brier / mean_brier_baseline) if mean_brier_baseline > 0 else 0.0

    pit_arr = np.array(pit_values)

    return {
        "city": city_name,
        "slug": slug,
        "n": n,
        "n_scored": n_scored,
        "rmse": float(np.sqrt(np.mean(rmse_sq))),
        "bias": bias,
        "sigma_emp": sigma,
        "brier": mean_brier,
        "brier_baseline": mean_brier_baseline,
        "bss": bss,
        "pit_mean": float(pit_arr.mean()),
        "pit_std": float(pit_arr.std()),
        "pit_ideal_std": 1.0 / np.sqrt(12),  # ≈ 0.289
        "out_of_range_pct": out_of_range / n_scored * 100,
        "top3_hit_pct": top3_hits / n_scored * 100,
        "unit": unit,
        "station": city.get("station", ""),
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def print_summary(results):
    """Pretty-print per-city results ranked by BSS."""
    # Filter out insufficient-data cities
    scored = [r for r in results if not r.get("insufficient")]
    insufficient = [r for r in results if r.get("insufficient")]

    # Sort by BSS descending (best first)
    scored.sort(key=lambda r: r["bss"], reverse=True)

    print(f"\n{'='*120}")
    print(f"PER-CITY BACKTEST — {len(scored)} cities scored")
    print(f"{'='*120}")
    print(f"{'City':<16} {'Stn':<5} {'N':>4} {'RMSE':>6} {'Bias':>6} {'σ':>5} "
          f"{'Brier':>6} {'BSS':>7} {'PIT σ':>6} {'OOR%':>6} {'Top3%':>6}  verdict")
    print('-' * 120)
    ideal_pit = 1.0 / np.sqrt(12)
    for r in scored:
        # Verdict rule of thumb:
        #   BSS > 0.1  and  |PIT_σ - 0.289| < 0.05  and  OOR < 5%   → TRADE
        #   BSS > 0    and  |PIT_σ - 0.289| < 0.10  and  OOR < 8%   → CAUTIOUS
        #   else                                                      → AVOID
        pit_err = abs(r["pit_std"] - ideal_pit)
        if r["bss"] > 0.10 and pit_err < 0.05 and r["out_of_range_pct"] < 5:
            verdict = "✅ TRADE"
        elif r["bss"] > 0 and pit_err < 0.10 and r["out_of_range_pct"] < 8:
            verdict = "⚠️  CAUTION"
        else:
            verdict = "❌ AVOID"

        unit_sym = r["unit"]
        print(f"{r['city']:<16} {r['station']:<5} {r['n']:>4} "
              f"{r['rmse']:>5.2f}{unit_sym} "
              f"{r['bias']:>+5.2f}{unit_sym} "
              f"{r['sigma_emp']:>4.2f}{unit_sym} "
              f"{r['brier']:>6.3f} "
              f"{r['bss']:>+6.3f} "
              f"{r['pit_std']:>6.3f} "
              f"{r['out_of_range_pct']:>5.1f}% "
              f"{r['top3_hit_pct']:>5.1f}%  {verdict}")

    if insufficient:
        print(f"\nInsufficient data (<10 paired obs): "
              f"{', '.join(r['city'] for r in insufficient)}")

    print('-' * 120)
    n_trade = sum(1 for r in scored if r["bss"] > 0.10 and
                  abs(r["pit_std"] - ideal_pit) < 0.05 and r["out_of_range_pct"] < 5)
    n_caution = sum(1 for r in scored if r["bss"] > 0 and
                    abs(r["pit_std"] - ideal_pit) < 0.10 and r["out_of_range_pct"] < 8) - n_trade
    n_avoid = len(scored) - n_trade - n_caution
    print(f"Verdict split: ✅ {n_trade} trade  ⚠️  {n_caution} caution  ❌ {n_avoid} avoid")
    print('=' * 120)
    print(f"PIT σ target: {ideal_pit:.3f} (uniform [0,1]).  OOR target: <1%  (gaussian expects ~0.3%).")
    print(f"BSS > 0 = beats climatology.  Brier baseline {scored[0]['brier_baseline']:.3f} shown for scale.")


def main():
    parser = argparse.ArgumentParser(description="Per-city Brier backtest")
    parser.add_argument("--city", type=str, help="Score a single city slug")
    parser.add_argument("--csv", type=str, help="Write results to CSV")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.city:
        match = [(n, c) for n, c in CITIES.items() if c["slug"] == args.city]
        if not match:
            print(f"City slug '{args.city}' not found")
            return
        city_name, city = match[0]
        r = score_city(city_name, city, verbose=True)
        import pprint; pprint.pprint(r)
        return

    results = []
    for city_name, city in CITIES.items():
        r = score_city(city_name, city, verbose=args.verbose)
        results.append(r)

    print_summary(results)

    if args.csv:
        import csv
        scored = [r for r in results if not r.get("insufficient")]
        with open(args.csv, "w", newline="") as f:
            fields = ["city", "slug", "station", "unit", "n", "n_scored", "rmse",
                      "bias", "sigma_emp", "brier", "brier_baseline", "bss",
                      "pit_mean", "pit_std", "out_of_range_pct", "top3_hit_pct"]
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in scored:
                w.writerow(r)
        print(f"\nWrote {len(scored)} rows to {args.csv}")


if __name__ == "__main__":
    main()
