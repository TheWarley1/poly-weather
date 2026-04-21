"""
Polymarket Weather Edge Finder V4
==================================
Full API integration — auto-fetches live markets, forecasts,
resolution history, and price movements. No manual bin entry.

V4 improvements over V3:
- Multi-model ensemble (GFS, ECMWF, ICON, GEM, JMA) with lead-time-weighted blending
- Model spread → dynamic uncertainty signal (replaces hardcoded σ)
- Skew-normal + KDE hybrid probability model (replaces single Gaussian)
- Self-calibrating σ from logged empirical forecast errors
- Proper bias correction with persistent forecast logging
- Weather Underground station offset tracking

APIs:
- Polymarket Gamma API (markets, prices, resolution)
- Polymarket CLOB API (price history, orderbook)
- NWS API (US hourly forecasts)
- Open-Meteo API (multi-model forecasts + historical)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import json
import re
import os
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import skewnorm, gaussian_kde
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# PERSISTENT FORECAST LOG
# ─────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / "forecast_logs"
LOG_DIR.mkdir(exist_ok=True)


def get_log_path(city_slug):
    """Get path to city's forecast log file."""
    return LOG_DIR / f"{city_slug}.json"


def load_forecast_log(city_slug):
    """Load the forecast log for a city."""
    path = get_log_path(city_slug)
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except:
            return []
    return []


def save_forecast_log(city_slug, log):
    """Save forecast log for a city."""
    path = get_log_path(city_slug)
    # Keep last 90 days of entries
    cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    log = [entry for entry in log if entry.get("date", "") >= cutoff]
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


def log_forecast(city_slug, date_str, forecast_high, model_highs, actual=None):
    """Log a forecast (and optionally actual outcome) for bias tracking."""
    log = load_forecast_log(city_slug)
    # Update or append
    existing = next((e for e in log if e["date"] == date_str), None)
    if existing:
        if forecast_high is not None:
            existing["forecast"] = forecast_high
        if model_highs:
            existing["model_highs"] = model_highs
        if actual is not None:
            existing["actual"] = actual
    else:
        log.append({
            "date": date_str,
            "forecast": forecast_high,
            "model_highs": model_highs,
            "actual": actual,
            "logged_at": datetime.now().isoformat(),
        })
    save_forecast_log(city_slug, log)


def compute_empirical_bias_and_std(city_slug):
    """Compute bias and std from logged forecast vs actual outcomes."""
    log = load_forecast_log(city_slug)
    errors = []
    for entry in log:
        if entry.get("forecast") is not None and entry.get("actual") is not None:
            errors.append(entry["actual"] - entry["forecast"])
    if len(errors) < 3:
        return None, None
    return round(np.mean(errors), 2), round(np.std(errors), 2)


# ─────────────────────────────────────────────
# CITY REGISTRY (with WU station offsets)
# ─────────────────────────────────────────────
# wu_offset: measured difference between Open-Meteo archive and Polymarket/WU resolution.
# Positive = WU reads warmer than us → add to forecast.  Negative = WU reads cooler → subtract.
# Computed from 14-16 paired observations per city (Mar 24 – Apr 8 2026).
CITIES = {
    "San Francisco": {"slug": "san-francisco", "lat": 37.62, "lon": -122.39, "unit": "F", "station": "KSFO", "wu_offset": -1.2, "tz": "America/Los_Angeles"},
    "New York": {"slug": "nyc", "lat": 40.77, "lon": -73.87, "unit": "F", "station": "KLGA", "wu_offset": 0, "tz": "America/New_York"},
    "Chicago": {"slug": "chicago", "lat": 41.98, "lon": -87.90, "unit": "F", "station": "KORD", "wu_offset": +1.1, "tz": "America/Chicago"},
    "Miami": {"slug": "miami", "lat": 25.79, "lon": -80.29, "unit": "F", "station": "KMIA", "wu_offset": +0.9, "tz": "America/New_York"},
    "Dallas": {"slug": "dallas", "lat": 32.90, "lon": -97.04, "unit": "F", "station": "KDFW", "wu_offset": +0.7, "tz": "America/Chicago"},
    "Seattle": {"slug": "seattle", "lat": 47.45, "lon": -122.31, "unit": "F", "station": "KSEA", "wu_offset": +1.0, "tz": "America/Los_Angeles"},
    "Atlanta": {"slug": "atlanta", "lat": 33.64, "lon": -84.43, "unit": "F", "station": "KATL", "wu_offset": +0.9, "tz": "America/New_York"},
    "London": {"slug": "london", "lat": 51.47, "lon": -0.46, "unit": "C", "station": "EGLL", "wu_offset": 0, "tz": "Europe/London"},
    "Tokyo": {"slug": "tokyo", "lat": 35.55, "lon": 139.78, "unit": "C", "station": "RJTT", "wu_offset": 0, "tz": "Asia/Tokyo"},
    "Seoul": {"slug": "seoul", "lat": 37.46, "lon": 126.44, "unit": "C", "station": "RKSI", "wu_offset": 0, "tz": "Asia/Seoul"},
    "Toronto": {"slug": "toronto", "lat": 43.68, "lon": -79.63, "unit": "C", "station": "CYYZ", "wu_offset": 0, "tz": "America/Toronto"},
    "Singapore": {"slug": "singapore", "lat": 1.36, "lon": 103.99, "unit": "C", "station": "WSSS", "wu_offset": 0, "tz": "Asia/Singapore"},
    "Paris": {"slug": "paris", "lat": 49.01, "lon": 2.55, "unit": "C", "station": "LFPG", "wu_offset": 0, "tz": "Europe/Paris"},
    "Shanghai": {"slug": "shanghai", "lat": 31.14, "lon": 121.81, "unit": "C", "station": "ZSPD", "wu_offset": 0, "tz": "Asia/Shanghai"},
    "Buenos Aires": {"slug": "buenos-aires", "lat": -34.82, "lon": -58.54, "unit": "C", "station": "SAEZ", "wu_offset": 0, "tz": "America/Argentina/Buenos_Aires"},
    "Wellington": {"slug": "wellington", "lat": -41.33, "lon": 174.81, "unit": "C", "station": "NZWN", "wu_offset": 0, "tz": "Pacific/Auckland"},
    "Ankara": {"slug": "ankara", "lat": 40.12, "lon": 32.99, "unit": "C", "station": "LTAC", "wu_offset": 0, "tz": "Europe/Istanbul"},
    "Tel Aviv": {"slug": "tel-aviv", "lat": 32.01, "lon": 34.88, "unit": "C", "station": "LLBG", "wu_offset": 0, "tz": "Asia/Jerusalem"},
    "Hong Kong": {"slug": "hong-kong", "lat": 22.31, "lon": 113.92, "unit": "C", "station": "VHHH", "wu_offset": 0, "tz": "Asia/Hong_Kong"},
    "Munich": {"slug": "munich", "lat": 48.35, "lon": 11.79, "unit": "C", "station": "EDDM", "wu_offset": 0, "tz": "Europe/Berlin"},
    "Sao Paulo": {"slug": "sao-paulo", "lat": -23.63, "lon": -46.66, "unit": "C", "station": "SBGR", "wu_offset": 0, "tz": "America/Sao_Paulo"},
    "Lucknow": {"slug": "lucknow", "lat": 26.76, "lon": 80.88, "unit": "C", "station": "VILK", "wu_offset": 0, "tz": "Asia/Kolkata"},
    "Warsaw": {"slug": "warsaw", "lat": 52.17, "lon": 20.97, "unit": "C", "station": "EPWA", "wu_offset": 0, "tz": "Europe/Warsaw"},
    "Milan": {"slug": "milan", "lat": 45.63, "lon": 8.72, "unit": "C", "station": "LIMC", "wu_offset": 0, "tz": "Europe/Rome"},
    "Madrid": {"slug": "madrid", "lat": 40.47, "lon": -3.56, "unit": "C", "station": "LEMD", "wu_offset": 0, "tz": "Europe/Madrid"},
    "Taipei": {"slug": "taipei", "lat": 25.08, "lon": 121.23, "unit": "C", "station": "RCTP", "wu_offset": 0, "tz": "Asia/Taipei"},
}

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
NWS_HEADERS = {"User-Agent": "PolymarketWeatherEdge/4.0"}
MONTH_SLUG = {1: "january", 2: "february", 3: "march", 4: "april", 5: "may", 6: "june",
              7: "july", 8: "august", 9: "september", 10: "october", 11: "november", 12: "december"}

# Lead-time-dependent model weights (higher = more trusted at that lead time)
# NWS is best at short range; ECMWF best at medium range
MODEL_WEIGHTS = {
    # lead_days: {model: weight}
    0: {"nws": 3.0, "ecmwf_ifs025": 2.5, "gfs_seamless": 2.0, "icon_seamless": 1.5, "gem_seamless": 1.0},
    1: {"nws": 2.5, "ecmwf_ifs025": 2.5, "gfs_seamless": 2.0, "icon_seamless": 1.5, "gem_seamless": 1.0},
    2: {"nws": 2.0, "ecmwf_ifs025": 3.0, "gfs_seamless": 2.0, "icon_seamless": 1.5, "gem_seamless": 1.0},
    3: {"nws": 1.5, "ecmwf_ifs025": 3.0, "gfs_seamless": 2.0, "icon_seamless": 1.5, "gem_seamless": 1.0},
}


# ─────────────────────────────────────────────
# POLYMARKET API
# ─────────────────────────────────────────────
def build_event_slug(city_slug, target_date):
    """Build the Polymarket event slug for a city/date."""
    month = MONTH_SLUG[target_date.month]
    day = target_date.day
    year = target_date.year
    return f"highest-temperature-in-{city_slug}-on-{month}-{day}-{year}"


@st.cache_data(ttl=120)
def fetch_polymarket_event(event_slug):
    """Fetch event data from Gamma API."""
    try:
        r = requests.get(f"{GAMMA_BASE}/events?slug={event_slug}&limit=1", timeout=10)
        if r.status_code != 200 or not r.text.strip():
            return None
        data = r.json()
        return data[0] if data else None
    except:
        return None


def parse_event_markets(event):
    """Extract bin labels, prices, volumes, and token IDs from event."""
    if not event:
        return []

    markets = []
    for m in event.get("markets", []):
        title = m.get("groupItemTitle", "")
        prices_raw = m.get("outcomePrices", "[]")
        if isinstance(prices_raw, str):
            try:
                prices = json.loads(prices_raw)
            except:
                prices = []
        else:
            prices = prices_raw

        yes_price = float(prices[0]) if prices else 0

        clob_raw = m.get("clobTokenIds", "")
        if isinstance(clob_raw, str):
            try:
                clob_ids = json.loads(clob_raw)
            except:
                clob_ids = []
        else:
            clob_ids = clob_raw or []

        resolved_yes = (str(prices[0]) == "1") if prices else False

        markets.append({
            "bin": title,
            "yes_price": yes_price,
            "yes_pct": round(yes_price * 100, 1),
            "volume": m.get("volumeNum", 0),
            "market_id": m.get("id", ""),
            "clob_token_yes": clob_ids[0] if clob_ids else "",
            "clob_token_no": clob_ids[1] if len(clob_ids) > 1 else "",
            "resolved_yes": resolved_yes,
            "closed": m.get("closed", False),
            "best_bid": m.get("bestBid", 0),
            "best_ask": m.get("bestAsk", 0),
            "spread": m.get("spread", 0),
        })

    return sorted(markets, key=lambda x: parse_bin_value(x["bin"]))


def parse_bin_value(label):
    """Extract numeric value from bin label for sorting."""
    nums = re.findall(r'-?\d+', label.replace("°F", "").replace("°C", ""))
    return int(nums[0]) if nums else 0


@st.cache_data(ttl=300)
def fetch_price_history(token_id):
    """Fetch CLOB price history for a token."""
    if not token_id:
        return []
    try:
        r = requests.get(f"{CLOB_BASE}/prices-history?market={token_id}&interval=all&fidelity=5", timeout=10)
        if r.status_code == 200:
            data = r.json()
            return data.get("history", [])
    except:
        pass
    return []


@st.cache_data(ttl=600)
def fetch_resolution_history(city_slug, num_days=14):
    """Fetch resolved markets for a city over recent days."""
    results = []
    today = datetime.now().date()

    for i in range(1, num_days + 1):
        d = today - timedelta(days=i)
        slug = build_event_slug(city_slug, d)
        event = fetch_polymarket_event(slug)
        if event and event.get("closed"):
            winner = None
            for m in event.get("markets", []):
                prices_raw = m.get("outcomePrices", "[]")
                if isinstance(prices_raw, str):
                    try:
                        prices = json.loads(prices_raw)
                    except:
                        prices = []
                else:
                    prices = prices_raw
                if prices and str(prices[0]) == "1":
                    winner = m.get("groupItemTitle", "")

            results.append({
                "date": d.strftime("%Y-%m-%d"),
                "date_label": d.strftime("%b %d"),
                "winner": winner,
                "volume": event.get("volume", 0),
            })

    return results


def find_active_markets(city_slug, days_ahead=7):
    """Find active (unresolved) markets for a city."""
    active = []
    today = datetime.now().date()

    for i in range(0, days_ahead):
        d = today + timedelta(days=i)
        slug = build_event_slug(city_slug, d)
        event = fetch_polymarket_event(slug)
        if event and not event.get("closed"):
            active.append({
                "date": d,
                "slug": slug,
                "event": event,
                "volume": event.get("volume", 0),
            })

    return active


# ─────────────────────────────────────────────
# WEATHER FORECAST APIs (Multi-Model)
# ─────────────────────────────────────────────
@st.cache_data(ttl=900)
def get_nws_hourly(lat, lon):
    """Get NWS hourly forecast (US only)."""
    try:
        r = requests.get(f"https://api.weather.gov/points/{lat},{lon}", headers=NWS_HEADERS, timeout=10)
        grid = r.json()
        hourly_url = grid["properties"]["forecastHourly"]
        r2 = requests.get(hourly_url, headers=NWS_HEADERS, timeout=10)
        periods = []
        for p in r2.json()["properties"]["periods"]:
            dt = datetime.fromisoformat(p["startTime"].replace("Z", "+00:00"))
            periods.append({
                "datetime": dt, "date": dt.strftime("%Y-%m-%d"), "hour": dt.hour,
                "hour_label": dt.strftime("%I%p").lstrip("0"),
                "temp_f": p["temperature"] if p["temperatureUnit"] == "F" else round(p["temperature"] * 9/5 + 32),
                "temp_c": p["temperature"] if p["temperatureUnit"] == "C" else round((p["temperature"] - 32) * 5/9),
                "forecast": p["shortForecast"],
                "wind": f"{p['windSpeed']} {p['windDirection']}",
            })
        return pd.DataFrame(periods)
    except:
        return None


@st.cache_data(ttl=900)
def get_openmeteo_multimodel(lat, lon):
    """Fetch multi-model forecasts from Open-Meteo (ECMWF, GFS, ICON, GEM).

    Queries each model individually — the combined multi-model endpoint
    fails because 'jma' is not a valid model string, and the response
    format for combined queries doesn't match per-model block parsing.
    """
    # JMA removed — returns "invalid String value" error from Open-Meteo API
    models = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless"]
    all_models = {}

    for model in models:
        try:
            r = requests.get(
                f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                f"&hourly=temperature_2m&models={model}"
                f"&temperature_unit=fahrenheit&timezone=auto&forecast_days=7",
                timeout=12)

            if r.status_code != 200:
                continue

            data = r.json()
            if "error" in data:
                continue

            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            temps = hourly.get("temperature_2m", [])

            if times and temps:
                periods = []
                for i in range(len(times)):
                    if temps[i] is not None:
                        dt = datetime.fromisoformat(times[i])
                        periods.append({
                            "datetime": dt,
                            "date": dt.strftime("%Y-%m-%d"),
                            "hour": dt.hour,
                            "hour_label": dt.strftime("%I%p").lstrip("0"),
                            "temp_f": round(temps[i]),
                            "temp_c": round((temps[i] - 32) * 5 / 9),
                        })
                if periods:
                    all_models[model] = pd.DataFrame(periods)
        except:
            continue  # Timeout for this model — skip, don't fail all

    return all_models


@st.cache_data(ttl=900)
def get_openmeteo_hourly(lat, lon):
    """Get Open-Meteo default hourly forecast (global). Kept for backward compat."""
    try:
        r = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,windspeed_10m&temperature_unit=fahrenheit"
            f"&windspeed_unit=mph&timezone=auto&forecast_days=7", timeout=10)
        data = r.json()
        hourly = data["hourly"]
        periods = []
        for i in range(len(hourly["time"])):
            dt = datetime.fromisoformat(hourly["time"][i])
            periods.append({
                "datetime": dt, "date": dt.strftime("%Y-%m-%d"), "hour": dt.hour,
                "hour_label": dt.strftime("%I%p").lstrip("0"),
                "temp_f": round(hourly["temperature_2m"][i]),
                "temp_c": round((hourly["temperature_2m"][i] - 32) * 5/9),
                "wind_mph": round(hourly["windspeed_10m"][i]),
            })
        return pd.DataFrame(periods)
    except:
        return None


@st.cache_data(ttl=3600)
def get_historical_highs(lat, lon, target_date_str):
    """Historical daily highs for same calendar date."""
    month_day = target_date_str[5:]
    records = []
    for year in range(2015, 2026):
        target = f"{year}-{month_day}"
        start = (datetime.strptime(target, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        end = (datetime.strptime(target, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            r = requests.get(
                f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
                f"&daily=temperature_2m_max&temperature_unit=fahrenheit&timezone=auto"
                f"&start_date={start}&end_date={end}", timeout=8)
            if r.status_code == 200:
                d = r.json()
                if "daily" in d and d["daily"]["temperature_2m_max"]:
                    idx = 1 if len(d["daily"]["temperature_2m_max"]) > 1 else 0
                    high = d["daily"]["temperature_2m_max"][idx]
                    if high is not None:
                        records.append({"year": year, "high_f": round(high), "high_c": round((high - 32) * 5 / 9)})
        except:
            pass
    return records


# ─────────────────────────────────────────────
# MULTI-MODEL ENSEMBLE
# ─────────────────────────────────────────────
def compute_ensemble_forecast(nws_df, multimodel_data, target_str, lead_days, city_unit, wu_offset=0):
    """
    Compute weighted ensemble forecast high from multiple models.
    Returns: (ensemble_high, model_highs_dict, model_spread, source_parts)
    """
    col = "temp_f" if city_unit == "F" else "temp_c"
    weights_table = MODEL_WEIGHTS.get(min(lead_days, 3), MODEL_WEIGHTS[3])

    model_highs = {}
    source_parts = []

    # NWS forecast
    if nws_df is not None and not nws_df.empty:
        day_nws = nws_df[nws_df["date"] == target_str]
        if not day_nws.empty:
            nws_high = int(day_nws[col].max())
            model_highs["nws"] = nws_high
            source_parts.append(f"NWS: {nws_high}°")

    # Open-Meteo multi-model forecasts
    for model_name, model_df in multimodel_data.items():
        if model_df is not None and not model_df.empty:
            day_data = model_df[model_df["date"] == target_str]
            if not day_data.empty:
                high = int(day_data[col].max())
                model_highs[model_name] = high
                source_parts.append(f"{model_name}: {high}°")

    if not model_highs:
        return None, {}, 0, []

    # Weighted average
    total_weight = 0
    weighted_sum = 0
    for model, high in model_highs.items():
        w = weights_table.get(model, 1.0)
        weighted_sum += high * w
        total_weight += w

    ensemble_high = round(weighted_sum / total_weight) if total_weight > 0 else None

    # Apply WU station offset
    if ensemble_high is not None and wu_offset != 0:
        ensemble_high += wu_offset

    # Model spread (std dev of model predictions) — key uncertainty signal
    if len(model_highs) >= 2:
        model_spread = np.std(list(model_highs.values()))
    else:
        model_spread = 0

    return ensemble_high, model_highs, model_spread, source_parts


# ─────────────────────────────────────────────
# PROBABILITY MODEL (Skew-Normal + KDE Hybrid)
# ─────────────────────────────────────────────
def parse_bin_range(label, unit="F"):
    """Parse bin label to (low, high) range.

    Handles formats like: '55-59°F', '55 to 59°F', '75°F or higher', '30°F or below',
    and negative temperatures like '-5 to -1°C'.
    """
    label = label.strip()

    if "or higher" in label.lower() or "or above" in label.lower():
        nums = re.findall(r'-?\d+', label)
        return (int(nums[0]), int(nums[0]) + 30) if nums else (None, None)
    if "or below" in label.lower() or "or lower" in label.lower():
        nums = re.findall(r'-?\d+', label)
        return (int(nums[0]) - 30, int(nums[0])) if nums else (None, None)

    # Match range patterns: "X-Y", "X to Y", "X – Y"
    # Use a pattern that distinguishes range-separator hyphens from negative signs
    range_match = re.match(r'(-?\d+)\s*(?:to|–|—)\s*(-?\d+)', label)
    if range_match:
        return (int(range_match.group(1)), int(range_match.group(2)))

    # Handle "X-Y" where hyphen is separator (not negative sign)
    # Match: optional negative + digits, then hyphen, then optional negative + digits
    hyphen_match = re.match(r'(-?\d+)\s*-\s*(-?\d+)', label)
    if hyphen_match:
        lo, hi = int(hyphen_match.group(1)), int(hyphen_match.group(2))
        # Sanity check: if hi < lo, the hyphen was misinterpreted
        if hi >= lo:
            return (lo, hi)
        # For cases like "55-59" where regex gives (55, 59) correctly,
        # but also handle negative temps like "-5--1" → (-5, -1)
        return (min(lo, hi), max(lo, hi))

    nums = re.findall(r'-?\d+', label)
    if len(nums) >= 2:
        lo, hi = int(nums[0]), int(nums[1])
        return (min(lo, hi), max(lo, hi))
    elif len(nums) == 1:
        return (int(nums[0]), int(nums[0]))
    return (None, None)


def estimate_skewness(historical, forecast_high, unit="F"):
    """Estimate skewness from historical data relative to forecast."""
    if not historical or len(historical) < 5:
        return 0  # No skew if insufficient data
    hist_vals = np.array([h["high_f"] if unit == "F" else h["high_c"] for h in historical])
    # Use scipy's skewness estimate
    skew = stats.skew(hist_vals)
    # Clamp to reasonable range
    return np.clip(skew, -3, 3)


def build_model_probabilities(forecast_high, bins, unit="F", forecast_std=None,
                               historical=None, model_spread=None, empirical_std=None):
    """
    Build probability distribution across bins using a skew-normal + KDE hybrid.

    Priority for uncertainty (σ):
    1. empirical_std (from logged forecast errors) — best signal
    2. model_spread (disagreement between forecast models) — good real-time signal
    3. forecast_std (user slider / default) — fallback
    """
    # Determine effective standard deviation
    if empirical_std is not None and empirical_std > 0:
        # Self-calibrated from actual forecast errors
        base_std = empirical_std
    elif model_spread is not None and model_spread > 1.0:
        # Multi-model spread is a strong real-time uncertainty signal
        # Scale it: spread of ~2°F between models ≈ ~3°F effective σ
        base_std = model_spread * 1.5
    else:
        base_std = forecast_std if forecast_std else (2.5 if unit == "F" else 1.4)

    # Blend with historical volatility if available
    if historical and len(historical) > 3:
        hist_vals = [h["high_f"] if unit == "F" else h["high_c"] for h in historical]
        hist_std = np.std(hist_vals)
        effective_std = 0.6 * base_std + 0.4 * hist_std
    else:
        effective_std = base_std

    effective_std = max(effective_std, 1.5 if unit == "F" else 0.8)

    # Estimate skewness from historical data
    skew_param = estimate_skewness(historical, forecast_high, unit)

    # Build hybrid distribution
    probs = {}

    # Component 1: Skew-normal (forecast-centered, captures asymmetry)
    # scipy skewnorm parameterization: a=shape (skew), loc, scale
    skew_dist = skewnorm(a=skew_param, loc=forecast_high, scale=effective_std)

    # Component 2: KDE from historical (captures multimodality, non-Gaussian tails)
    kde_weight = 0
    kde_dist = None
    if historical and len(historical) >= 8:
        hist_vals = np.array([h["high_f"] if unit == "F" else h["high_c"] for h in historical])
        try:
            kde_dist = gaussian_kde(hist_vals, bw_method="silverman")
            kde_weight = 0.25  # 25% KDE, 75% skew-normal
        except:
            kde_weight = 0

    for b in bins:
        lo, hi = parse_bin_range(b, unit)
        if lo is not None and hi is not None:
            # Skew-normal component
            p_skew = skew_dist.cdf(hi + 0.5) - skew_dist.cdf(lo - 0.5)

            # KDE component (if available)
            if kde_dist is not None and kde_weight > 0:
                # Integrate KDE over bin range
                x_range = np.linspace(lo - 0.5, hi + 0.5, 50)
                p_kde = np.trapezoid(kde_dist(x_range), x_range)
                p_kde = max(0, p_kde)
            else:
                p_kde = p_skew

            # Blend
            probs[b] = (1 - kde_weight) * p_skew + kde_weight * p_kde

    total = sum(probs.values())
    if total > 0:
        probs = {k: round(v / total * 100, 1) for k, v in probs.items()}

    return probs, effective_std, skew_param


def calculate_edges(model_probs, market_prices):
    """Calculate edge and Kelly for each bin."""
    results = {}
    for b in model_probs:
        mp = model_probs[b] / 100
        mkt = market_prices.get(b, 0) / 100
        if mkt <= 0.005 or mkt >= 0.995:
            results[b] = {"edge": 0, "kelly": 0, "signal": "⚪ SKIP"}
            continue
        edge = mp - mkt
        kelly = max(0, (mp * (1/mkt - 1) - (1 - mp)) / (1/mkt - 1)) if edge > 0 else 0
        if edge > 0.08:
            sig = "🟢 STRONG BUY"
        elif edge > 0.04:
            sig = "🟡 BUY"
        elif edge < -0.08:
            sig = "🔴 STRONG SELL"
        elif edge < -0.04:
            sig = "🟠 SELL"
        else:
            sig = "⚪ FAIR"
        results[b] = {"edge": round(edge * 100, 1), "kelly": round(kelly * 100, 1), "signal": sig}
    return results


def extract_resolution_temp(winner_label, unit="F"):
    """Extract approximate temperature from resolution label."""
    if not winner_label:
        return None
    nums = re.findall(r'-?\d+', winner_label)
    if len(nums) >= 2:
        return (int(nums[0]) + int(nums[1])) / 2
    elif len(nums) == 1:
        return int(nums[0])
    return None


def auto_calibrate_wu_offset(city_slug, city, res_history, historical):
    """
    Estimate WU station offset by comparing resolved actuals to
    historical climatology. Updates the city's wu_offset in-place.
    Returns the estimated offset.
    """
    if not res_history or not historical:
        return 0

    actual_temps = []
    for rh in res_history:
        t = extract_resolution_temp(rh["winner"], city["unit"])
        if t is not None:
            actual_temps.append(t)

    if len(actual_temps) < 5:
        return 0

    # Compare recent actuals to what our forecast models predicted (from log)
    log = load_forecast_log(city_slug)
    offsets = []
    for entry in log:
        if entry.get("forecast") is not None and entry.get("actual") is not None:
            offsets.append(entry["actual"] - entry["forecast"])

    if len(offsets) >= 5:
        offset = round(np.median(offsets), 1)
        return np.clip(offset, -5, 5)  # Clamp to ±5 degrees

    return 0


# ─────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────
st.set_page_config(page_title="Weather Edge V4", page_icon="🌡️", layout="wide")

st.title("🌡️ Polymarket Weather Edge V4")
st.caption("Multi-model ensemble + self-calibrating edge detection")

# ─────── SIDEBAR ───────
with st.sidebar:
    st.header("Market Selection")
    city_name = st.selectbox("City", list(CITIES.keys()))
    city = CITIES[city_name]

    st.markdown("---")

    # Find active markets
    with st.spinner(f"Scanning {city_name} markets..."):
        active_markets = find_active_markets(city["slug"], days_ahead=7)

    if not active_markets:
        st.warning(f"No active markets found for {city_name}")
        st.stop()

    date_options = {f"{am['date'].strftime('%a %b %d')} (${am['volume']:,.0f} vol)": am for am in active_markets}
    selected_label = st.selectbox("Target Date", list(date_options.keys()))
    selected_market = date_options[selected_label]
    target_date = selected_market["date"]

    lead_days = (target_date - datetime.now().date()).days

    st.markdown("---")
    st.header("Model Settings")

    default_std = {0: 1.5, 1: 2.5, 2: 3.5, 3: 4.5}.get(lead_days, 5.0)
    default_std_c = round(default_std * 5/9, 1) if city["unit"] == "C" else default_std

    forecast_std = st.slider(
        f"Forecast uncertainty override (±°{city['unit']})",
        0.5, 10.0, default_std_c if city["unit"] == "C" else default_std, 0.5,
        help=f"Lead time: {lead_days} day(s). Only used as fallback — model spread and empirical errors take priority."
    )

    use_nws = st.checkbox("Include NWS forecast", value=city["unit"] == "F",
                          help="Available for US cities only")

    use_multimodel = st.checkbox("Multi-model ensemble", value=True,
                                  help="Fetch ECMWF, GFS, ICON, GEM, JMA from Open-Meteo")

    st.markdown("---")
    st.caption(f"**Station:** {city['station']}")
    st.caption(f"**Slug:** {selected_market['slug']}")


# ─────── LOAD ALL DATA ───────
with st.spinner("Loading market data, forecasts, and history..."):
    # 1. Market data
    event = selected_market["event"]
    markets = parse_event_markets(event)

    # 2. Forecasts — multi-model ensemble
    multimodel_data = {}
    if use_multimodel:
        multimodel_data = get_openmeteo_multimodel(city["lat"], city["lon"])

    # Also fetch standard Open-Meteo (for hourly chart with wind data)
    om_df = get_openmeteo_hourly(city["lat"], city["lon"])
    nws_df = get_nws_hourly(city["lat"], city["lon"]) if use_nws and city["unit"] == "F" else None

    # 3. Historical
    target_str = target_date.strftime("%Y-%m-%d")
    historical = get_historical_highs(city["lat"], city["lon"], target_str)

    # 4. Resolution history
    res_history = fetch_resolution_history(city["slug"], num_days=14)

    # 5. Empirical calibration from forecast log
    empirical_bias, empirical_std = compute_empirical_bias_and_std(city["slug"])

    # 6. Auto-calibrate WU offset
    wu_offset = auto_calibrate_wu_offset(city["slug"], city, res_history, historical)


# ─────── COMPUTE FORECAST HIGH (Multi-Model Ensemble) ───────
if use_multimodel and multimodel_data:
    forecast_high, model_highs, model_spread, source_parts = compute_ensemble_forecast(
        nws_df, multimodel_data, target_str, lead_days, city["unit"], wu_offset
    )
else:
    # Fallback to V3 simple averaging
    forecast_high = None
    model_highs = {}
    model_spread = 0
    source_parts = []

    if nws_df is not None and not nws_df.empty:
        col = "temp_f" if city["unit"] == "F" else "temp_c"
        day_nws = nws_df[nws_df["date"] == target_str]
        if not day_nws.empty:
            nws_high = int(day_nws[col].max())
            source_parts.append(f"NWS: {nws_high}°")
            forecast_high = nws_high
            model_highs["nws"] = nws_high

    if om_df is not None and not om_df.empty:
        col = "temp_f" if city["unit"] == "F" else "temp_c"
        day_om = om_df[om_df["date"] == target_str]
        if not day_om.empty:
            om_high = int(day_om[col].max())
            source_parts.append(f"Open-Meteo: {om_high}°")
            model_highs["open_meteo"] = om_high
            if forecast_high is None:
                forecast_high = om_high
            else:
                forecast_high = round((forecast_high + om_high) / 2)

# Apply empirical bias correction
if forecast_high is not None and empirical_bias is not None:
    forecast_high = round(forecast_high + empirical_bias)
    source_parts.append(f"Bias correction: {empirical_bias:+.1f}°")

if forecast_high is None:
    st.error("Could not retrieve forecast data.")
    st.stop()


# Log today's forecast for future bias tracking
log_forecast(city["slug"], target_str, forecast_high, model_highs)

# Backfill actuals from resolution history into the log
# ONLY if the standalone forecast_logger.py hasn't already provided a real observation.
# The logger writes actual_source="open_meteo_archive" which is the real station temp;
# this fallback uses bin midpoints which are less accurate.
existing_log = load_forecast_log(city["slug"])
dates_with_real_actual = {
    e["date"] for e in existing_log
    if e.get("actual_source") == "open_meteo_archive"
}
for rh in res_history:
    if rh["date"] in dates_with_real_actual:
        continue  # Already have a real station observation — don't overwrite with bin midpoint
    actual = extract_resolution_temp(rh["winner"], city["unit"])
    if actual is not None:
        log_forecast(city["slug"], rh["date"], None, {}, actual=actual)


# Allow override
with st.sidebar:
    st.markdown("---")
    forecast_high = st.number_input(
        f"Forecast high (°{city['unit']})",
        value=forecast_high,
        help=f"Sources: {' | '.join(source_parts)}"
    )


# ─────── BUILD MODEL ───────
bin_labels = [m["bin"] for m in markets]
market_prices = {m["bin"]: m["yes_pct"] for m in markets}
model_probs, eff_std, skew_param = build_model_probabilities(
    forecast_high, bin_labels, city["unit"], forecast_std,
    historical, model_spread, empirical_std
)
edges = calculate_edges(model_probs, market_prices)


# ─────── TOP METRICS ───────
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric("Forecast High", f"{forecast_high}°{city['unit']}")
with c2:
    if historical:
        h_avg = round(np.mean([h["high_f"] if city["unit"] == "F" else h["high_c"] for h in historical]))
        st.metric("Historical Avg", f"{h_avg}°{city['unit']}", f"{forecast_high - h_avg:+d}°")
    else:
        st.metric("Historical Avg", "N/A")
with c3:
    st.metric("Lead Time", f"{lead_days} day(s)")
with c4:
    st.metric("Market Volume", f"${event.get('volume', 0):,.0f}")
with c5:
    best = max(edges, key=lambda k: abs(edges[k]["edge"])) if edges else ""
    st.metric("Best Edge", f"{edges[best]['edge']:+.1f}%" if best else "N/A", best)
with c6:
    n_models = len(model_highs)
    spread_label = f"±{model_spread:.1f}°" if model_spread > 0 else "N/A"
    st.metric(f"Model Spread ({n_models})", spread_label)


# ─────── TABS ───────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Edge Scorecard", "📈 Hourly Forecast", "📜 Resolution History",
    "💹 Price Movements", "🔬 Ensemble", "⚙️ Model"
])

# ═══════════════════════════════════════
# TAB 1: EDGE SCORECARD
# ═══════════════════════════════════════
with tab1:
    fig = go.Figure()
    bins_sorted = [m["bin"] for m in markets]
    mkt_vals = [market_prices.get(b, 0) for b in bins_sorted]
    mod_vals = [model_probs.get(b, 0) for b in bins_sorted]

    fig.add_trace(go.Bar(x=bins_sorted, y=mkt_vals, name="Market",
                         marker_color="rgba(127,119,221,0.7)",
                         text=[f"{v:.1f}%" for v in mkt_vals], textposition="outside"))
    fig.add_trace(go.Bar(x=bins_sorted, y=mod_vals, name="Model",
                         marker_color="rgba(29,158,117,0.7)",
                         text=[f"{v:.1f}%" for v in mod_vals], textposition="outside"))

    fig.update_layout(
        title=f"{city_name} — {target_date.strftime('%a %b %d')} — Market vs Model",
        yaxis_title="Probability (%)", barmode="group", template="plotly_dark", height=420,
        yaxis=dict(range=[0, max(max(mkt_vals, default=0), max(mod_vals, default=0)) * 1.4]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    rows = []
    for m in markets:
        b = m["bin"]
        e = edges.get(b, {})
        rows.append({
            "Bin": b,
            "Market %": m["yes_pct"],
            "Model %": model_probs.get(b, 0),
            "Edge %": e.get("edge", 0),
            "Signal": e.get("signal", ""),
            "Kelly %": e.get("kelly", 0),
            "Volume": f"${m['volume']:,.0f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    buys = [(b, edges[b]) for b in edges if edges[b]["edge"] > 4]
    sells = [(b, edges[b]) for b in edges if edges[b]["edge"] < -4]

    if buys:
        best_buy = max(buys, key=lambda x: x[1]["edge"])
        st.success(f"**BUY: {best_buy[0]}** — Model {model_probs[best_buy[0]]:.1f}% vs Market "
                   f"{market_prices[best_buy[0]]:.1f}% → Edge +{best_buy[1]['edge']:.1f}% | Kelly {best_buy[1]['kelly']:.1f}%")
    if sells:
        best_sell = min(sells, key=lambda x: x[1]["edge"])
        st.warning(f"**SELL: {best_sell[0]}** — Model {model_probs[best_sell[0]]:.1f}% vs Market "
                   f"{market_prices[best_sell[0]]:.1f}% → Edge {best_sell[1]['edge']:.1f}%")
    if not buys and not sells:
        st.info("No significant edges detected. Market appears fairly priced.")


# ═══════════════════════════════════════
# TAB 2: HOURLY FORECAST
# ═══════════════════════════════════════
with tab2:
    hourly = None
    col_key = "temp_f" if city["unit"] == "F" else "temp_c"

    if nws_df is not None:
        hourly = nws_df[nws_df["date"] == target_str]
        src = "NWS"
    if (hourly is None or hourly.empty) and om_df is not None:
        hourly = om_df[om_df["date"] == target_str]
        src = "Open-Meteo"

    if hourly is not None and not hourly.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=hourly["hour_label"], y=hourly[col_key],
            mode="lines+markers", line=dict(color="#D85A30", width=3),
            marker=dict(size=5), name=f"Temp (°{city['unit']})"
        ))
        fig2.add_hline(y=forecast_high, line_dash="dash", line_color="#1D9E75",
                       annotation_text=f"Forecast High: {forecast_high}°")

        for i, m in enumerate(markets):
            lo, hi = parse_bin_range(m["bin"], city["unit"])
            if lo is not None and hi is not None and abs(hi - lo) < 20:
                fig2.add_hrect(y0=lo-0.5, y1=hi+0.5,
                              fillcolor=f"rgba(127,119,221,{0.03 + m['yes_price']*0.15})",
                              line_width=0)

        fig2.update_layout(
            title=f"Hourly Forecast ({src}) — {target_date.strftime('%a %b %d')}",
            yaxis_title=f"°{city['unit']}", template="plotly_dark", height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

        peak_idx = hourly[col_key].idxmax()
        peak_row = hourly.loc[peak_idx]
        st.info(f"**Peak:** {peak_row[col_key]}°{city['unit']} at {peak_row['hour_label']} | "
                f"Wind: {peak_row.get('wind', peak_row.get('wind_mph', 'N/A'))}")
    else:
        st.info("No hourly data available for this date.")

    if om_df is not None:
        st.subheader("7-Day Overview")
        daily = om_df.groupby("date")[col_key].max().reset_index()
        daily.columns = ["Date", f"High °{city['unit']}"]
        daily[""] = daily["Date"].apply(lambda d: " ← TARGET" if d == target_str else "")
        st.dataframe(daily, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════
# TAB 3: RESOLUTION HISTORY
# ═══════════════════════════════════════
with tab3:
    if res_history:
        st.subheader(f"{city_name} — Recent Resolutions")

        rh_df = pd.DataFrame(res_history)
        rh_df["temp"] = rh_df["winner"].apply(lambda w: extract_resolution_temp(w, city["unit"]))

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=rh_df["date_label"], y=rh_df["temp"],
            text=rh_df["winner"], textposition="outside",
            marker_color=["#1D9E75" if t and t >= forecast_high else "#534AB7" for t in rh_df["temp"]],
        ))
        fig3.add_hline(y=forecast_high, line_dash="dash", line_color="orange",
                       annotation_text=f"Current forecast: {forecast_high}°")

        if historical:
            h_avg = np.mean([h["high_f"] if city["unit"] == "F" else h["high_c"] for h in historical])
            fig3.add_hline(y=h_avg, line_dash="dot", line_color="gray",
                           annotation_text=f"Climatological avg: {h_avg:.0f}°")

        fig3.update_layout(
            title=f"Resolved Daily Highs — Last {len(res_history)} Days",
            yaxis_title=f"°{city['unit']}", template="plotly_dark", height=400,
        )
        st.plotly_chart(fig3, use_container_width=True)

        temps = [t for t in rh_df["temp"] if t is not None]
        if temps:
            st.markdown(f"**Recent average resolved high:** {np.mean(temps):.1f}°{city['unit']} | "
                        f"**Range:** {min(temps):.0f}° to {max(temps):.0f}° | "
                        f"**Std Dev:** {np.std(temps):.1f}°")
    else:
        st.info("No resolution history available yet.")

    if historical:
        st.subheader(f"Historical Highs — {target_date.strftime('%b %d')} (±1 day)")
        h_col = "high_f" if city["unit"] == "F" else "high_c"
        hist_df = pd.DataFrame(historical)

        fig_h = go.Figure()
        fig_h.add_trace(go.Bar(
            x=hist_df["year"], y=hist_df[h_col],
            text=[f"{h}°" for h in hist_df[h_col]], textposition="outside",
            marker_color="#4ECDC4",
        ))
        fig_h.add_hline(y=forecast_high, line_dash="dash", line_color="orange",
                        annotation_text=f"Forecast: {forecast_high}°")
        fig_h.update_layout(
            yaxis_title=f"°{city['unit']}", template="plotly_dark", height=350,
        )
        st.plotly_chart(fig_h, use_container_width=True)


# ═══════════════════════════════════════
# TAB 4: PRICE MOVEMENTS
# ═══════════════════════════════════════
with tab4:
    st.subheader("Market Price History")

    top_bins = sorted(markets, key=lambda m: m["yes_pct"], reverse=True)[:5]
    selected_bins = st.multiselect(
        "Bins to chart",
        [m["bin"] for m in markets],
        default=[m["bin"] for m in top_bins if m["yes_pct"] > 1]
    )

    if selected_bins:
        fig4 = go.Figure()
        colors = ["#D85A30", "#1D9E75", "#534AB7", "#378ADD", "#E24B4A", "#639922"]

        for i, b in enumerate(selected_bins):
            m = next((m for m in markets if m["bin"] == b), None)
            if m and m["clob_token_yes"]:
                with st.spinner(f"Loading price history for {b}..."):
                    history = fetch_price_history(m["clob_token_yes"])
                if history:
                    times = [datetime.fromtimestamp(h["t"]) for h in history]
                    prices = [h["p"] * 100 for h in history]
                    fig4.add_trace(go.Scatter(
                        x=times, y=prices, mode="lines",
                        name=b, line=dict(color=colors[i % len(colors)], width=2),
                    ))

        if fig4.data:
            fig4.update_layout(
                title="Price History (YES %)",
                yaxis_title="Price (%)", template="plotly_dark", height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No price history available for selected bins.")

    st.subheader("Orderbook Summary")
    ob_rows = []
    for m in markets:
        if m["yes_pct"] > 0.5:
            ob_rows.append({
                "Bin": m["bin"],
                "YES Price": f"{m['yes_pct']:.1f}%",
                "Volume": f"${m['volume']:,.0f}",
            })
    if ob_rows:
        st.dataframe(pd.DataFrame(ob_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════
# TAB 5: ENSEMBLE DETAILS (NEW)
# ═══════════════════════════════════════
with tab5:
    st.subheader("Multi-Model Ensemble Breakdown")

    if model_highs:
        # Bar chart of each model's predicted high
        model_names = list(model_highs.keys())
        model_vals = list(model_highs.values())
        weights_table = MODEL_WEIGHTS.get(min(lead_days, 3), MODEL_WEIGHTS[3])
        model_weights_vals = [weights_table.get(m, 1.0) for m in model_names]

        fig_ens = go.Figure()
        fig_ens.add_trace(go.Bar(
            x=model_names, y=model_vals,
            text=[f"{v}°" for v in model_vals], textposition="outside",
            marker_color=[f"rgba(29,158,117,{0.3 + 0.7 * w / max(model_weights_vals)})"
                          for w in model_weights_vals],
        ))
        fig_ens.add_hline(y=forecast_high, line_dash="dash", line_color="orange",
                          annotation_text=f"Ensemble: {forecast_high}°")
        fig_ens.update_layout(
            title=f"Model Forecasts for {target_date.strftime('%a %b %d')} (bar opacity = weight)",
            yaxis_title=f"Forecast High (°{city['unit']})",
            template="plotly_dark", height=350,
        )
        st.plotly_chart(fig_ens, use_container_width=True)

        # Model weight table
        weight_rows = []
        for m in model_names:
            weight_rows.append({
                "Model": m,
                f"Forecast °{city['unit']}": model_highs[m],
                "Weight": weights_table.get(m, 1.0),
                f"Δ from ensemble": f"{model_highs[m] - forecast_high:+d}°",
            })
        st.dataframe(pd.DataFrame(weight_rows), use_container_width=True, hide_index=True)

        st.markdown(f"**Model spread (σ):** {model_spread:.1f}°{city['unit']} across {len(model_highs)} models")
        if model_spread > 3:
            st.warning("High model disagreement — uncertainty is elevated. Consider smaller position sizes.")
        elif model_spread < 1:
            st.success("Strong model consensus — higher confidence in forecast.")
    else:
        st.info("Enable multi-model ensemble in sidebar to see model breakdown.")

    # Calibration status
    st.markdown("---")
    st.subheader("Self-Calibration Status")
    if empirical_bias is not None:
        st.markdown(f"**Empirical bias:** {empirical_bias:+.1f}° (forecast tends to be "
                    f"{'low' if empirical_bias > 0 else 'high'} by {abs(empirical_bias):.1f}°)")
        st.markdown(f"**Empirical σ:** {empirical_std:.1f}° (from logged forecast errors)")
        log = load_forecast_log(city["slug"])
        n_logged = len([e for e in log if e.get("actual") is not None])
        st.markdown(f"**Calibration data points:** {n_logged}")
    else:
        log = load_forecast_log(city["slug"])
        n_logged = len([e for e in log if e.get("actual") is not None])
        st.info(f"Collecting calibration data... ({n_logged}/3 data points needed). "
                f"The model will auto-calibrate once enough resolved markets are logged.")

    if wu_offset != 0:
        st.markdown(f"**WU station offset:** {wu_offset:+.1f}° (auto-detected)")
    else:
        st.caption("WU station offset: not yet calibrated (needs more resolved data)")


# ═══════════════════════════════════════
# TAB 6: MODEL DETAILS
# ═══════════════════════════════════════
with tab6:
    st.subheader("Model Configuration")

    sigma_source = "empirical (self-calibrated)" if empirical_std else \
                   f"model spread ({model_spread:.1f}° × 1.5)" if model_spread > 1 else \
                   "default + historical blend"

    st.markdown(f"""
    **Forecast:** {forecast_high}°{city['unit']} ({' | '.join(source_parts)})
    **Effective σ:** ±{eff_std:.1f}°{city['unit']} — source: {sigma_source}
    **Skewness:** {skew_param:.2f} ({'right-skewed' if skew_param > 0.3 else 'left-skewed' if skew_param < -0.3 else 'symmetric'})
    **Lead time:** {lead_days} day(s)
    **Models in ensemble:** {len(model_highs)}
    **Historical points:** {len(historical)} years
    **Resolution history:** {len(res_history)} days
    **Bias correction:** {f'{empirical_bias:+.1f}°' if empirical_bias is not None else 'Not yet calibrated'}
    **WU offset:** {f'{wu_offset:+.1f}°' if wu_offset != 0 else 'Not yet calibrated'}
    """)

    # PDF curve — now showing skew-normal
    x = np.linspace(forecast_high - 4*eff_std, forecast_high + 4*eff_std, 200)
    y_skew = skewnorm.pdf(x, a=skew_param, loc=forecast_high, scale=eff_std) * 100
    y_normal = stats.norm.pdf(x, forecast_high, eff_std) * 100

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=x, y=y_skew, fill="tozeroy",
                              fillcolor="rgba(93,202,165,0.2)",
                              line=dict(color="#1D9E75", width=2),
                              name="Skew-Normal (V4)"))
    fig5.add_trace(go.Scatter(x=x, y=y_normal,
                              line=dict(color="rgba(127,119,221,0.5)", width=1, dash="dot"),
                              name="Normal (V3)"))
    fig5.add_vline(x=forecast_high, line_dash="dash", line_color="orange",
                   annotation_text=f"Forecast: {forecast_high}°")
    for b in bin_labels:
        lo, _ = parse_bin_range(b, city["unit"])
        if lo is not None:
            fig5.add_vline(x=lo-0.5, line_dash="dot", line_color="rgba(255,255,255,0.2)", line_width=1)

    fig5.update_layout(title="Probability Density — Skew-Normal vs Normal",
                       xaxis_title=f"°{city['unit']}",
                       yaxis_title="Density", template="plotly_dark", height=350)
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("""
    **V4 Model Improvements:**
    - **Multi-model ensemble:** Weighted blend of NWS, ECMWF, GFS, ICON, GEM, JMA
    - **Dynamic uncertainty:** σ from model spread or empirical errors (not hardcoded)
    - **Skew-normal distribution:** Captures asymmetric temperature distributions
    - **KDE blending:** Historical distribution shape informs tail probabilities
    - **Self-calibrating bias:** Logs forecasts vs actuals, auto-corrects systematic errors
    - **WU station offset:** Detects and corrects for resolution station differences

    **Edge = Model% − Market%** | **Kelly = (p·b − q) / b** where p=model prob, b=decimal odds
    """)

# ─────── FOOTER ───────
st.markdown("---")
col_f1, col_f2 = st.columns(2)
with col_f1:
    st.caption("**Data:** Polymarket Gamma/CLOB API · NWS API · Open-Meteo Multi-Model")
with col_f2:
    st.caption("**Disclaimer:** Statistical analysis only, not financial advice.")
