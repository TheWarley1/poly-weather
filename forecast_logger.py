"""
Standalone Forecast Logger
===========================
Decoupled from Streamlit — runs headlessly every 6 hours (cron, Task Scheduler, etc.).

For every city in the registry:
  1. Fetches multi-model ensemble forecasts for active markets (next 3 days)
  2. Fetches actual observed daily-max temps from Open-Meteo archive for resolved dates
  3. Writes paired (forecast, actual) entries to forecast_logs/{city_slug}.json

This is the critical piece that makes V4's self-calibration loop actually work.
The Streamlit app's calibration code (empirical_bias, empirical_std, wu_offset)
already reads these logs — it just never had data because logging only happened
when a user manually opened a city in the UI.

Usage:
  python forecast_logger.py              # run once (all cities)
  python forecast_logger.py --city nyc   # run for one city
  python forecast_logger.py --backfill   # backfill last 90 days of actuals

Schedule (cron example, every 6 hours):
  0 */6 * * * cd /path/to/Polymarket-weather-edge && python forecast_logger.py >> logs/logger.log 2>&1
"""

import requests
import json
import os
import sys
import time
import re
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# ─────────────────────────────────────────────
# CITY REGISTRY (must match app.py exactly)
# ─────────────────────────────────────────────
# NOTE: NYC changed from KJFK to KLGA (LaGuardia) — Polymarket resolves against
# Weather Underground which uses LaGuardia for NYC markets, not JFK.
# See: https://x.com/gavelsvtw/status/1979549975655719355
# wu_offset: measured difference between Open-Meteo archive and Polymarket/WU resolution.
# Positive = WU reads warmer than us → add to forecast.  Negative = WU reads cooler → subtract.
# Computed from 14-16 paired observations per city (Mar 24 – Apr 8 2026).
# Cities with 0.0 have no data yet — will be updated as calibration data accumulates.
CITIES = {
    "San Francisco": {"slug": "san-francisco", "lat": 37.62, "lon": -122.39, "unit": "F", "station": "KSFO", "tz": "America/Los_Angeles", "wu_offset": -1.2},
    "New York":      {"slug": "nyc",           "lat": 40.77, "lon": -73.87, "unit": "F", "station": "KLGA", "tz": "America/New_York",      "wu_offset": 0.0},
    "Chicago":       {"slug": "chicago",       "lat": 41.98, "lon": -87.90, "unit": "F", "station": "KORD", "tz": "America/Chicago",       "wu_offset": +1.1},
    "Miami":         {"slug": "miami",         "lat": 25.79, "lon": -80.29, "unit": "F", "station": "KMIA", "tz": "America/New_York",      "wu_offset": +0.9},
    "Dallas":        {"slug": "dallas",        "lat": 32.90, "lon": -97.04, "unit": "F", "station": "KDFW", "tz": "America/Chicago",       "wu_offset": +0.7},
    "Seattle":       {"slug": "seattle",       "lat": 47.45, "lon": -122.31, "unit": "F", "station": "KSEA", "tz": "America/Los_Angeles",  "wu_offset": +1.0},
    "Atlanta":       {"slug": "atlanta",       "lat": 33.64, "lon": -84.43, "unit": "F", "station": "KATL", "tz": "America/New_York",      "wu_offset": +0.9},
    "London":        {"slug": "london",        "lat": 51.47, "lon": -0.46,  "unit": "C", "station": "EGLL", "tz": "Europe/London",         "wu_offset": 0.0},
    "Tokyo":         {"slug": "tokyo",         "lat": 35.55, "lon": 139.78, "unit": "C", "station": "RJTT", "tz": "Asia/Tokyo",            "wu_offset": 0.0},
    "Seoul":         {"slug": "seoul",         "lat": 37.46, "lon": 126.44, "unit": "C", "station": "RKSI", "tz": "Asia/Seoul",            "wu_offset": 0.0},
    "Toronto":       {"slug": "toronto",       "lat": 43.68, "lon": -79.63, "unit": "C", "station": "CYYZ", "tz": "America/Toronto",       "wu_offset": 0.0},
    "Singapore":     {"slug": "singapore",     "lat": 1.36,  "lon": 103.99, "unit": "C", "station": "WSSS", "tz": "Asia/Singapore",        "wu_offset": 0.0},
    "Paris":         {"slug": "paris",         "lat": 49.01, "lon": 2.55,   "unit": "C", "station": "LFPG", "tz": "Europe/Paris",          "wu_offset": 0.0},
    "Shanghai":      {"slug": "shanghai",      "lat": 31.14, "lon": 121.81, "unit": "C", "station": "ZSPD", "tz": "Asia/Shanghai",         "wu_offset": 0.0},
    "Buenos Aires":  {"slug": "buenos-aires",  "lat": -34.82,"lon": -58.54, "unit": "C", "station": "SAEZ", "tz": "America/Argentina/Buenos_Aires", "wu_offset": 0.0},
    "Wellington":    {"slug": "wellington",    "lat": -41.33, "lon": 174.81, "unit": "C", "station": "NZWN", "tz": "Pacific/Auckland",     "wu_offset": 0.0},
    "Ankara":        {"slug": "ankara",        "lat": 40.12, "lon": 32.99,  "unit": "C", "station": "LTAC", "tz": "Europe/Istanbul",       "wu_offset": 0.0},
    "Tel Aviv":      {"slug": "tel-aviv",      "lat": 32.01, "lon": 34.88,  "unit": "C", "station": "LLBG", "tz": "Asia/Jerusalem",        "wu_offset": 0.0},
    "Hong Kong":     {"slug": "hong-kong",     "lat": 22.31, "lon": 113.92, "unit": "C", "station": "VHHH", "tz": "Asia/Hong_Kong",        "wu_offset": 0.0},
    "Munich":        {"slug": "munich",        "lat": 48.35, "lon": 11.79,  "unit": "C", "station": "EDDM", "tz": "Europe/Berlin",         "wu_offset": 0.0},
    "Sao Paulo":     {"slug": "sao-paulo",     "lat": -23.63,"lon": -46.66, "unit": "C", "station": "SBGR", "tz": "America/Sao_Paulo",     "wu_offset": 0.0},
    "Lucknow":       {"slug": "lucknow",       "lat": 26.76, "lon": 80.88,  "unit": "C", "station": "VILK", "tz": "Asia/Kolkata",          "wu_offset": 0.0},
    "Warsaw":        {"slug": "warsaw",        "lat": 52.17, "lon": 20.97,  "unit": "C", "station": "EPWA", "tz": "Europe/Warsaw",         "wu_offset": 0.0},
    "Milan":         {"slug": "milan",         "lat": 45.63, "lon": 8.72,   "unit": "C", "station": "LIMC", "tz": "Europe/Rome",           "wu_offset": 0.0},
    "Madrid":        {"slug": "madrid",        "lat": 40.47, "lon": -3.56,  "unit": "C", "station": "LEMD", "tz": "Europe/Madrid",         "wu_offset": 0.0},
    "Taipei":        {"slug": "taipei",        "lat": 25.08, "lon": 121.23, "unit": "C", "station": "RCTP", "tz": "Asia/Taipei",           "wu_offset": 0.0},
}

GAMMA_BASE = "https://gamma-api.polymarket.com"
NWS_HEADERS = {"User-Agent": "PolymarketWeatherEdge/4.0"}
MONTH_SLUG = {1: "january", 2: "february", 3: "march", 4: "april", 5: "may", 6: "june",
              7: "july", 8: "august", 9: "september", 10: "october", 11: "november", 12: "december"}

# Same model weights as app.py
MODEL_WEIGHTS = {
    0: {"nws": 3.0, "ecmwf_ifs025": 2.5, "gfs_seamless": 2.0, "icon_seamless": 1.5, "gem_seamless": 1.0},
    1: {"nws": 2.5, "ecmwf_ifs025": 2.5, "gfs_seamless": 2.0, "icon_seamless": 1.5, "gem_seamless": 1.0},
    2: {"nws": 2.0, "ecmwf_ifs025": 3.0, "gfs_seamless": 2.0, "icon_seamless": 1.5, "gem_seamless": 1.0},
    3: {"nws": 1.5, "ecmwf_ifs025": 3.0, "gfs_seamless": 2.0, "icon_seamless": 1.5, "gem_seamless": 1.0},
}

LOG_DIR = Path(__file__).parent / "forecast_logs"
LOG_DIR.mkdir(exist_ok=True)


def city_today(city):
    """Get today's date in the CITY's timezone, not your local timezone.

    This matters because when you run the logger at 11pm EST,
    it's already April 8 in Tokyo/Seoul/Singapore/etc.
    Using your local 'today' would mean looking for yesterday's
    market (already resolved) and missing today's active one.
    """
    tz = ZoneInfo(city["tz"])
    return datetime.now(tz).date()


def city_now(city):
    """Get current datetime in the city's timezone."""
    tz = ZoneInfo(city["tz"])
    return datetime.now(tz)


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
def get_log_path(city_slug):
    return LOG_DIR / f"{city_slug}.json"


def load_forecast_log(city_slug):
    path = get_log_path(city_slug)
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except:
            return []
    return []


def save_forecast_log(city_slug, log):
    path = get_log_path(city_slug)
    # Keep last 180 days (extended from 90 for better calibration)
    cutoff = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    log = [entry for entry in log if entry.get("date", "") >= cutoff]
    # Atomic write: dump to a temp file then rename. Without this, a SIGTERM
    # during json.dump leaves the log truncated mid-object — we hit this
    # across 17 forecast log files after a timeout mid-standard-run.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(log, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def upsert_log_entry(log, date_str, forecast=None, model_highs=None, actual=None,
                     actual_source=None, resolution_bin=None):
    """Update or insert a log entry. Preserves existing data when merging."""
    existing = next((e for e in log if e["date"] == date_str), None)
    if existing:
        if forecast is not None:
            existing["forecast"] = forecast
        if model_highs:
            existing["model_highs"] = model_highs
        if actual is not None:
            existing["actual"] = actual
        if actual_source is not None:
            existing["actual_source"] = actual_source
        if resolution_bin is not None:
            existing["resolution_bin"] = resolution_bin
        existing["updated_at"] = datetime.now().isoformat()
    else:
        log.append({
            "date": date_str,
            "forecast": forecast,
            "model_highs": model_highs or {},
            "actual": actual,
            "actual_source": actual_source,
            "resolution_bin": resolution_bin,
            "logged_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        })
    return log


# ─────────────────────────────────────────────
# PATCH L: RETRYABLE HTTP HELPER
# ─────────────────────────────────────────────
def retry_get(url, max_attempts=3, base_delay=1.0, max_delay=9.0, **kwargs):
    """requests.get with exponential backoff on retryable failures.

    Retries on: timeouts, connection errors, chunked-encoding errors, 5xx
    responses. Does NOT retry on 4xx (which almost always indicate a real
    error like 404 / 429 that retry won't fix — though a future improvement
    could honor Retry-After headers for 429).

    Returns the final Response (even if 4xx) on any successful call.
    Raises the last exception on total failure so existing `except: return
    None` blocks in callers still work as before — no behavioral change for
    success cases.

    Backoff schedule (default): 0s, 1s, 3s between attempts.
    """
    last_err = None
    for attempt in range(max_attempts):
        if attempt > 0:
            delay = min(base_delay * (3 ** (attempt - 1)), max_delay)
            time.sleep(delay)
        try:
            resp = requests.get(url, **kwargs)
            # 5xx = server-side hiccup, retry. 4xx = we won't get a better
            # answer, return it and let the caller decide.
            if 500 <= resp.status_code < 600:
                last_err = requests.exceptions.HTTPError(
                    f"HTTP {resp.status_code} from {url}", response=resp)
                continue
            return resp
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            last_err = e
            continue
    # All attempts exhausted — re-raise so caller's try/except returns None.
    if last_err:
        raise last_err
    raise requests.exceptions.RequestException(f"Failed to GET {url}")


# ─────────────────────────────────────────────
# POLYMARKET API
# ─────────────────────────────────────────────
def build_event_slug(city_slug, target_date):
    month = MONTH_SLUG[target_date.month]
    day = target_date.day
    year = target_date.year
    return f"highest-temperature-in-{city_slug}-on-{month}-{day}-{year}"


def fetch_polymarket_event(event_slug):
    try:
        r = retry_get(f"{GAMMA_BASE}/events?slug={event_slug}&limit=1", timeout=10)
        if r.status_code != 200 or not r.text.strip():
            return None
        data = r.json()
        return data[0] if data else None
    except:
        return None


def get_resolution_winner(event):
    """Get the winning bin label from a resolved event."""
    if not event or not event.get("closed"):
        return None
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
            return m.get("groupItemTitle", "")
    return None


# ─────────────────────────────────────────────
# WEATHER FORECAST APIs
# ─────────────────────────────────────────────
def fetch_nws_forecast_high(lat, lon, target_str, city_tz="America/New_York"):
    """Fetch NWS hourly forecast and extract daily high for target date.

    NWS returns timezone-aware timestamps. We convert to the CITY's local
    timezone before comparing against target_str to avoid date-boundary mismatches.
    """
    try:
        r = retry_get(f"https://api.weather.gov/points/{lat},{lon}",
                       headers=NWS_HEADERS, timeout=10)
        grid = r.json()
        hourly_url = grid["properties"]["forecastHourly"]
        r2 = retry_get(hourly_url, headers=NWS_HEADERS, timeout=10)
        periods = r2.json()["properties"]["periods"]

        tz = ZoneInfo(city_tz)
        temps = []
        for p in periods:
            dt = datetime.fromisoformat(p["startTime"])
            # Convert to city's local timezone before extracting date
            local_dt = dt.astimezone(tz)
            if local_dt.strftime("%Y-%m-%d") == target_str:
                temp_f = p["temperature"] if p["temperatureUnit"] == "F" else round(p["temperature"] * 9/5 + 32)
                temps.append(temp_f)

        return max(temps) if temps else None
    except:
        return None


def fetch_openmeteo_multimodel_highs(lat, lon, target_str):
    """Fetch multi-model forecast highs from Open-Meteo.

    Queries each model individually because the combined multi-model endpoint
    doesn't reliably return per-model data in a consistent format.
    One call per model is slightly slower but always works.
    """
    # JMA removed — returns "invalid String value" error from Open-Meteo API
    models = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless"]
    model_highs = {}

    for model in models:
        try:
            r = retry_get(
                f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                f"&hourly=temperature_2m&models={model}"
                f"&temperature_unit=fahrenheit&timezone=auto&forecast_days=7",
                timeout=12)

            if r.status_code != 200:
                continue

            data = r.json()
            if "error" in data:
                continue  # Model returned an error — skip

            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            temps = hourly.get("temperature_2m", [])

            if times and temps:
                day_temps = []
                for i, t in enumerate(times):
                    if t and t.startswith(target_str) and temps[i] is not None:
                        day_temps.append(temps[i])
                if day_temps:
                    model_highs[model] = round(max(day_temps))

            time.sleep(0.15)  # Be polite — 5 calls per city
        except:
            continue

    return model_highs


def fetch_openmeteo_historical_multimodel_highs(lat, lon, start_date, end_date, unit="F"):
    """Fetch *historical* per-model daily-max highs for a date range.

    Uses Open-Meteo's historical-forecast-api (not the archive/reanalysis API).
    This returns the forecasts as they were archived from past model runs, so
    we can retroactively pair them with the observed actuals already in the log.

    Returns {date_str: {model_name: high_in_requested_unit}}.

    Notes:
      • Each model is queried separately — when you pass multiple models in one
        request the daily key becomes `temperature_2m_max_<model>`, whereas a
        single-model request just uses `temperature_2m_max`. Single-queries keep
        the parser simple and match the style of the live fetcher.
      • NWS is not available historically via this endpoint — skip it for
        backfill. The pipeline gains 4 model inputs (ECMWF/GFS/ICON/GEM) per
        historical day, which is enough for a representative ensemble.
    """
    models = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless"]
    temp_unit = "fahrenheit" if unit == "F" else "celsius"
    # {date_str: {model: high}}
    per_date = {}

    for model in models:
        try:
            r = retry_get(
                "https://historical-forecast-api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "start_date": start_date, "end_date": end_date,
                    "daily": "temperature_2m_max",
                    "models": model,
                    "temperature_unit": temp_unit,
                    "timezone": "auto",
                }, timeout=30)
            if r.status_code != 200:
                continue
            data = r.json()
            if "error" in data:
                continue
            dates = data.get("daily", {}).get("time", [])
            temps = data.get("daily", {}).get("temperature_2m_max", [])
            for i, d_str in enumerate(dates):
                if i < len(temps) and temps[i] is not None:
                    per_date.setdefault(d_str, {})[model] = round(temps[i], 1)
            time.sleep(0.3)  # rate-limit between models
        except Exception:
            continue
    return per_date


def compute_ensemble_high(nws_high, model_highs, lead_days, city_unit):
    """Weighted ensemble of all model forecasts. Returns (ensemble_high, model_highs_dict)."""
    all_highs = dict(model_highs)  # copy
    if nws_high is not None:
        all_highs["nws"] = nws_high

    if not all_highs:
        return None, {}

    weights_table = MODEL_WEIGHTS.get(min(lead_days, 3), MODEL_WEIGHTS[3])
    total_weight = 0
    weighted_sum = 0
    for model, high in all_highs.items():
        w = weights_table.get(model, 1.0)
        # Convert to city unit if needed
        if city_unit == "C":
            high_converted = round((high - 32) * 5 / 9, 1)
        else:
            high_converted = high
        weighted_sum += high_converted * w
        total_weight += w

    ensemble_high = round(weighted_sum / total_weight, 1) if total_weight > 0 else None

    # Return model highs in city unit
    converted_highs = {}
    for model, high in all_highs.items():
        if city_unit == "C":
            converted_highs[model] = round((high - 32) * 5 / 9, 1)
        else:
            converted_highs[model] = high

    return ensemble_high, converted_highs


# ─────────────────────────────────────────────
# ACTUAL OBSERVATION (Open-Meteo Archive)
# ─────────────────────────────────────────────
def fetch_openmeteo_daily_high(lat, lon, date_str, city_unit="F"):
    """
    Fetch the Open-Meteo *grid-cell* daily max for the city lat/lon.

    This is reanalysis data at city-center resolution — it does NOT match the
    airport station Polymarket resolves against. Used as fallback only; for
    resolution-grade actuals prefer fetch_metar_daily_high.

    Returns °F or °C per city_unit, or None.
    """
    try:
        r = retry_get(
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max"
            f"&temperature_unit={'fahrenheit' if city_unit == 'F' else 'celsius'}"
            f"&timezone=auto"
            f"&start_date={date_str}&end_date={date_str}",
            timeout=10)

        if r.status_code != 200:
            return None

        data = r.json()
        daily = data.get("daily", {})
        temps = daily.get("temperature_2m_max", [])

        if temps and temps[0] is not None:
            return round(temps[0], 1)
        return None
    except:
        return None


def fetch_metar_daily_high(station_code, date_str, tz_str, city_unit="F"):
    """
    Fetch the ICAO station's daily max temperature from the IEM ASOS archive.

    Queries every METAR/ASOS observation for the station during the local
    calendar date (in tz_str) and returns the max temperature in the city's
    native unit.

    This is the most resolution-faithful source we have: Weather Underground
    (which Polymarket uses) republishes the same ICAO METAR data, so our max
    here should match WU's reported daily high to within rounding.

    Returns float (°F or °C) or None if the station is missing data or
    unreachable.
    """
    try:
        r = retry_get(
            "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
            params={
                "station": station_code,
                "data": "tmpf",  # Fahrenheit — we convert to °C below if needed
                "year1": date_str[:4], "month1": date_str[5:7], "day1": date_str[8:10],
                "year2": date_str[:4], "month2": date_str[5:7], "day2": date_str[8:10],
                "tz": tz_str,
                "format": "onlycomma",
                "latlon": "no",
                "missing": "M",
                "trace": "T",
                "direct": "no",
            },
            timeout=20,
        )
        if r.status_code != 200:
            return None

        lines = r.text.strip().split("\n")
        if len(lines) < 2:
            return None

        # Header is "station,valid,tmpf"; parse rows and collect numeric tmpf
        temps_f = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < 3:
                continue
            tmpf_str = parts[2].strip()
            if tmpf_str in ("M", "", "T"):
                continue
            try:
                temps_f.append(float(tmpf_str))
            except ValueError:
                continue

        if not temps_f:
            return None

        max_f = max(temps_f)
        if city_unit == "F":
            return round(max_f, 1)
        else:
            # Convert to Celsius
            return round((max_f - 32) * 5.0 / 9.0, 1)
    except Exception:
        return None


def fetch_metar_range_daily_highs(station_code, start_date_str, end_date_str,
                                   tz_str, city_unit="F"):
    """
    Fetch per-day max temps for an ICAO station across a date range in one call.

    Used by the backfill path — per-day calls would be 90× slower. IEM's
    archive accepts a date range and a tz, so we get back every obs inside
    [start, end] already localized, then group by local calendar date.

    Returns a dict {date_str: max_temp_in_city_unit}; dates with no data are
    simply absent from the dict rather than mapped to None.
    """
    try:
        r = retry_get(
            "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
            params={
                "station": station_code,
                "data": "tmpf",
                "year1": start_date_str[:4], "month1": start_date_str[5:7], "day1": start_date_str[8:10],
                "year2": end_date_str[:4], "month2": end_date_str[5:7], "day2": end_date_str[8:10],
                "tz": tz_str,
                "format": "onlycomma",
                "latlon": "no",
                "missing": "M",
                "trace": "T",
                "direct": "no",
            },
            timeout=60,
        )
        if r.status_code != 200:
            return {}

        by_date = {}
        lines = r.text.strip().split("\n")
        if len(lines) < 2:
            return {}

        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < 3:
                continue
            valid = parts[1].strip()     # "2026-04-19 14:53" (in tz_str)
            tmpf_str = parts[2].strip()
            if tmpf_str in ("M", "", "T"):
                continue
            try:
                tmpf = float(tmpf_str)
            except ValueError:
                continue

            date_key = valid.split(" ")[0]  # "2026-04-19"
            if date_key not in by_date or tmpf > by_date[date_key]:
                by_date[date_key] = tmpf

        # Convert to the requested unit
        if city_unit == "F":
            return {d: round(v, 1) for d, v in by_date.items()}
        else:
            return {d: round((v - 32) * 5.0 / 9.0, 1) for d, v in by_date.items()}
    except Exception:
        return {}


def fetch_actual_observation(lat, lon, date_str, city_unit="F",
                             station=None, tz=None):
    """
    Get the authoritative daily max for a city/date, preferring the ICAO
    station observation (matches Polymarket's WU resolution) and falling
    back to the Open-Meteo archive if the station is unreachable.

    Returns a tuple (value, source) so callers can record provenance:
      source is "metar_iem" if the station pull succeeded,
               "open_meteo_archive" if we fell back,
               None if neither source returned data.
    """
    # Try the station first — matches resolution source
    if station and tz:
        metar_val = fetch_metar_daily_high(station, date_str, tz, city_unit)
        if metar_val is not None and is_sane_temp(metar_val, city_unit):
            return metar_val, "metar_iem"

    # Fall back to Open-Meteo grid-cell reanalysis
    om_val = fetch_openmeteo_daily_high(lat, lon, date_str, city_unit)
    if om_val is not None and is_sane_temp(om_val, city_unit):
        return om_val, "open_meteo_archive"

    return None, None


def is_sane_temp(temp, city_unit):
    """Reject obviously wrong temperatures."""
    if temp is None:
        return False
    if city_unit == "F":
        return -40 <= temp <= 140
    else:
        return -40 <= temp <= 60


# ─────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────
def log_forecasts_for_city(city_name, city, days_ahead=3, verbose=True):
    """Log ensemble forecasts for all active markets in a city."""
    slug = city["slug"]
    log = load_forecast_log(slug)
    today = city_today(city)  # Use CITY's local date, not your machine's
    logged_count = 0

    for i in range(0, days_ahead):
        target_date = today + timedelta(days=i)
        target_str = target_date.strftime("%Y-%m-%d")
        lead_days = i

        # Check if market exists
        event_slug = build_event_slug(slug, target_date)
        event = fetch_polymarket_event(event_slug)
        if event is None or event.get("closed"):
            continue

        # Fetch forecasts
        nws_high = None
        if city["unit"] == "F":
            nws_high = fetch_nws_forecast_high(city["lat"], city["lon"], target_str, city["tz"])

        model_highs = fetch_openmeteo_multimodel_highs(city["lat"], city["lon"], target_str)
        ensemble_high, converted_highs = compute_ensemble_high(
            nws_high, model_highs, lead_days, city["unit"]
        )

        if ensemble_high is not None:
            log = upsert_log_entry(log, target_str,
                                   forecast=ensemble_high,
                                   model_highs=converted_highs)
            logged_count += 1
            if verbose:
                spread = list(converted_highs.values())
                spread_str = f"±{np.std(spread):.1f}°" if len(spread) >= 2 else "N/A"
                print(f"  [{slug}] {target_str} D+{lead_days}: "
                      f"ensemble={ensemble_high}°{city['unit']} "
                      f"({len(converted_highs)} models, spread {spread_str})")

        # Be polite to APIs
        time.sleep(0.3)

    save_forecast_log(slug, log)
    return logged_count


def log_actuals_for_city(city_name, city, lookback_days=14, verbose=True):
    """
    Fetch real station observations for recently resolved dates.

    Prefers METAR (from IEM) for the city's ICAO station — this matches what
    Weather Underground (Polymarket's resolver) reports. Falls back to
    Open-Meteo archive if the station is unavailable.

    Entries already sourced from "metar_iem" are skipped; entries sourced from
    "open_meteo_archive" get upgraded to METAR if a station fetch succeeds.
    """
    slug = city["slug"]
    log = load_forecast_log(slug)
    today = city_today(city)  # Use CITY's local date
    logged_count = 0
    station = city.get("station")
    tz = city.get("tz")

    for i in range(1, lookback_days + 1):
        d = today - timedelta(days=i)
        date_str = d.strftime("%Y-%m-%d")

        # Skip if we already have the best source (METAR). Leave open_meteo
        # entries alone so they can be upgraded on this pass.
        existing = next((e for e in log if e["date"] == date_str), None)
        if existing and existing.get("actual_source") == "metar_iem":
            continue

        actual, source = fetch_actual_observation(
            city["lat"], city["lon"], date_str, city["unit"],
            station=station, tz=tz,
        )

        if actual is not None and is_sane_temp(actual, city["unit"]):
            # If the existing entry is already open_meteo and we only got
            # open_meteo again, no need to rewrite.
            if existing and existing.get("actual_source") == source \
                    and existing.get("actual") == actual:
                continue

            # Also check if there was a Polymarket resolution for this date
            event_slug = build_event_slug(slug, d)
            event = fetch_polymarket_event(event_slug)
            resolution_bin = get_resolution_winner(event) if event else None

            log = upsert_log_entry(log, date_str,
                                   actual=actual,
                                   actual_source=source,
                                   resolution_bin=resolution_bin)
            logged_count += 1
            if verbose:
                res_str = f" | resolution: {resolution_bin}" if resolution_bin else ""
                src_tag = "🛰" if source == "metar_iem" else "📍"
                print(f"  {src_tag} [{slug}] {date_str}: actual={actual}°{city['unit']} ({source}){res_str}")

        time.sleep(0.2)

    save_forecast_log(slug, log)
    return logged_count


def backfill_actuals_for_city(city_name, city, days=90, verbose=True):
    """
    One-time backfill: fetch actual observed temps for the last N days.
    Seeds (or upgrades) the calibration log with real station observations.

    Strategy: try METAR (via IEM) first for the whole range in a single call,
    then fill any missing dates from Open-Meteo archive. This prefers the
    resolution-grade station source and degrades gracefully for stations
    the IEM archive doesn't cover.

    Upgrade behavior: entries already sourced from "open_meteo_archive" are
    overwritten when a METAR reading becomes available — that's the whole
    point of this pass for the Tier-3 station-mismatch cities.
    """
    slug = city["slug"]
    log = load_forecast_log(slug)

    # First, purge any bad entries (e.g., -0.5 from bin-midpoint parsing bugs)
    purged = 0
    for entry in log:
        if entry.get("actual") is not None and not is_sane_temp(entry["actual"], city["unit"]):
            if verbose:
                print(f"  [{slug}] PURGE bad actual: {entry['date']} = {entry['actual']}")
            entry["actual"] = None
            entry["actual_source"] = None
            purged += 1

    if purged > 0 and verbose:
        print(f"  [{slug}] Purged {purged} bad entries")

    today = city_today(city)  # Use CITY's local date
    start_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    station = city.get("station")
    tz = city.get("tz")
    unit = city["unit"]

    logged_count = 0
    upgraded_count = 0

    # ── Step 1: METAR range pull (preferred source)
    metar_highs = {}
    if station and tz:
        if verbose:
            print(f"  [{slug}] Fetching METAR range {start_date} → {end_date} from {station}...")
        metar_highs = fetch_metar_range_daily_highs(
            station, start_date, end_date, tz, unit
        )
        if verbose:
            print(f"  [{slug}] Got {len(metar_highs)} days of METAR data")

    for date_str, actual in metar_highs.items():
        if not is_sane_temp(actual, unit):
            continue
        existing = next((e for e in log if e["date"] == date_str), None)
        if existing and existing.get("actual_source") == "metar_iem" \
                and existing.get("actual") == actual:
            continue  # Already have this exact METAR reading
        if existing and existing.get("actual_source") == "open_meteo_archive":
            upgraded_count += 1
        log = upsert_log_entry(log, date_str,
                               actual=actual,
                               actual_source="metar_iem")
        logged_count += 1

    # ── Step 2: Open-Meteo fallback for dates METAR couldn't cover
    if verbose:
        print(f"  [{slug}] Filling any METAR gaps from Open-Meteo archive...")
    try:
        r = retry_get(
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={city['lat']}&longitude={city['lon']}"
            f"&daily=temperature_2m_max"
            f"&temperature_unit={'fahrenheit' if unit == 'F' else 'celsius'}"
            f"&timezone=auto"
            f"&start_date={start_date}&end_date={end_date}",
            timeout=30)
        if r.status_code == 200:
            data = r.json()
            dates = data.get("daily", {}).get("time", [])
            temps = data.get("daily", {}).get("temperature_2m_max", [])
            for j, date_str in enumerate(dates):
                if j >= len(temps) or temps[j] is None:
                    continue
                actual = round(temps[j], 1)
                if not is_sane_temp(actual, unit):
                    continue
                existing = next((e for e in log if e["date"] == date_str), None)
                # Don't overwrite METAR with Open-Meteo; only fill gaps.
                if existing and existing.get("actual_source") == "metar_iem":
                    continue
                if existing and existing.get("actual_source") == "open_meteo_archive" \
                        and existing.get("actual") == actual:
                    continue
                log = upsert_log_entry(log, date_str,
                                       actual=actual,
                                       actual_source="open_meteo_archive")
                logged_count += 1
    except Exception as e:
        if verbose:
            print(f"  [{slug}] Open-Meteo fallback error: {e}")

    # ── Step 3: Cross-reference recent Polymarket resolutions
    if verbose:
        print(f"  [{slug}] Cross-referencing Polymarket resolutions (last 14 days)...")
    for i in range(1, min(15, days)):
        d = today - timedelta(days=i)
        date_str = d.strftime("%Y-%m-%d")
        event_slug = build_event_slug(slug, d)
        event = fetch_polymarket_event(event_slug)
        if event:
            resolution_bin = get_resolution_winner(event)
            if resolution_bin:
                existing = next((e for e in log if e["date"] == date_str), None)
                if existing:
                    existing["resolution_bin"] = resolution_bin
        time.sleep(0.2)

    save_forecast_log(slug, log)
    if verbose:
        msg = f"  [{slug}] Backfill complete: {logged_count} actuals written"
        if upgraded_count > 0:
            msg += f" ({upgraded_count} upgraded from Open-Meteo to METAR)"
        print(msg)
    return logged_count


def backfill_forecasts_for_city(city_name, city, days=90, verbose=True):
    """
    One-time backfill of HISTORICAL forecasts for a city.

    Why this exists:
      At V3 we had plenty of actuals (Open-Meteo archive is cheap and works
      for any date) but forecasts were only being written from the moment
      forecast_logger.py started running on a given city. That left 19 of 26
      cities stuck at 2-3 paired observations — below the `min_calibration_obs`
      threshold that the paper_trader uses to gate trading. Entire continents
      were untradable.

      This function backfills per-model forecasts for the last N days from
      Open-Meteo's historical-forecast-api and merges them into each city's
      forecast log alongside the existing actuals. Once this runs, any date
      with both a backfilled forecast and a pre-existing actual becomes a
      calibration pair.

    What it does NOT do:
      Touch NWS (no historical API for that), or overwrite existing forecast
      fields. Dates that already have a forecast are skipped so scheduled
      forward-logging stays authoritative.
    """
    slug = city["slug"]
    log = load_forecast_log(slug)

    today = city_today(city)
    # Go back N days and stop at yesterday — we want forecasts for dates that
    # already have an actual available.
    start_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    if verbose:
        print(f"  [{slug}] Backfilling forecasts {start_date} → {end_date}...")

    per_date = fetch_openmeteo_historical_multimodel_highs(
        city["lat"], city["lon"], start_date, end_date, unit=city["unit"])

    if not per_date:
        if verbose:
            print(f"  [{slug}] No historical forecasts returned")
        return 0

    added = 0
    for date_str, model_highs in per_date.items():
        if not model_highs:
            continue

        # Don't clobber a forecast that's already there (live logger is truth).
        existing = next((e for e in log if e["date"] == date_str), None)
        if existing and existing.get("forecast") is not None:
            continue

        # Use compute_ensemble_high with a "representative" lead of 1 day
        # (the most common paper-trading window). We're not trying to
        # reconstruct the exact issue-time lead — just to produce a
        # reasonable ensemble high for calibration purposes.
        # Historical fetcher already returns values in the requested unit,
        # but compute_ensemble_high expects Fahrenheit internally. Convert
        # if this is a °C city before passing in.
        highs_in_f = {}
        if city["unit"] == "C":
            for m, h in model_highs.items():
                highs_in_f[m] = round(h * 9 / 5 + 32)
        else:
            highs_in_f = dict(model_highs)

        ensemble_high, converted_highs = compute_ensemble_high(
            nws_high=None,
            model_highs=highs_in_f,
            lead_days=1,
            city_unit=city["unit"],
        )

        if ensemble_high is None:
            continue

        log = upsert_log_entry(log, date_str,
                               forecast=ensemble_high,
                               model_highs=converted_highs)
        added += 1

    save_forecast_log(slug, log)
    if verbose:
        print(f"  [{slug}] Backfilled {added} historical forecasts")
    return added


def print_calibration_status():
    """Print a summary of calibration data across all cities."""
    print("\n" + "=" * 70)
    print("CALIBRATION STATUS")
    print("=" * 70)
    print(f"{'City':<20} {'Paired':>7} {'FC only':>8} {'Act only':>9} {'Bias':>8} {'Std':>8}")
    print("-" * 70)

    for city_name, city in CITIES.items():
        log = load_forecast_log(city["slug"])
        paired = sum(1 for e in log if e.get("forecast") is not None and e.get("actual") is not None)
        fc_only = sum(1 for e in log if e.get("forecast") is not None and e.get("actual") is None)
        act_only = sum(1 for e in log if e.get("forecast") is None and e.get("actual") is not None)

        errors = [e["actual"] - e["forecast"]
                  for e in log
                  if e.get("forecast") is not None and e.get("actual") is not None]

        bias_str = f"{np.mean(errors):+.1f}°" if len(errors) >= 3 else "---"
        std_str = f"±{np.std(errors):.1f}°" if len(errors) >= 3 else "---"

        status = "✅" if paired >= 30 else "🔄" if paired >= 3 else "❌"
        print(f"{status} {city_name:<18} {paired:>7} {fc_only:>8} {act_only:>9} {bias_str:>8} {std_str:>8}")

    print("=" * 70)
    print("✅ = calibrated (30+ pairs)  🔄 = partial (3+ pairs)  ❌ = no data")
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Polymarket Weather Forecast Logger")
    parser.add_argument("--city", type=str, help="Run for a single city slug (e.g., 'nyc')")
    parser.add_argument("--backfill", action="store_true", help="Backfill last 90 days of actuals")
    parser.add_argument("--backfill-forecasts", action="store_true",
                        help="Backfill last N days of HISTORICAL forecasts (pairs w/ existing actuals)")
    parser.add_argument("--backfill-days", type=int, default=90, help="Number of days to backfill")
    parser.add_argument("--status", action="store_true", help="Print calibration status and exit")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    verbose = not args.quiet
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if args.status:
        print_calibration_status()
        return

    # Filter to single city if requested
    if args.city:
        cities_to_process = {
            name: city for name, city in CITIES.items()
            if city["slug"] == args.city
        }
        if not cities_to_process:
            print(f"Error: city slug '{args.city}' not found. Available: "
                  f"{', '.join(c['slug'] for c in CITIES.values())}")
            sys.exit(1)
    else:
        cities_to_process = CITIES

    if args.backfill and args.backfill_forecasts:
        mode_str = "backfill (actuals + forecasts)"
    elif args.backfill:
        mode_str = "backfill (actuals)"
    elif args.backfill_forecasts:
        mode_str = "backfill (forecasts)"
    else:
        mode_str = "standard"

    print(f"\n{'='*60}")
    print(f"Forecast Logger — {timestamp}")
    print(f"Cities: {len(cities_to_process)} | Mode: {mode_str}")
    print(f"{'='*60}\n")

    total_forecasts = 0
    total_actuals = 0
    total_hist_forecasts = 0

    for city_name, city in cities_to_process.items():
        if verbose:
            print(f"\n📍 {city_name} ({city['station']}, {city['unit']})")

        if args.backfill or args.backfill_forecasts:
            # Either or both backfill modes.
            if args.backfill:
                n = backfill_actuals_for_city(
                    city_name, city, days=args.backfill_days, verbose=verbose)
                total_actuals += n
            if args.backfill_forecasts:
                nh = backfill_forecasts_for_city(
                    city_name, city, days=args.backfill_days, verbose=verbose)
                total_hist_forecasts += nh
        else:
            # Standard mode: log forecasts + recent actuals
            nf = log_forecasts_for_city(city_name, city, days_ahead=3, verbose=verbose)
            na = log_actuals_for_city(city_name, city, lookback_days=14, verbose=verbose)
            total_forecasts += nf
            total_actuals += na

        # Rate limiting between cities
        time.sleep(0.5)

    print(f"\n{'='*60}")
    if args.backfill or args.backfill_forecasts:
        parts = []
        if args.backfill:
            parts.append(f"{total_actuals} actual obs")
        if args.backfill_forecasts:
            parts.append(f"{total_hist_forecasts} historical forecasts")
        print(f"Backfill complete: {' + '.join(parts)}")
    else:
        print(f"Done: {total_forecasts} forecasts + {total_actuals} actuals logged")
    print(f"{'='*60}\n")

    # Always show calibration status at the end
    print_calibration_status()


if __name__ == "__main__":
    main()
