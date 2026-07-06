"""
Purge corrupt entries from all forecast log files.

Corrupt signals identified during model analysis:
  1. actual values of -0.5°F in San Francisco (April 2026) — clearly bogus for
     a city whose record April low is 38°F.
  2. Other entries where |actual - forecast| > 15°F (or > 10°C) — almost
     certainly a data ingestion error, not a genuine forecast miss.

This script runs through every city log, identifies corrupt entries, prints
what it would remove, and (with --commit) actually removes them.

Usage:
  python purge_corrupt_logs.py              # dry run — show what would be purged
  python purge_corrupt_logs.py --commit     # actually purge
"""

import json
import sys
from pathlib import Path
from forecast_logger import CITIES, load_forecast_log, is_sane_temp

LOG_DIR = Path(__file__).parent / "forecast_logs"


def is_entry_plausible(entry, unit, forecast=None, actual=None):
    """An entry is corrupt if |actual - forecast| is implausibly large."""
    f = forecast if forecast is not None else entry.get("forecast")
    a = actual if actual is not None else entry.get("actual")
    if f is None or a is None:
        return False, None  # can't judge — neither present
    max_err = 15.0 if unit == "F" else 10.0
    err = abs(a - f)
    return err < max_err, err


def is_actual_sane(entry, unit, city_name):
    """Check if the actual temperature is physically plausible for this city.

    Known corruption pattern: San Francisco had actual=-0.5°F in April 2026
    from a buggy Open-Meteo archive response. SF's record April low is ~38°F.

    Other patterns: any non-negative zero/-1/-0.5 values in warm-season months
    for cities where it would be extraordinary.
    """
    actual = entry.get("actual")
    if actual is None:
        return True

    date_str = entry.get("date", "")
    if not date_str:
        return True

    try:
        month = int(date_str[5:7])
    except (ValueError, IndexError):
        return True

    # Season/city specific sanity floors (°F / °C)
    city_slug = entry.get("slug", city_name)

    if unit == "F":
        # San Francisco: never below 38°F in April-October
        if city_name == "San Francisco" and 4 <= month <= 10:
            if actual < 35:
                return False
        # General warm-season sanity: below 10°F is implausible for any
        # non-Alaska US city in Apr-Oct
        if 4 <= month <= 10 and actual < 10:
            return False
    else:
        if unit == "C":
            # Below -10°C in Apr-Sep is implausible for most international cities
            if 4 <= month <= 9 and actual < -10:
                return False

    return True


def purge_city_log(city_name, city, dry_run=True):
    """Purge corrupt entries from one city's forecast log. Returns count purged."""
    slug = city["slug"]
    unit = city["unit"]
    path = LOG_DIR / f"{slug}.json"

    if not path.exists():
        return 0, []

    log = json.loads(path.read_text())
    removed = []
    kept = []

    for entry in log:
        remove = False
        reason = None

        # Check 1: is the actual temperature physically plausible?
        if entry.get("actual") is not None and not is_sane_temp(entry["actual"], unit):
            remove = True
            reason = f"failed is_sane_temp (actual={entry['actual']})"
        elif entry.get("actual") is not None and not is_actual_sane(entry, unit, city_name):
            remove = True
            reason = f"actual={entry['actual']} implausible for {city_name} in month {entry['date'][5:7]}"

        # Check 2: if both forecast and actual exist, is the error plausible?
        fc = entry.get("forecast")
        ac = entry.get("actual")
        if not remove and fc is not None and ac is not None:
            plausible, err = is_entry_plausible(entry, unit, forecast=fc, actual=ac)
            if not plausible:
                remove = True
                reason = f"|error|={err:.1f} > max (forecast={fc}, actual={ac})"

        # Check 3: any model_highs entry with an absurd value?
        if not remove:
            for model, high in (entry.get("model_highs") or {}).items():
                if high is not None and not is_sane_temp(high, unit):
                    # Don't remove the whole entry — just null out the bad model value
                    entry["model_highs"][model] = None
                    if not any(v is not None for v in entry["model_highs"].values()):
                        entry["model_highs"] = {}

        if remove:
            removed.append((entry["date"], reason))
        else:
            kept.append(entry)

    if removed:
        print(f"  [{slug}] {city_name}: {len(removed)} corrupt entries to purge:")
        for date, reason in removed[:10]:
            print(f"    {date} — {reason}")
        if len(removed) > 10:
            print(f"    ... and {len(removed) - 10} more")

    if not dry_run and removed:
        # Atomic write
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(kept, f, indent=2)
            f.flush()
        tmp.replace(path)
        print(f"  → Written {len(kept)} clean entries to {path.name}")

    return len(removed), removed


def main():
    dry_run = "--commit" not in sys.argv

    if dry_run:
        print("=== DRY RUN — use --commit to actually purge ===\n")
    else:
        print("=== COMMITTING: purging corrupt entries ===\n")

    total = 0
    removed_entries = []

    for city_name, city in CITIES.items():
        count, removed = purge_city_log(city_name, city, dry_run=dry_run)
        total += count
        removed_entries.extend((city_name, d, r) for d, r in removed)

    print(f"\nTotal: {total} corrupt entries across {len(CITIES)} cities")
    if total == 0:
        print("No corrupt entries found. Logs are clean.")
    elif dry_run:
        print("Re-run with --commit to apply these purges.")


if __name__ == "__main__":
    main()
