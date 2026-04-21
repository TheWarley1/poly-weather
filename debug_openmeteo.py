"""Quick diagnostic: what does Open-Meteo actually return for multi-model queries?"""
import requests, json

print("=" * 60)
print("TEST 1: Combined multi-model query")
print("=" * 60)

r = requests.get(
    "https://api.open-meteo.com/v1/forecast?latitude=40.77&longitude=-73.87"
    "&hourly=temperature_2m&models=ecmwf_ifs025,gfs_seamless,icon_seamless,gem_seamless,jma"
    "&temperature_unit=fahrenheit&timezone=auto&forecast_days=1",
    timeout=15)

data = r.json()

print(f"Status: {r.status_code}")
print(f"Top-level keys: {sorted(data.keys())}")

if "error" in data:
    print(f"ERROR: {data['error']}")
if "reason" in data:
    print(f"REASON: {data['reason']}")

if "hourly" in data:
    hourly = data["hourly"]
    print(f"\nhourly keys ({len(hourly)}): {sorted(hourly.keys())}")
    # Show first 3 values of each temperature column
    for k in sorted(hourly.keys()):
        if k != "time":
            sample = [v for v in hourly[k][:3] if v is not None]
            print(f"  {k}: {sample}")

# Check for model-specific top-level blocks
for k in sorted(data.keys()):
    if k.startswith("hourly_"):
        print(f"\nFOUND separate block: {k} -> keys: {list(data[k].keys())[:5]}")

print("\n" + "=" * 60)
print("TEST 2: Individual model queries (one at a time)")
print("=" * 60)

models = {
    "ecmwf_ifs025": "https://api.open-meteo.com/v1/forecast",
    "gfs_seamless": "https://api.open-meteo.com/v1/forecast",
    "icon_seamless": "https://api.open-meteo.com/v1/forecast",
    "gem_seamless": "https://api.open-meteo.com/v1/forecast",
    "jma": "https://api.open-meteo.com/v1/forecast",
}

for model, url in models.items():
    try:
        r = requests.get(
            f"{url}?latitude=40.77&longitude=-73.87"
            f"&hourly=temperature_2m&models={model}"
            f"&temperature_unit=fahrenheit&timezone=auto&forecast_days=1",
            timeout=10)
        d = r.json()
        if "error" in d:
            print(f"  {model}: ERROR - {d['error']}")
        elif "hourly" in d:
            temps = d["hourly"].get("temperature_2m", [])
            valid = [t for t in temps[:24] if t is not None]
            high = max(valid) if valid else "N/A"
            print(f"  {model}: OK - high={high}F, {len(valid)}/24 hours")
        else:
            print(f"  {model}: no 'hourly' key, keys={list(d.keys())}")
    except Exception as e:
        print(f"  {model}: FAILED - {e}")
