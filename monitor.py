"""
Daily Performance Monitor
==========================
Reads paper_trades.json and forecast_logs to produce a one-page performance
snapshot. Writes incremental JSON snapshots so you can track trends without
opening GitHub Actions logs.

Run manually:
  python monitor.py

Run from GitHub Actions (added to trading.yml):
  python monitor.py >> pipeline.log

Output goes to:
  monitor_logs/snapshot_YYYY-MM-DD.json  — daily state
  monitor_logs/latest.json               — always the most recent
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).parent
TRADES_FILE = ROOT / "paper_trades.json"
MONITOR_DIR = ROOT / "monitor_logs"
MONITOR_DIR.mkdir(exist_ok=True)

from forecast_logger import CITIES, load_forecast_log, compute_model_calibration


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_trades():
    if not TRADES_FILE.exists():
        return []
    try:
        return json.loads(TRADES_FILE.read_text())
    except Exception:
        return []


def load_bss():
    """Read backtest_metar.csv for per-city BSS scores."""
    bss_path = ROOT / "backtest_metar.csv"
    bss = {}
    if not bss_path.exists():
        return bss
    try:
        lines = bss_path.read_text().strip().split('\n')
        if len(lines) < 2:
            return bss
        headers = lines[0].split(',')
        for line in lines[1:]:
            vals = line.split(',')
            if len(vals) >= 8:
                slug = vals[1]
                try:
                    bss[slug] = float(vals[7])
                except (ValueError, IndexError):
                    pass
    except Exception:
        pass
    return bss


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_trade_stats(trades):
    """Core trade metrics."""
    resolved = [t for t in trades if t.get("resolved") and not t.get("voided")]
    voided = [t for t in trades if t.get("voided")]
    open_trades = [t for t in trades if not t.get("resolved")]
    wins = [t for t in resolved if t.get("pnl", 0) > 0]
    losses = [t for t in resolved if t.get("pnl", 0) < 0]

    total_cost = sum(t.get("cost", 0) for t in resolved)
    total_pnl = sum(t.get("pnl", 0) for t in resolved)

    # Bankroll from latest trade's bankroll_at_entry, or starting
    bankroll = 1000.0
    if trades:
        # Find the most recent trade with bankroll_at_entry
        for t in sorted(trades, key=lambda x: x.get("logged_at", ""), reverse=True):
            if "bankroll_at_entry" in t:
                bankroll = t["bankroll_at_entry"]
                break

    return {
        "total": len(trades),
        "open": len(open_trades),
        "resolved": len(resolved),
        "voided": len(voided),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(resolved) if resolved else None,
        "total_cost": round(total_cost, 2),
        "total_pnl": round(total_pnl, 2),
        "roi": round(total_pnl / total_cost * 100, 1) if total_cost > 0 else None,
        "bankroll": round(bankroll, 2),
        "drawdown_pct": round((1000.0 - bankroll) / 1000.0 * 100, 1) if bankroll < 1000 else 0.0,
        "avg_win": round(sum(t["pnl"] for t in wins) / len(wins), 2) if wins else None,
        "avg_loss": round(abs(sum(t["pnl"] for t in losses)) / len(losses), 2) if losses else None,
        "profit_factor": round(sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses)), 2)
            if wins and losses and abs(sum(t["pnl"] for t in losses)) > 0 else None,
    }


def compute_lead_day_stats(trades):
    """Breakdown by lead_days."""
    resolved = [t for t in trades if t.get("resolved") and not t.get("voided")]
    groups = defaultdict(list)
    for t in resolved:
        groups[t.get("lead_days", -1)].append(t)

    result = {}
    for lead in sorted(groups.keys()):
        grp = groups[lead]
        wins = sum(1 for t in grp if t.get("pnl", 0) > 0)
        cost = sum(t.get("cost", 0) for t in grp)
        pnl = sum(t.get("pnl", 0) for t in grp)
        result[f"D+{lead}"] = {
            "trades": len(grp),
            "wins": wins,
            "win_rate": round(wins / len(grp) * 100, 1) if grp else None,
            "cost": round(cost, 2),
            "pnl": round(pnl, 2),
            "roi": round(pnl / cost * 100, 1) if cost > 0 else None,
        }
    return result


def compute_city_stats(trades):
    """Top and bottom cities by P&L."""
    resolved = [t for t in trades if t.get("resolved") and not t.get("voided")]
    city_groups = defaultdict(list)
    for t in resolved:
        city_groups[t.get("city", "Unknown")].append(t)

    cities = []
    for city, grp in city_groups.items():
        wins = sum(1 for t in grp if t.get("pnl", 0) > 0)
        pnl = sum(t.get("pnl", 0) for t in grp)
        cities.append({
            "city": city,
            "trades": len(grp),
            "wins": wins,
            "win_rate": round(wins / len(grp) * 100, 1) if grp else None,
            "pnl": round(pnl, 2),
        })

    cities.sort(key=lambda x: x["pnl"], reverse=True)
    return cities[:5], cities[-5:]  # top 5, bottom 5


def compute_calibration_status():
    """Per-city calibration health from forecast logs and model dropping."""
    status = {}
    for city_name, city in CITIES.items():
        slug = city["slug"]
        log = load_forecast_log(slug)
        paired = sum(1 for e in log if e.get("forecast") and e.get("actual"))
        w, b = compute_model_calibration(slug, city["unit"])

        all_models = {"ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless"}
        kept = set(w.keys()) if w else set()
        dropped = all_models - kept

        status[slug] = {
            "city": city_name,
            "paired": paired,
            "models_used": len(kept),
            "models_dropped": sorted(dropped) if dropped else [],
            "nws_dropped": True,  # globally dropped in Fix 1
        }
    return status


def compute_recent_signals(trades, days=7):
    """Count signals generated in the last N days."""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    recent_open = [t for t in trades
                   if not t.get("resolved") and t.get("logged_at", "") >= cutoff]
    recent_resolved = [t for t in trades
                       if t.get("resolved") and t.get("resolved_at", "") >= cutoff]
    return {
        "signals_generated": len(recent_open),
        "resolved_in_period": len(recent_resolved),
        "days": days,
    }


# ─────────────────────────────────────────────
# SNAPSHOT
# ─────────────────────────────────────────────

def build_snapshot():
    """Build the full snapshot dict."""
    trades = load_trades()

    snap = {
        "timestamp": datetime.now().isoformat(),
        "trading": compute_trade_stats(trades),
        "lead_days": compute_lead_day_stats(trades),
    }

    top5, bot5 = compute_city_stats(trades)
    snap["top_cities"] = top5
    snap["bottom_cities"] = bot5
    snap["calibration"] = compute_calibration_status()
    snap["recent"] = compute_recent_signals(trades)

    # Drawdown circuit breaker status
    bankroll = snap["trading"]["bankroll"]
    dd_pct = snap["trading"]["drawdown_pct"]
    if dd_pct >= 40:
        snap["drawdown"] = "HALTED"
    elif dd_pct >= 20:
        snap["drawdown"] = "HALVED"
    else:
        snap["drawdown"] = "OK"

    return snap


def save_snapshot(snap):
    """Write snapshot files."""
    ts = datetime.now().strftime("%Y-%m-%d")
    daily_path = MONITOR_DIR / f"snapshot_{ts}.json"
    latest_path = MONITOR_DIR / "latest.json"

    daily_path.write_text(json.dumps(snap, indent=2))
    latest_path.write_text(json.dumps(snap, indent=2))


def load_previous_snapshot():
    """Load the previous day's snapshot for delta comparison."""
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    path = MONITOR_DIR / f"snapshot_{yesterday}.json"
    if not path.exists():
        # Try 2 days back
        two_days = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        path = MONITOR_DIR / f"snapshot_{two_days}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


# ─────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────

def print_report(snap, prev=None):
    """Print the formatted report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%MZ")
    tr = snap["trading"]
    dd = snap["drawdown"]
    ld = snap["lead_days"]

    print()
    print("═" * 55)
    print(f"  POLY-WEATHER DAILY MONITOR — {ts}")
    print("═" * 55)

    # Trading summary
    print(f"\n  📊 TRADING")
    print(f"  {'─' * 50}")
    wr_str = f"{tr['win_rate']*100:.1f}%" if tr["win_rate"] is not None else "—"
    roi_str = f"{tr['roi']:+.1f}%" if tr["roi"] is not None else "—"
    pf_str = f"{tr['profit_factor']:.2f}x" if tr["profit_factor"] else "—"
    print(f"  Signals: {tr['total']:>4d} total | {tr['open']:>3d} open | "
          f"{tr['resolved']:>3d} resolved | {tr['voided']} voided")
    print(f"  Win rate: {wr_str:>6s} ({tr['wins']}W / {tr['losses']}L)")

    # Delta from previous
    if prev:
        prev_tr = prev["trading"]
        delta_pnl = tr["total_pnl"] - prev_tr["total_pnl"]
        delta_trades = tr["resolved"] - prev_tr["resolved"]
        if delta_trades > 0:
            print(f"  Since yesterday: {delta_trades} new resolutions, "
                  f"P&L ${delta_pnl:+.2f}")

    print(f"  P&L:    ${tr['total_pnl']:>+.2f} | ROI: {roi_str}")
    print(f"  Bankroll: ${tr['bankroll']:.2f} "
          f"(start $1,000, drawdown {tr['drawdown_pct']:.1f}%)")

    dd_label = {"OK": "✅ Active (1.0×)", "HALVED": "⚠  Halved (0.5×)", "HALTED": "🛑 Halted (0×)"}
    print(f"  Drawdown: {dd_label.get(dd, dd)}")

    # By lead day
    if ld:
        print(f"\n  📅 BY LEAD DAY")
        print(f"  {'─' * 50}")
        print(f"  {'':>4s} {'Trd':>5s} {'WR':>7s} {'P&L':>10s} {'ROI':>8s}")
        for label, stats in ld.items():
            wr = f"{stats['win_rate']:.1f}%" if stats['win_rate'] is not None else "—"
            roi = f"{stats['roi']:+.1f}%" if stats['roi'] is not None else "—"
            print(f"  {label:>4s} {stats['trades']:>5d} {wr:>7s} "
                  f"${stats['pnl']:>+9.2f} {roi:>8s}")

    # Top/Bottom cities
    top = snap["top_cities"]
    bot = snap["bottom_cities"]
    if any(c["pnl"] != 0 for c in top + bot):
        print(f"\n  🏙  CITIES")
        print(f"  {'─' * 50}")
        print(f"  {'Best:':>5s}", end="")
        for c in top[:3]:
            if c["trades"] > 0:
                wr_c = f"{c['win_rate']:.0f}%" if c['win_rate'] is not None else "—"
                print(f"  {c['city']} ${c['pnl']:+.0f} ({wr_c})", end="")
        print()
        print(f"  {'Worst:':>5s}", end="")
        for c in bot[:3]:
            if c["trades"] > 0:
                wr_c = f"{c['win_rate']:.0f}%" if c['win_rate'] is not None else "—"
                print(f"  {c['city']} ${c['pnl']:+.0f} ({wr_c})", end="")
        print()

    # Calibration summary
    cal = snap["calibration"]
    n_calibrated = sum(1 for s in cal.values() if s["paired"] >= 30)
    n_dropping = sum(1 for s in cal.values() if s["models_dropped"])
    print(f"\n  🔬 CALIBRATION")
    print(f"  {'─' * 50}")
    print(f"  {n_calibrated}/26 cities calibrated (30+ pairs)")
    if n_dropping:
        dropped_strs = []
        for s in cal.values():
            if s["models_dropped"]:
                short_names = [m.split('_')[0] for m in s["models_dropped"]]
                dropped_strs.append(f"{s['city']}({','.join(short_names)})")
        print(f"  Models dropped: {', '.join(dropped_strs)}")

    # Recent activity
    rec = snap["recent"]
    if rec["signals_generated"] > 0 or rec["resolved_in_period"] > 0:
        print(f"\n  🕐 LAST 7 DAYS")
        print(f"  {'─' * 50}")
        print(f"  {rec['signals_generated']} signals generated, "
              f"{rec['resolved_in_period']} resolved")

    # Quick verdict
    print(f"\n  {'═' * 55}")
    trd = snap["trading"]
    if trd["total"] == 0:
        print(f"  📭 No trades yet — pipeline starting fresh.")
    elif trd["open"] > 0 and trd["resolved"] == 0:
        print(f"  ⏳ {trd['open']} open trades — awaiting resolution.")
    elif trd["roi"] is not None and trd["roi"] > 0:
        print(f"  ✅ Green — ROI {trd['roi']:+.1f}%, {trd['win_rate']*100:.0f}% WR")
    elif trd["roi"] is not None and trd["roi"] > -10:
        print(f"  ⚠  Near flat — ROI {trd['roi']:+.1f}%, monitor closely")
    else:
        print(f"  🚨 Attention needed — ROI {trd['roi']:+.1f}%")
    print(f"  {'═' * 55}")
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    snap = build_snapshot()
    prev = load_previous_snapshot()
    print_report(snap, prev)
    save_snapshot(snap)


if __name__ == "__main__":
    main()
