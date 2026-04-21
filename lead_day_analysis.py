"""
Lead-Day Edge Decay Analysis
============================
Slice resolved paper trades by lead_days (0/1/2) and compare:

  * Model-predicted win rate   (mean model_prob across trades in that group)
  * Actual win rate             (observed from outcomes)
  * Calibration gap             (actual − predicted)
  * Realized edge vs market     (actual win rate − mean market_prob)
  * Realized P&L and ROI

Why this matters
----------------
If the probability model's σ is too narrow at longer lead times, our D+2 bins
will look artificially confident and the market will chew through them. A
healthy system has D+0, D+1, D+2 ROIs that are roughly comparable; if D+2
trails by a wide margin we should raise min_ev for that lead or block it
entirely.

Run: python lead_day_analysis.py
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

TRADES_FILE = Path(__file__).parent / "paper_trades.json"


def wilson_ci(wins, n, z=1.96):
    """Wilson 95% CI for a binomial proportion. Honest error bars on small N."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = (z/denom) * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (max(0.0, center - half), min(1.0, center + half))


def analyze(resolved):
    """Group resolved trades by lead_days and compute summary stats."""
    groups = defaultdict(list)
    for t in resolved:
        groups[t["lead_days"]].append(t)

    rows = []
    for lead, trades in sorted(groups.items()):
        n = len(trades)
        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        win_rate = wins / n if n else 0
        lo, hi = wilson_ci(wins, n)

        predicted = sum(t["model_prob"] / 100 for t in trades) / n
        market_avg = sum(t["market_prob"] / 100 for t in trades) / n
        edge_avg = sum(t["edge_pct"] / 100 for t in trades) / n

        total_cost = sum(t["cost"] for t in trades)
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        roi = total_pnl / total_cost * 100 if total_cost > 0 else 0

        calibration_gap = win_rate - predicted
        realized_vs_market = win_rate - market_avg

        rows.append({
            "lead": lead,
            "n": n,
            "wins": wins,
            "win_rate": win_rate,
            "ci_lo": lo,
            "ci_hi": hi,
            "predicted": predicted,
            "market_avg": market_avg,
            "edge_avg": edge_avg,
            "calibration_gap": calibration_gap,
            "realized_vs_market": realized_vs_market,
            "total_cost": total_cost,
            "total_pnl": total_pnl,
            "roi": roi,
        })
    return rows


def print_table(rows):
    if not rows:
        print("No resolved trades to analyze.")
        return

    header = (
        f"{'Lead':>4}  {'N':>4}  {'W':>4}  {'Win%':>7}  {'95%CI':>14}  "
        f"{'Pred%':>7}  {'Mkt%':>7}  {'Edge%':>7}  "
        f"{'CalGap':>7}  {'vs Mkt':>7}  {'Cost':>8}  {'P&L':>8}  {'ROI%':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        ci_str = f"[{r['ci_lo']*100:4.1f},{r['ci_hi']*100:4.1f}]"
        print(
            f"D+{r['lead']:<2d}  {r['n']:>4d}  {r['wins']:>4d}  "
            f"{r['win_rate']*100:>6.1f}%  {ci_str:>14s}  "
            f"{r['predicted']*100:>6.1f}%  {r['market_avg']*100:>6.1f}%  "
            f"{r['edge_avg']*100:>+6.1f}%  "
            f"{r['calibration_gap']*100:>+6.1f}%  "
            f"{r['realized_vs_market']*100:>+6.1f}%  "
            f"${r['total_cost']:>6.2f}  ${r['total_pnl']:>+7.2f}  "
            f"{r['roi']:>+7.1f}%"
        )

    # Column legend (short)
    print()
    print("Legend:")
    print("  Win%    = actual win rate in this lead-day bucket")
    print("  95%CI   = Wilson 95% confidence interval on win rate")
    print("  Pred%   = mean model_prob across trades (what we expected to win)")
    print("  Mkt%    = mean market_prob (what the market priced)")
    print("  Edge%   = mean (Pred − Mkt) — what the scanner saw as edge")
    print("  CalGap  = Win% − Pred% (positive = model underestimated, good)")
    print("  vs Mkt  = Win% − Mkt% (positive = we beat the market)")


def print_diagnosis(rows):
    print()
    print("=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    for r in rows:
        lead = f"D+{r['lead']}"
        n = r["n"]
        if n < 5:
            print(f"  {lead}: only {n} resolved — too thin to conclude. "
                  f"Wait for more data.")
            continue

        ci_width = r["ci_hi"] - r["ci_lo"]
        pred = r["predicted"]
        win = r["win_rate"]

        within_ci = r["ci_lo"] <= pred <= r["ci_hi"]
        pos_roi = r["roi"] > 0

        if within_ci and pos_roi:
            verdict = "✅ Well-calibrated AND profitable"
        elif within_ci and not pos_roi:
            verdict = ("⚠  Calibrated but unprofitable — variance/cost bite, "
                       "not a model problem")
        elif not within_ci and win > pred:
            verdict = ("🎁 Under-confident — model is missing edge it has. "
                       "Could widen bets here")
        else:
            verdict = ("🚨 Over-confident — model too narrow/optimistic. "
                       "Raise min_ev for this lead")

        print(f"  {lead}: {verdict}")
        print(f"         N={n}, Win%={win*100:.1f}%, Pred%={pred*100:.1f}%, "
              f"95%CI={ci_width*100:.1f} pts wide, ROI={r['roi']:+.1f}%")


def compare_leads(rows):
    """If we have 2+ lead buckets, print ROI deltas."""
    if len(rows) < 2:
        return
    print()
    print("=" * 60)
    print("ROI DECAY ACROSS LEAD DAYS")
    print("=" * 60)
    baseline = rows[0]  # D+0
    for r in rows[1:]:
        drop = baseline["roi"] - r["roi"]
        arrow = "↓" if drop > 0 else "↑"
        print(f"  D+{r['lead']} ROI vs D+0: {r['roi']:+.1f}% vs "
              f"{baseline['roi']:+.1f}%  →  {arrow} {abs(drop):.1f} points")
    print()
    print("Rule of thumb: > 10-point ROI drop from D+0 → D+2 suggests the")
    print("model's lead-day σ scaling (base_std × (1+0.3×lead)) is too")
    print("tight. Consider increasing the 0.3 factor or setting a higher")
    print("min_ev for long leads.")


def main():
    if not TRADES_FILE.exists():
        print(f"{TRADES_FILE} not found.")
        sys.exit(1)

    trades = json.loads(TRADES_FILE.read_text())
    resolved = [t for t in trades if t.get("resolved")]

    print(f"\nLead-day edge decay analysis")
    print(f"Loaded {len(trades)} trades ({len(resolved)} resolved)\n")

    rows = analyze(resolved)
    print_table(rows)
    print_diagnosis(rows)
    compare_leads(rows)


if __name__ == "__main__":
    main()
