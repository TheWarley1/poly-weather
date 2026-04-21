# Polymarket Weather Trading — Insights Synthesis

**Source:** 10 X/Twitter threads (Oct 2025 – Apr 2026) on weather prediction trading.
**Focus:** data sources, modeling approaches, edge strategies, results. Raw extracts in `weather_threads_raw.md`.

---

## 1. Credibility Filter (read this first)

Of the 10 threads, the credibility profile is uneven. It matters for how much weight to give each claim:

| # | Author | Type | Credibility |
|---|---|---|---|
| 1 | @Vlad_Web3 | Paid partnership, points at `@speeda` 99% WR wallet | Low — screenshot marketing |
| 2 | @paonx_eth | Paid partnership, promotes `neobrother` + ares.pro copytrade | Low — promo |
| 3 | @0xMovez | Promotes `alteregoeth-ai/weatherbot` GitHub + wallet `0x594e...` | Low — promo, matches malware pattern |
| 4 | @paonx_eth | Promotes `coldmath` wallet (= same `0x594e...` as thread 3) + ares.pro | Low — same ring as #3 |
| 5 | @LunarResearcher | Long technical article with real formulas; also sells alpha channel | **Medium–High — real content** |
| 6 | @AlterEgo_eth | Full working bot walkthrough with config + code | **Medium–High — real engineering**, but same author ecosystem as #3 |
| 7 | @gavelsvtw | Individual user describing a specific manual trade (Oct 2025) | **High — non-promotional** |
| 8 | @leshuuuk | Paid partnership, promotes `handsanitizer23` + Telegram sniper bot | Low — promo |
| 9 | @KyleDeWriter | Factual claim about Pyth free API access | Medium — verifiable |
| 10 | @Mask4che | Single trade screenshot celebration | Medium — real trade |

⚠ **Scam pattern to note.** Threads 2/3/4/8 follow the same template: "X turned $25 into $12K → here's the copytrade link." Wallet `0x594edb9112...` is promoted by BOTH thread 3 (as "neobrother's bot") and thread 4 (as "coldmath"). Thread 5 (Lunar) explicitly warns that the Dec 2025 `polymarket-copy-trading-bot` repo **stole private keys via a hidden dependency**. Assume any repo linked in these threads is hostile until proven otherwise. Do **not** run `alteregoeth-ai/weatherbot` against a funded main wallet — if you want to study it, clone into an isolated VM with no keys.

That said, the **underlying ideas** (forecast-source stacking, EV + Kelly, self-calibration, station-level precision) are legitimate and align with how real prediction-market quants work. The rest of this document takes the signal and drops the marketing.

---

## 2. What the market looks like (Lunar's numbers)

- Polymarket Feb 2026 volume: **$7.94B**. Weekly ATH **$2.1B** in March 2026.
- **87% of wallets are in the red.** 14,000+ wallets traded in March alone.
- Top 20 wallets captured more profit than the bottom 13,000 combined.
- "92% of Polymarket traders lose money. The top 0.1% extracted $3.7B."

**Implication for your model:** this is a power-law market. Your goal isn't to win most trades — it's to be in the right 8% of the distribution. That requires *measurable* edge, not gut feel.

---

## 3. Data sources actually used

### Forecast sources (weather inputs)
| Source | Type | Best for | Cost |
|---|---|---|---|
| **HRRR** | High-Resolution Rapid Refresh (NOAA) | US cities, D+0 / D+1 short-range | Free |
| **ECMWF** | European Centre medium-range model | Non-US cities, D+1 onward | Free (tier) / paid |
| **METAR** | Station-level current observations | Nowcasting, last-hour pinning | Free |
| **NWS** | US National Weather Service public API | Baseline US forecasts | Free |
| **Visual Crossing** | Historical + forecast, easy JSON | Resolution look-up after close | Free tier |
| **WeatherAPI** | Aggregated feeds | Multi-city scanning | Free / paid |
| **OpenWeather / Tomorrow.io / OpenMeteo** | Commercial-grade forecasts | Ensemble/redundancy | Paid |
| **Zoom Earth** (gavelsvtw) | Free web viewer; same grid as Wunderground | Quick manual check before entry | Free, no API |

### Resolution sources (what the market resolves against)
- **Weather Underground** — Polymarket's primary resolver for US temp markets.
- Resolves against a **specific ICAO airport station**, e.g. NYC = **LaGuardia (KLGA)**, Wellington = **NZWN**, Seoul = **airport station (not city)**.
- Using city-center coordinates instead of station coordinates introduces **3–8°F error before the algorithm even runs** (AlterEgo). This is the single most actionable alpha in the entire set.

### Market / execution data
- **py-clob-client** — Python client for Polymarket's CLOB.
- **Pyth public API** — Polymarket's oracle for price-resolved markets (Gold/Silver/ETF). Free for Polymarket users, 400ms public latency vs 1ms Pro ($120K/yr list). Not weather but worth knowing for cross-market plays.
- **poly_data**, **insider-tracker**, **polyterm** — open-source analytics pipes Lunar names.

---

## 4. Core modeling approach (consensus across threads)

```
┌──────────────────────────────────────────────────────────────┐
│  1. SCAN     : pull all active weather markets (universe     │
│                ≈ 20 cities, 4 continents)                    │
│                                                              │
│  2. FORECAST : pull forecasts from 2–3 sources per city,     │
│                routed by region (US → HRRR, rest → ECMWF)    │
│                                                              │
│  3. ESTIMATE : compute P_true for each bucket/contract       │
│                using an ensemble (equal or reliability-      │
│                weighted average) + historical calibration    │
│                                                              │
│  4. EV       : EV = P_true·(1 − P_mkt) − (1 − P_true)·P_mkt  │
│                SKIP if EV < 5%                               │
│                                                              │
│  5. SIZE     : Quarter Kelly                                 │
│                f* = 0.25 · (p·b − q) / b                     │
│                capped at max_bet ($20) and max_price (0.45)  │
│                                                              │
│  6. EXECUTE  : py-clob-client, slippage cap 3%               │
│                                                              │
│  7. STORE    : one JSON per market with every forecast       │
│                snapshot + resolution + PnL                   │
│                                                              │
│  8. CALIBRATE: after 30+ resolutions per city×source,        │
│                replace p=1.0 placeholder with historical     │
│                reliability and loop                          │
└──────────────────────────────────────────────────────────────┘
```

**Default config values that work (AlterEgo):**
- `min_ev = 0.05` (5% — matches Lunar's "skip below 5%" rule)
- `max_price = 0.45` — never pay favorites, only underpriced outcomes
- `min_volume = 2000` — liquidity filter so you can actually get out
- `min_hours = 2`, `max_hours = 72` — time-to-resolution sweet spot
- `kelly_fraction = 0.25` — Quarter Kelly
- `max_slippage = 0.03`
- `max_bet = 20` — hard cap even if Kelly says more
- `calibration_min = 30` — resolutions before trusting a (city, source) pair
- `scan_interval = 3600` — hourly rescan

---

## 5. Edge / strategy variants observed

Three distinct playbooks emerge. They're **not mutually exclusive** — you could run them in parallel.

### A. Cheap-tail scatter ("neobrother" — thread 2)
- $10–50 tickets on contracts priced **0.1–3¢**.
- Dozens of positions; most go to zero; a handful hit **10–50x**.
- 63% hit rate cited, but that's meaningless without the payoff distribution. The real math is lottery-ticket: E[payoff] is carried by the few tail hits.
- **Risk:** requires strict sizing discipline; easy to drift and bleed if the hit-rate on tails drops.

### B. Safe-favorite fade ("coldmath" second strategy — thread 4)
- $500–3000 tickets on contracts at **93–97¢**.
- Clip 3–7% on near-certain outcomes where the market hasn't fully converged.
- Only works if your forecast is genuinely more accurate than the market on the endpoint.
- **Risk:** blow-up risk on the rare adverse resolution — one 0 can wipe 20+ winners.

### C. High-conviction selective ("handsanitizer23" — thread 8)
- **39 trades = $75K** in a month. Opposite of A.
- Only enters when mispricing is large and defensible.
- Mixed ticket sizes scaling with conviction ($594 / $3.7K / $18.7K examples).
- **Risk:** hardest to automate — requires a reliable "don't trade" signal.

### D. Manual visual read (gavelsvtw / Mask4che — threads 7, 10)
- Use **Zoom Earth** or live radar to see the forecast tile for the exact station.
- Enter when the forecast peak is safely inside (or outside) a bucket edge and the market hasn't priced it.
- Mask4che's Seoul trade: 10¢ → 100¢ on a **+0.1°C** margin at the airport station.
- This is real, manual, scalable only up to your personal bandwidth — but it's the cleanest alpha source and the cheapest to validate. **Start here** to prove the edge exists before you code anything.

---

## 6. Results claims (normalized)

| Trader/Bot | Win rate | Key PnL claim | Strategy | Trust |
|---|---|---|---|---|
| `@speeda` (Chinese cities) | 99% / 300 bets | $11,468/mo | Unknown | "insider or hacker" per Vlad — likely an outlier/insider, not reproducible |
| `neobrother` | 63% | $29,212 lifetime | A (cheap-tail scatter) | Promo source |
| Bot at wallet `0x594e...` | n/a | $300 → $101K in 2 months | Full bot w/ Kelly + ECMWF/HRRR | Promo — same ecosystem as `alteregoeth-ai/weatherbot` |
| `coldmath` | 63% / 5,000+ preds | $98,850 lifetime | A + B mix | Same `0x594e...` wallet as above |
| `handsanitizer23` | n/a | $75K in 39 trades / month | C (selective) | Promo |

**Honest read:** PnL screenshots are unverifiable and often same-ecosystem. Use these as **idea sources**, not performance targets. Lunar's 87%-lose stat is the realistic baseline.

---

## 7. Five mental bugs to avoid (Lunar, verbatim-adjacent)

1. **Base rate neglect.** A 99%-accurate test on a 0.1% event gives a 9% true-positive rate. "Looks likely" ≠ "is likely."
2. **Sunk cost.** You bought at 70¢, it's 40¢, new info says NO. Only question: would you buy at 40¢ right now with fresh cash?
3. **Survivorship bias.** Every +$50K screenshot hides 13,000 losing wallets.
4. **Copying without filtering.** A wallet 91% WR on crypto, 15% on politics — copy only their dominant category.
5. **Overfitting.** 3 historical examples is noise, not a pattern.

---

## 8. Security — do not skip

From Lunar (and the pattern in this very set of threads):

- **Never use your main wallet.** Dedicated wallet, minimal funds only.
- **Audit every dependency** of anything you clone. `pip list`, google each package.
- **Suspect repos** created after Feb 2026 with 500+ stars → star-farmed.
- **Revoke.cash** limits on USDC approvals. Never grant unlimited.
- **Start with $100–300** and only scale after 2 weeks of green.
- **664 malicious Polymarket repos** on GitHub per Lunar. The December 2025 `polymarket-copy-trading-bot` repo stole private keys via a hidden dependency.

Specifically: **do not run `alteregoeth-ai/weatherbot` on a box that has any wallet key environment variable you care about.** If you want its logic, read the code, don't execute it. Port the pieces you want into your own repo.

---

## 9. Actionable takeaways for your own model

Ranked by impact-per-effort:

1. **Map the exact resolution stations** for every weather market Polymarket lists. Maintain a `(market_slug → ICAO_station, unit, timezone)` table. This is a one-time effort that unlocks everything downstream. Nothing else matters if your forecast is for the wrong spot.
2. **Run an ensemble of ≥2 free forecast sources** (HRRR for US, ECMWF or OpenMeteo elsewhere, plus Visual Crossing for resolution truth). Ensemble disagreement is itself a feature — wide spread = skip.
3. **Enforce the EV ≥ 5% + Quarter Kelly rule** with a hard `max_bet` and `max_price` cap. These defaults are consistent across Lunar + AlterEgo and align with the standard prediction-market quant playbook.
4. **Log every forecast snapshot** (hourly) alongside eventual resolution. You need ≥30 resolved markets per (city, source) before you can trust calibration weights. Start collecting this *before* you trade so the bot isn't flying blind on day one.
5. **Prove the manual edge first.** Before you build the whole pipeline, do 10–20 manual trades off Zoom Earth + the ICAO station map the way gavelsvtw and Mask4che describe. If you can't make it work manually on 10 trades, automation won't save you.
6. **Treat time-to-resolution as a feature, not a filter.** The 2–72 hour window is where mispricings live; <2h is efficient, >72h is too much forecast drift.
7. **Decide on one playbook to automate first.** Playbook D (manual visual read) → Playbook A (cheap-tail scatter) is the easiest on-ramp because entries are cheap and the worst case per ticket is bounded. Playbook B (fade 93–97¢ favorites) is higher absolute PnL but has blow-up risk you shouldn't take until calibration is proven.
8. **Monitor Pyth's free feed** (KyleDeWriter) for price-resolved markets as a separate product line. Same underlying principle (know the oracle, beat the latency) applies to Gold/Silver/ETF dailies.

---

## 10. Open questions worth investigating

- What's the actual historical accuracy of HRRR vs ECMWF vs OpenMeteo at the specific ICAO stations Polymarket uses? (Calibration data — can be pre-built from public archives without placing a single trade.)
- How tight are the bucket edges vs the native forecast uncertainty? If buckets are 1°F wide and HRRR's RMSE at that station is 1.8°F, the market is structurally a coin flip — no amount of modeling fixes that.
- Which cities have the **thinnest liquidity** (biggest mispricings) vs **deepest liquidity** (reliable exits)? The answer is probably "trade mispricings in thin cities, don't go over 30% of ADV."
- What's the resolution-source latency gap? If Weather Underground publishes the resolving observation 5–30 min before Polymarket resolves, that's a risk-free arbitrage window per Kyle's Pyth point.

---

*Generated from 10 X threads. See `weather_threads_raw.md` for raw per-thread extracts.*
