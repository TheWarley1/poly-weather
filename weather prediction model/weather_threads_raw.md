# Raw X/Twitter Thread Extracts — Weather Prediction Trading Models

## Thread 1 — @Vlad_Web3 (Apr 4, 2026)
URL: https://x.com/Vlad_Web3/status/2040515700385849395

> looks like I found a weather insider on Polymarket...
>
> this guy bets only on weather in Chinese cities
> 99% winrate across 300+ bets
> $11,468 this month alone
>
> take a look at his acc → polymarket.com/@speeda
>
> either he's an insider… or he hacked the Chinese weather center
> how is this even possible?

Reply from @PM_TopTraders: "99% on 300 bets is not luck"
Vlad reply: "yeah"

Notes: Paid partnership post. Flags `@speeda` account on Polymarket as a suspected insider/edge trader focused exclusively on Chinese-city weather markets. 99% win rate / 300+ bets / $11,468 monthly PnL.

---

## Thread 2 — @paonx_eth (Apr 4, 2026)
URL: https://x.com/paonx_eth/status/2040515561776624041

> Trader turned $119 into $4,923 on a single weather bet on Polymarket
>
> 'neobrother' made $29,212 on weather markets without even using Claude
>
> His strategy is simple:
> - bet $10-50 on dozens of weather contracts priced at 0.1-3 cents
> - a few hit 1,000-5,000% returns
> - those wins more than cover everything else
>
> His 63% win rate and constantly growing PnL show he knows exactly what he's doing
>
> Here's how I copy it:
> - go to the Ares website: ares.pro/wallets/0x6297b93ea37ff92a57fd636410f3b71ebf74517e
> - fund my wallet
> - click "CopyTrade"
> - set minimum spread for buys and sells (skip this and you'll lose money)
> - click "Start Copying"
> - done

Notes: Core insight — "lottery ticket" strategy on cheap (0.1–3¢) tails of weather markets. Small stakes ($10–50), dozens of positions, tails that hit 10–50x cover the losers. 63% win rate. Copy-trade service promoted (Ares). Trader alias: `neobrother`.

---

## Thread 3 — @0xMovez (Apr 4, 2026)
URL: https://x.com/0xMovez/status/2040485237910548799

> Bot turned $300 → $101K in 2 months on Polymarket weather markets
>
> scanning forecasts across 20 cities via WeatherAPI hunting ultra-rare weather events {0.01¢ - 0.1¢}
>
> I found a fully working, self-calibrating Python bot running this exact strategy
>
> bot results:
> - $25 → $12,452
> - $16 → $8,106
> - $11 → $5,752
>
> bot uses (ECMWF, HRRR, METAR) forecast, calculates EV, sizes positions with Kelly, and self-calibrates.
>
> how to set up:
> - install python 3.10
> - get free weather API from visualcrossing.com
> - set polymarket account and get API in /profile
> - git clone github.com/alteregoeth-ai/weatherbot
> - setup config.json using article below
> - run $1,000 dry test → then go live with real USDC
> - give it 100+ trades to self-calibrate
>
> for more accurate results add paid APIs like OpenWeather, Tomorrow.io, OpenMeteo
>
> bot profile: polymarket.com/profile/0x594edb9112f526fa6a80b8f858a6379c8a2c1c11

Notes: Most technically detailed thread. Key elements:
- **Data sources**: ECMWF, HRRR, METAR as forecast inputs; VisualCrossing (free), OpenWeather, Tomorrow.io, OpenMeteo (paid), WeatherAPI for scanning.
- **Universe**: ~20 cities.
- **Edge hunting**: ultra-rare events priced 0.01–0.1¢ (deep tails).
- **Sizing**: Kelly criterion.
- **Calibration**: self-calibrates after ~100+ trades.
- **Workflow**: dry-run → live USDC.
- ⚠ Promotional — links to GitHub repo `alteregoeth-ai/weatherbot` (same author as thread 6). Treat results claims skeptically.

---

## Thread 4 — @paonx_eth (Apr 3, 2026)
URL: https://x.com/paonx_eth/status/2040123061668569183

> Chinese trader turned $25 into $12,427 just by betting on the weather on Polymarket
>
> Every day he analyzes weather markets with Claude and bets on the most undervalued ones
>
> His profit exceeds $98,850 with a 63% win rate
>
> his strategy is simple:
> - he gives Claude a list of all current weather markets
> - Claude analyzes the most undervalued ones in the 1-5¢ price range and returns them
> - the trader buys them for $20-40
>
> sometimes he also uses a safer strategy: buying at 93-97¢ for $500-3000
>
> in total he has made over 5,000 predictions
>
> His profile: polymarket.com/@coldmath
> Copying his trades: ares.pro/wallets/0x594edb9112f526fa6a80b8f858a6379c8a2c1c11

Notes: Two-pronged strategy:
1. **Cheap-tail hunt**: 1–5¢ longshots at $20–40 each, looking for 10–50x.
2. **Safe favorite fade**: buying 93–97¢ contracts at $500–3000 size (clip 3–7% on near-certain outcomes).
- Uses Claude as analyst — feeds full market list, asks for undervalued picks.
- 5,000+ predictions, 63% WR, $98,850 profit.
- Trader alias: `coldmath`. Note wallet `0x594e...` matches thread 3's "weatherbot" profile wallet — possibly same entity being repeatedly promoted under different aliases.
- ⚠ Copy-trade promo again (Ares).

---

## Thread 5 — @LunarResearcher (Mar 30, 2026)
URL: https://x.com/LunarResearcher/status/2038622884642398503
Title: "I Mass-Analyzed 14,000 Polymarket Wallets With Claude. Here's Guide How to Print Money."

Key stats cited:
- Polymarket $7.94B Feb 2026 volume; weekly volume broke $2.1B in March (ATH).
- 87% of wallets in the red; 14,000+ wallets traded last month.
- Top 20 wallets captured more profit than bottom 13,000 combined.
- "92% of Polymarket traders lose money. The top 0.1% extracted $3.7B."

**Formulas given:**
- EV per $1: `EV = P_true × (1 − P_market) − (1 − P_true) × P_market`. Example: market 40%, you believe 60% → 20¢ edge per $1. Rule: **EV < 5% → SKIP.**
- Kelly: `f* = (p·b − q) / b`, where `b = (1 − P_market) / P_market`. Full Kelly says bet 33% of bankroll → don't. **Use Quarter Kelly.** With $1,000 bankroll → bet $83.
- Bayesian updating for news shocks: `P(H|E) = P(E|H)·P(H) / P(E)`.

**Architecture (Claude-as-analyst pipeline):**
```python
def claude_probability(market_question, market_price):
    client = anthropic.Anthropic(api_key="sk-ant-...")
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role":"user","content": f"""
You are a calibrated prediction market analyst.
Market: {market_question}
Current price: {market_price}
Estimate the TRUE probability (0.00-1.00).
Consider base rates. Penalize extreme confidence.
If you say 70%, ~7 out of 10 such calls should resolve YES.
Return JSON only: {{"probability": 0.XX, "confidence": "high/medium/low"}}
"""}])
    return json.loads(response.content[0].text)
```

Pipeline components:
- `poly_data` → Claude scores wallets
- `insider-tracker` → Claude cross-references with news
- `polyterm` → Claude decides on whale data
- `py-clob-client` → executes
- "50 lines of custom code. Everything else is open source."

**5 Mental Bugs:**
1. Base Rate Neglect — 99% accurate test on 0.1% event → 9% true positive
2. Sunk Cost Fallacy — only question: would you buy at current price with cash?
3. Survivorship Bias — you never see the 13,000 losing wallets behind 1 screenshot
4. Copying Without Filtering — filter by category; copy only domain dominance
5. Overfitting — 3 examples is noise, not signal

**⚠ SECURITY WARNING (critical for you):**
> "In December 2025, a GitHub repo called polymarket-copy-trading-bot contained malware. Professional README. Working code. Real API connections. Hidden inside a dependency: code that read your .env, extracted your private key, and sent it to a remote server. The bot worked. Your money disappeared."
>
> Rules:
> - NEVER use your main wallet. Dedicated wallet, minimal funds
> - Audit every dependency. `pip list`. Google suspicious packages
> - Repo created after Feb 2026 with 500+ stars → likely star-farmed
> - Use Revoke.cash to limit USDC approvals. Never unlimited
> - Start with $100–300. If it works for 2 weeks, scale gradually
> - 664 malicious repos on GitHub...

Self-promo: links to a paid "alpha channel" and kreo.app copytrade — also promotional but the technical content is real.

---

## Thread 6 — @AlterEgo_eth
URL: https://x.com/AlterEgo_eth/status/2034970007369916590
Title: "How to Build a Self-Calibrating Polymarket Weather Bot in Python (Complete Guide)"

The most technically substantive of all threads — a full walkthrough of the bot that threads 3/4 link to.

**Architecture:**
- Part 1 baseline: 6 US cities, NWS forecasts, paper trades on underpriced buckets.
- Part 2 (this): 20 cities across 4 continents, 3 forecast sources, EV + Kelly sizing, full data-storage for self-calibration. 24/7.

**Stack:**
- Python 3.10+, VS Code
- Visual Crossing API (free tier) for **actual resolution temps** after market close

**config.json** (complete):
```json
{
  "balance": 10000.0,
  "max_bet": 20.0,
  "min_ev": 0.05,
  "max_price": 0.45,
  "min_volume": 2000,
  "min_hours": 2.0,
  "max_hours": 72.0,
  "kelly_fraction": 0.25,
  "max_slippage": 0.03,
  "scan_interval": 3600,
  "calibration_min": 30,
  "vc_key": "YOUR_KEY_HERE"
}
```

**Key design decisions:**
- **Station-level coordinates**: Each city points to a specific **ICAO airport station** (e.g. NZWN for Wellington), not city center. Polymarket resolves against Weather Underground, which pulls these exact stations. Using city-center coords can introduce **3–8°F error before the algo even starts**. This is a critical edge-source observation.
- **Unit system per city**: US in °F, everything else °C — bot handles conversion, requests forecasts in native units, all comparisons stay in same scale.
- **Region-routed forecast source**: US cities → HRRR for D+0/D+1. Non-US → ECMWF. Stored in a `region` field in the city dict.
- **City universe includes**: 20 cities including Wellington (NZWN) — Europe, Asia, South America, Oceania. 4 continents.
- **Market storage**: One JSON per market in `data/markets/{city}_{date}.json`, containing full lifecycle — discovery → resolution, incl. `forecast_snapshots[]` (hourly) and `pnl` at close.

**Entry filters (from config defaults):**
- `min_ev = 0.05` (5% EV floor — matches Lunar's thread)
- `max_price = 0.45` (avoid favorites, focus on underpriced)
- `min_volume = 2000` (liquidity filter)
- `min_hours = 2` / `max_hours = 72` (time-to-resolution window)
- `kelly_fraction = 0.25` (Quarter Kelly — matches Lunar)
- `max_slippage = 0.03`
- `max_bet = 20` (hard cap even if Kelly says more)

**Self-calibration mechanism:**
- Bot currently uses `p = 1.0` placeholder (assumes forecast correct).
- Once `calibration_min = 30` resolved markets per city × source exist, `p` is replaced with **actual historical accuracy of each forecast source per city**.
- This is the self-calibration loop: store every forecast snapshot + actual → compute per-source per-city reliability → feed back into EV calculation.

Notes:
- This is genuinely useful engineering content. The code appears real.
- However the wallet(s) promoted in threads 3/4 as "proof it works" link back to this author's ecosystem. The **performance claims** should be treated as marketing.
- ⚠ Direct overlap with the `polymarket-copy-trading-bot` malware pattern that @LunarResearcher warned about. If you clone this repo, treat as hostile: run in an isolated venv, audit every dependency, never use your main wallet.

---

## Thread 7 — @gavelsvtw (Oct 18, 2025)
URL: https://x.com/gavelsvtw/status/1979549975655719355

> Temperature markets are very underrated on Polymarket
>
> Let's look at NYC market on @Polymarket
> 'Highest temperature in NYC on October 18'. Resolution source is Wunderground.
>
> 40¢ for 64-65°F
>
> The alpha here is that Zoom Earth can show future temperature and it's same as Wunderground
>
> The highest temperature for today is 65°F at 3-4 p.m
> So we can actually buy $1 for 40¢ cause we won't see 66°F
>
> Always pay attention about the market info, cause here the location is not NYC, it's LaGuardia Airport Station
>
> Temperature markets is a great niche with money glitch

Notes — concrete manual edge example:
- **Specific concrete alpha**: Zoom Earth's forecast tiles mirror the exact station Wunderground uses. Free tool. Live, no API key needed.
- **Critical gotcha confirmed**: "NYC" market resolves against **LaGuardia Airport station**, not NYC generally. This is the same station-level-resolution point AlterEgo makes.
- Trade example: 64–65°F bucket at 40¢ with forecast peak 65°F → 2.5x edge on intraday.
- Earliest-dated post in set (Oct 2025) and written by an actual user describing a manual trade — the most credible/non-promotional of the set.

---

## Thread 8 — @leshuuuk (Apr 4, 2026)
URL: https://x.com/leshuuuk/status/2040410754810106282

> Polymarket trader made $75K last month trading weather markets
>
> Just 39 trades were enough to make him one of the top weather traders
>
> He doesn't bet every day - only when there's an edge
>
> If market is mispriced, he enters:
> - $594 → $16.5K
> - $3.7K → $8K
> - $18.7K → $64.6K
>
> Whether it's specific degrees or ranges, it doesn't matter to him
>
> No luck involved, just trading what he knows
>
> Profile: polymarket.com/@handsanitizer23
> Tool for copytrading: t.me/PolyGunSniperBot

Notes:
- Paid partnership. Promotes `@handsanitizer23` wallet and a Telegram sniper bot.
- **Contrarian data point vs the "spray the tails" thesis**: this trader ran only **39 trades** for $75K — opposite strategy from `neobrother` (thread 2) who spams dozens of cheap longshots. Implies a **selective, high-conviction** model also works.
- Mixed ticket sizes: $594, $3.7K, $18.7K — scales bet with conviction.

---

## Thread 9 — @KyleDeWriter (Apr 3, 2026)
URL: https://x.com/KyleDeWriter/status/2040132130009743670

> What is greater: Polymarket provides free access to the Pyth Pro version API data
>
> You could have spent $120,000 a year
> But now it costs total $0 if you're a Polymarket trader
>
> There're over 1k articles on how to build your trading bot
> Or you can hire a quant engineer with the funds you just saved

Quoted earlier tweet (Apr 2):
> Polymarket uses Pyth Data for Gold, Silver and ETFs daily markets
> Public API data has 400ms delay, which is still fast
> The ones who may outperform you are those who purchased $10,000 for Pro Version with 1ms delay
> Take Pyth public API, integrate chart and rent a server for a...

Notes — **not weather-specific but directly relevant to your model's data pipeline**:
- Polymarket now offers Pyth Pro API data **free** to Polymarket traders. Normal price ~$120K/yr.
- Public Pyth feed: **400ms latency**. Pro: **1ms**.
- Used by Polymarket for **Gold/Silver/ETF daily markets** (price-resolved). Shows how Polymarket plumbs in external oracle data — useful to understand for **resolution-source arbitrage** (same concept as weather stations: know the exact data source the market resolves against, and you can be faster than the market).

---

## Thread 10 — @Mask4che (Apr 3, 2026)
URL: https://x.com/Mask4che/status/2040123954346500485

> Last night, I prayed to the Weather Gods for that +0.1°C at Seoul airport. They heard me.
>
> From 10c to the promised land. Amen.

Notes:
- Short, attached trade screenshots (not extracted as text).
- **10¢ → 100¢** on a razor-thin +0.1°C margin at a Seoul airport station. Illustrates how tight resolution thresholds create binary coin-flip style payoffs — and why **station-level precision** (AlterEgo, gavelsvtw points) matters so much.
- Second independent confirmation that manual edge exists in temperature markets by reading forecast snapshots close to resolution.

---

