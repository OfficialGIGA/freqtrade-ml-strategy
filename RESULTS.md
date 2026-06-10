# Dryrun Results — Ultimate Alpha v16

Honest accounting of live dryrun performance. Numbers pulled directly from the Freqtrade SQLite database.

---

## Summary

| Metric | Value |
|--------|-------|
| Period | April 19 - May 21, 2026 |
| Exchange | Kraken (dry-run, no real capital) |
| Pairs | 17 USDT pairs, 4h timeframe |
| Total closed trades | 139 |
| Win rate | 32.4% |
| Total dryrun PnL | -$94.00 |
| Status | Running under updated regime gate |

---

## By regime at entry

This breakdown produced the most important finding from the dryrun.

| Entry tag | Trades | Win rate | Net PnL |
|-----------|--------|----------|---------|
| v16_risk_on | 116 | 23.3% | -$98.84 |
| v16_neutral | 8 | 75.0% | +$6.58 |
| (none / pre-tagging) | 15 | 80.0% | -$1.72 |

The strategy treated risk_on as the most permissive entry condition. The data showed the opposite: risk_on entries lost money at a 77% rate. Neutral entries won at a 75% rate.

---

## By exit reason

| Exit reason | Trades | Wins | Net PnL |
|-------------|--------|------|---------|
| Custom stoploss | 105 | 12 (11%) | -$141.11 |
| ROI target hit | 33 | 33 (100%) | +$47.55 |
| Exit signal | 1 | 0 | -$0.42 |

When the strategy identifies a winning trade, it exits correctly via the ROI table (33/33). The problem is that 76% of trades exit via stoploss with an 11% win rate. The entry signal fires too many bad setups in risk_on conditions.

---

## What changed as a result (June 6, 2026)

**Change 1 — Mirror-symmetric regime gate**

Before:
```python
regime_ok = (regime != "risk_off") | (df["final_score"] > 0.75)
```

After:
```python
if regime == "neutral":
    regime_ok = pd.Series(True, index=df.index)
else:  # risk_on OR risk_off
    regime_ok = df["final_score"] > 0.75
```

Rationale: Only neutral has demonstrated positive expected value in live dryrun. Both risk_on and risk_off now require exceptional score to enter.

**Change 2 — Native exchange stops**

stoploss_on_exchange enabled. Kraken holds the stop-market order server-side so it fires even if the local process goes down.

---

## Honest assessment

The 139-trade dryrun showed that the entry signal in risk_on regime does not have positive expected value on these pairs at 4h timeframe. The neutral regime signal shows promise (75% in 8 trades) but n=8 is too small to conclude anything.

Showing real results including negative ones is more useful than hiding them. The methodology is sound. Whether it produces edge on live data is an open question that requires more accumulation under the updated gate.
