# Changelog — Ultimate Alpha v16

All notable changes to this strategy are documented here.
Format: what changed, why it changed, what data drove the decision.

---

## [v16.0] — April 2026

Complete rewrite. Previous version (v14 / UltimateAlphaBotPro) had accumulated
components over time without removing ones that did not work. v16 starts clean.

### Added
- Single LightGBM classifier replacing the 6-model torch ensemble
- Shared `features.py` module used at both training and live time
- BTC-driven regime classifier with hysteresis (risk_on / neutral / risk_off)
- Additive scoring pipeline bounded [0, 1]
- ATR-scaled custom stoploss (hard stop, no trailing)
- Kelly fraction position sizing scaled by regime and portfolio heat
- CryptoQuant netflow signal (reads from CRYPTOQUANT_TOKEN env var)
- Proper merge_asof for 1d informative timeframe

### Removed
- 6-model torch ensemble — untrained on available data volume, produced random noise
- Genetic weight optimizer — noise amplifier on small sample sizes
- "True RL policy update" — was single-sample gradient descent, not RL
- Market memory with hash-state keys — keys never collided usefully in production
- Champion/challenger model switching — trigger condition never fired
- 5 fake on-chain/DeFi/tokenomics/whale/microstructure scores — fabricated
  fallback data when API was unavailable
- Multiplicative scoring — caused score explosion on correlated signal conditions
- Trailing stop logic — was exiting winners prematurely (now ATR hard stop only)
- "Quantum elite 20/10 master" naming — renamed to reflect actual implementation

---

## [v16.0 — Regime Gate Update] — June 6, 2026

### Changed
- Regime gate updated based on 139-trade dryrun analysis

**Before:**
```python
regime_ok = (regime != "risk_off") | (df["final_score"] > 0.75)
```

**After:**
```python
if regime == "neutral":
    regime_ok = pd.Series(True, index=df.index)
else:  # risk_on OR risk_off
    regime_ok = df["final_score"] > 0.75
```

**Why:** Dryrun data showed risk_on entries had 23% win rate across 116 trades
(-$98.84 net). Neutral entries had 75% win rate across 8 trades (+$6.58).
The strategy was treating risk_on as the most permissive entry condition.
The data showed the opposite. Gate now mirrors risk_off treatment for risk_on:
both require score > 0.75. Only neutral trades freely at standard threshold.

This is a data-driven change. The gate was not changed because of theory —
it was changed because 139 trades of live dryrun showed a specific failure mode.

- Native Kraken stop-loss enabled (stoploss_on_exchange: true in order_types)
  Stops now held server-side on Kraken, survive local process restarts.

---

## [Pending — next review]

Accumulating dryrun data under the updated regime gate.
Next review when 50+ trades have closed under new gate logic.
Decision: whether neutral-regime signal has sufficient edge to continue,
or whether the underlying signal needs to be retrained on new data.
