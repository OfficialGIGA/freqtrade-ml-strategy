# Methodology — Ultimate Alpha v16

## What this is

A Freqtrade strategy for Kraken spot trading. It runs a LightGBM classifier offline-trained on historical OHLCV data, then uses the model output probability as one input into a scoring pipeline that gates entries by market regime.

This document describes how it works, why design decisions were made, and what was explicitly removed from earlier versions.

---

## Architecture

### 1. Feature engineering (`features.py`)

77 features computed from OHLCV data: momentum indicators (RSI, MACD, EMA crossovers), volatility measures (ATR, Bollinger width), volume signals, and a BTC-dominance macro feature pulled via 1d informative merge. The same module runs at training time and live — no feature drift between backtest and production.

### 2. Model

One LightGBM binary classifier. Input: 77 features at signal time. Output: P(trade wins), where "wins" is defined as closing above the entry price within the ROI table window. Trained offline on Kraken 4h OHLCV data. No retraining during live operation — the model is static until a deliberate retrain cycle.

No ensemble. Earlier versions used a 6-model torch ensemble that produced random noise on the available data volume. It was removed.

### 3. Regime classifier

BTC-driven three-state regime: `risk_on`, `neutral`, `risk_off`. Uses BTC price relative to rolling EMAs and volatility percentile. Hysteresis prevents rapid regime flipping on noise. The regime gates which signals are allowed to become entries.

### 4. Scoring pipeline

Final score is an additive fusion:
Bounded [0, 1]. No multiplicative terms — earlier multiplicative designs caused score explosion on correlated conditions.

### 5. Entry logic

Three conditions must all pass:

- `score > threshold` (regime-dependent)
- Trend gates: price above EMA-200, EMA-21 above EMA-50, MACD bullish, RSI 50-75
- Execution-cost gate: expected profit must exceed estimated taker fee x 2

**Regime gate (updated June 2026 based on live dryrun data):**

After 139 dryrun trades revealed that risk_on entries had a 23% win rate and -$98.84 net PnL, the gate was updated to require score > 0.75 in BOTH risk_on AND risk_off. Only neutral regime entries trade freely at the standard threshold. This is a data-driven change, not a theoretical one.

### 6. Exit logic

- ROI table: tiered profit targets at 1%, 2%, 3%, 5%
- Custom stoploss: ATR-scaled hard stop, -2% baseline widening to -5% on high-volatility pairs. Does NOT trail — earlier trailing logic was removed after it was found to exit winners prematurely.
- Score-collapse exit: if model score drops significantly while in a position, exit early regardless of price

### 7. Position sizing

Kelly fraction scaled by regime confidence and portfolio heat (correlation-aware). Maximum 10 concurrent positions. Stake per trade shrinks as open positions increase.

---

## What was explicitly removed from v14

The previous version accumulated features over time without removing ones that did not work. v16 was a complete rewrite.

| Component | Why removed |
|-----------|-------------|
| 6-model torch ensemble | Untrained on available data volume — produced random noise |
| Genetic weight optimizer | Noise amplifier on small samples |
| "True RL policy update" | Was gradient descent on a single sample, not RL |
| Market memory (hash-state keys) | Keys never collided usefully in production |
| Champion/challenger switching | Never actually triggered a switch |
| 5 on-chain/DeFi/tokenomics scores | Fabricated fallback data when API unavailable |
| Quantum/elite naming conventions | Renamed to reflect actual implementation |

---

## Infrastructure

- Exchange: Kraken (spot, USDT pairs)
- Timeframe: 4h candles
- Pairs: 17 USDT pairs (BTC, ETH, SOL, XRP, DOGE, ADA, AVAX, LINK, DOT, LTC, XMR, BCH, SHIB, TON, ATOM, ALGO, KAS)
- Mode: Freqtrade dry-run (paper trading against live Kraken data)
- Native exchange stops: enabled via stoploss_on_exchange in order_types config
