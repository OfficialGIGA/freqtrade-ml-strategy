# Architecture

The signal-to-trade pipeline, end to end.

```mermaid
flowchart TD
    A[Kraken OHLCV<br/>4h candles] --> B[features.py<br/>77 features]
    B --> C[LightGBM classifier<br/>P trade wins]
    A --> D[BTC regime detector<br/>risk_on / neutral / risk_off]
    C --> E[Scoring pipeline<br/>additive fusion, bounded 0-1]
    D --> E
    E --> F{Entry gates}
    F -->|score > threshold| G[Trend gates<br/>EMA200, MACD, RSI]
    F -->|score too low| X[No entry]
    G -->|all pass| H[Execution-cost gate<br/>profit > fee x 2]
    G -->|fail| X
    H -->|pass| I{Regime gate}
    H -->|fail| X
    I -->|neutral| J[ENTER]
    I -->|risk_on or risk_off<br/>requires score > 0.75| K{score > 0.75?}
    K -->|yes| J
    K -->|no| X
    J --> L[Position sizing<br/>Kelly x regime x heat]
    L --> M[Open position]
    M --> N{Exit conditions}
    N -->|ROI target| O[Exit: profit]
    N -->|ATR stoploss| P[Exit: loss]
    N -->|score collapse| Q[Exit: signal gone]

    style J fill:#2ecc71,color:#fff
    style X fill:#e74c3c,color:#fff
    style I fill:#f39c12,color:#fff
```

## Reading the diagram

The **regime gate** (orange) is the critical control point. It was updated in
June 2026 after dryrun data showed `risk_on` entries had a 23% win rate. Now
only the `neutral` regime trades freely; both `risk_on` and `risk_off` require
an exceptional score (> 0.75) to enter.

The pipeline is deliberately a series of **gates that reject**, not a single
score that triggers. A signal must survive every gate to become a trade. This
is what keeps trade frequency low and quality high — most candidates are
rejected, by design.

See [METHODOLOGY.md](METHODOLOGY.md) for component details and
[RESULTS.md](RESULTS.md) for what the live dryrun showed.
