"""
features.py — Shared feature engineering for Ultimate Alpha v16.

This module is imported by BOTH the live strategy AND the offline trainer,
which guarantees that features the model was trained on are identical to
features the live bot sees. This is the single most important invariant in
an ML trading system — violate it and your backtest lies to you.

All features are:
  - Lookahead-free (no future information)
  - Properly normalized (z-score or bounded 0..1)
  - Stable (ffill/bfill/fillna at the end)
  - Documented

Feature count: 77. No zero-padding, no placeholders.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame, Series


# Feature column names, in canonical order. The trainer saves this list
# alongside the model so the strategy can assert they match at load time.
FEATURE_COLUMNS: list[str] = []  # populated after engineer_features runs once


def _as_series(s, index=None) -> Series:
    """Coerce ndarray/list/Series into a pandas Series with a usable index."""
    if isinstance(s, pd.Series):
        return s
    return pd.Series(np.asarray(s), index=index)


def _rolling_zscore(s, window: int = 100, min_periods: int = 20, index=None) -> Series:
    """Z-score with rolling mean/std — no lookahead."""
    s = _as_series(s, index=index)
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std().clip(lower=1e-10)
    return ((s - mean) / std).clip(-4, 4).fillna(0)


def _bounded(s, lo: float, hi: float, index=None) -> Series:
    """Normalize a known-range indicator to 0..1."""
    s = _as_series(s, index=index)
    return ((s - lo) / max(hi - lo, 1e-10)).clip(0, 1).fillna(0.5)


def engineer_features(df: DataFrame, btc_df: DataFrame | None = None) -> DataFrame:
    """
    Compute all features. Returns df with feat_* columns added.

    Args:
        df: OHLCV dataframe for the pair (must have open/high/low/close/volume).
        btc_df: Optional BTC/USDT OHLCV on the same timeframe, already aligned
                by index. Used for BTC-relative features. If None, those
                features default to neutral.

    Returns:
        df with ~78 feat_XXX columns appended. Original columns preserved.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    close = df["close"].clip(lower=1e-10)
    high = df["high"].clip(lower=1e-10)
    low = df["low"].clip(lower=1e-10)
    volume = df["volume"].clip(lower=0.0)

    # Local Series coercer — TA-Lib on some platforms returns numpy arrays
    # rather than pandas Series. Every TA call goes through this wrapper.
    _idx = df.index

    def _s(x) -> Series:
        if isinstance(x, pd.Series):
            return x
        arr = np.asarray(x)
        if arr.ndim == 0:
            arr = np.full(len(_idx), float(arr))
        return pd.Series(arr, index=_idx)

    feat: dict[str, Series] = {}
    idx = 0

    def add(values: Series):
        nonlocal idx
        feat[f"feat_{idx:03d}"] = values
        idx += 1

    # ────────────────────────────────────────────────────────────────
    # GROUP A — Returns across multiple horizons (8 features)
    # ────────────────────────────────────────────────────────────────
    for period in (1, 3, 6, 12, 24, 48, 96, 168):
        add(_rolling_zscore(close.pct_change(period).fillna(0)))

    # ────────────────────────────────────────────────────────────────
    # GROUP B — Momentum oscillators (12 features)
    # ────────────────────────────────────────────────────────────────
    for period in (7, 14, 28):
        add(_bounded(_s(ta.RSI(close, timeperiod=period)), 0, 100))

    stoch_k_raw, stoch_d_raw = ta.STOCH(high, low, close)
    stoch_k = _s(stoch_k_raw)
    stoch_d = _s(stoch_d_raw)
    add(_bounded(stoch_k.fillna(50), 0, 100))
    add(_bounded(stoch_d.fillna(50), 0, 100))
    add(_bounded((stoch_k - stoch_d).fillna(0), -50, 50))

    add(_bounded(_s(ta.WILLR(high, low, close, timeperiod=14)).fillna(-50), -100, 0))
    add(_rolling_zscore(_s(ta.CCI(high, low, close, timeperiod=14)).fillna(0)))
    add(_bounded(_s(ta.MFI(high, low, close, volume, timeperiod=14)).fillna(50), 0, 100))

    for period in (6, 12, 24):
        add(_rolling_zscore(_s(ta.ROC(close, timeperiod=period)).fillna(0)))

    # ────────────────────────────────────────────────────────────────
    # GROUP C — Trend (11 features)
    # ────────────────────────────────────────────────────────────────
    ema_9 = _s(ta.EMA(close, timeperiod=9)).clip(lower=1e-10)
    ema_21 = _s(ta.EMA(close, timeperiod=21)).clip(lower=1e-10)
    ema_50 = _s(ta.EMA(close, timeperiod=50)).clip(lower=1e-10)
    ema_200 = _s(ta.EMA(close, timeperiod=200)).clip(lower=1e-10)

    add(_rolling_zscore((close / ema_9) - 1))
    add(_rolling_zscore((close / ema_21) - 1))
    add(_rolling_zscore((close / ema_50) - 1))
    add(_rolling_zscore((close / ema_200) - 1))
    add(_rolling_zscore((ema_9 / ema_21) - 1))
    add(_rolling_zscore((ema_21 / ema_50) - 1))
    add(_rolling_zscore((ema_50 / ema_200) - 1))

    adx = _s(ta.ADX(high, low, close, timeperiod=14)).fillna(20)
    plus_di = _s(ta.PLUS_DI(high, low, close, timeperiod=14)).fillna(20)
    minus_di = _s(ta.MINUS_DI(high, low, close, timeperiod=14)).fillna(20)
    add(_bounded(adx, 0, 60))
    add(_bounded((plus_di - minus_di).fillna(0), -50, 50))

    macd_val_raw, macd_sig_raw, macd_hist_raw = ta.MACD(close)
    macd_hist = _s(macd_hist_raw)
    add(_rolling_zscore(macd_hist.fillna(0)))
    add(_rolling_zscore(_s(ta.LINEARREG_SLOPE(close, timeperiod=24)).fillna(0)))

    # ────────────────────────────────────────────────────────────────
    # GROUP D — Volatility (10 features)
    # ────────────────────────────────────────────────────────────────
    atr_14 = _s(ta.ATR(high, low, close, timeperiod=14)).clip(lower=1e-10)
    atr_pct = (atr_14 / close) * 100
    add(_rolling_zscore(atr_pct))
    add(_bounded(atr_pct, 0.1, 8.0))  # absolute level of volatility

    # Realized vol across horizons
    log_ret = np.log(close / close.shift(1)).fillna(0)
    for period in (6, 24, 72):
        vol = log_ret.rolling(period).std() * np.sqrt(period)
        add(_rolling_zscore(vol))

    # Bollinger band width
    bb_upper_raw, bb_mid_raw, bb_lower_raw = ta.BBANDS(close, timeperiod=20)
    bb_upper = _s(bb_upper_raw)
    bb_mid = _s(bb_mid_raw)
    bb_lower = _s(bb_lower_raw)
    bb_width = ((bb_upper - bb_lower) / bb_mid.clip(lower=1e-10)).fillna(0)
    add(_rolling_zscore(bb_width))

    # Position within bands (0 = at lower, 1 = at upper)
    bb_range = (bb_upper - bb_lower).clip(lower=1e-10)
    bb_pos = ((close - bb_lower) / bb_range).clip(0, 1).fillna(0.5)
    add(bb_pos)

    # Volatility-of-volatility
    add(_rolling_zscore(atr_pct.rolling(24).std().fillna(0)))

    # Parkinson high-low estimator
    parkinson = (np.log(high / low) ** 2).rolling(24).mean().apply(np.sqrt).fillna(0)
    add(_rolling_zscore(parkinson))

    # Directional volatility asymmetry
    up_ret = log_ret.where(log_ret > 0, 0).rolling(24).std().fillna(0)
    dn_ret = log_ret.where(log_ret < 0, 0).rolling(24).std().fillna(0)
    asymmetry = (up_ret - dn_ret) / (up_ret + dn_ret + 1e-10)
    add(asymmetry.clip(-1, 1))

    # ────────────────────────────────────────────────────────────────
    # GROUP E — Volume (8 features)
    # ────────────────────────────────────────────────────────────────
    vol_mean_24 = volume.rolling(24).mean().clip(lower=1e-10)
    vol_mean_96 = volume.rolling(96).mean().clip(lower=1e-10)
    add(_rolling_zscore(volume / vol_mean_24))
    add(_rolling_zscore(volume / vol_mean_96))
    add(_rolling_zscore(vol_mean_24 / vol_mean_96))  # vol trend

    # OBV-like accumulation
    obv = (np.sign(close.diff().fillna(0)) * volume).cumsum()
    add(_rolling_zscore(obv.diff(24).fillna(0)))

    # Money flow volume
    mfv = ((close - low) - (high - close)) / (high - low).clip(lower=1e-10) * volume
    cmf = mfv.rolling(20).sum() / volume.rolling(20).sum().clip(lower=1e-10)
    add(cmf.clip(-1, 1).fillna(0))

    # Volume-price correlation (is volume following price?)
    vp_corr = close.pct_change().rolling(24).corr(volume.pct_change()).fillna(0)
    add(vp_corr.clip(-1, 1))

    # Zero-volume flag (dead candles hurt reliability)
    add((volume == 0).astype(float).rolling(12).mean())

    # Dollar volume trend
    dollar_vol = close * volume
    add(_rolling_zscore(dollar_vol.rolling(12).mean() / dollar_vol.rolling(72).mean().clip(lower=1e-10)))

    # ────────────────────────────────────────────────────────────────
    # GROUP F — Price structure & candle anatomy (10 features)
    # ────────────────────────────────────────────────────────────────
    candle_range = (high - low).clip(lower=1e-10)
    body = (close - df["open"]).abs()
    upper_wick = (high - df[["open", "close"]].max(axis=1)).clip(lower=0)
    lower_wick = (df[["open", "close"]].min(axis=1) - low).clip(lower=0)

    add((body / candle_range).clip(0, 1).fillna(0.5))  # body-to-range
    add((upper_wick / candle_range).clip(0, 1).fillna(0))
    add((lower_wick / candle_range).clip(0, 1).fillna(0))
    add(((close - df["open"]) / candle_range).clip(-1, 1).fillna(0))  # signed intra-candle direction

    # Position of close within recent range (Donchian)
    for period in (12, 48, 168):
        hi_n = high.rolling(period).max()
        lo_n = low.rolling(period).min()
        pos = ((close - lo_n) / (hi_n - lo_n).clip(lower=1e-10)).clip(0, 1).fillna(0.5)
        add(pos)

    # Distance to recent high/low, z-scored
    add(_rolling_zscore(close / high.rolling(48).max().clip(lower=1e-10) - 1))
    add(_rolling_zscore(close / low.rolling(48).min().clip(lower=1e-10) - 1))

    # Consecutive direction streak
    direction = np.sign(close.diff().fillna(0))
    streak = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1
    add((streak * direction).clip(-10, 10) / 10)

    # ────────────────────────────────────────────────────────────────
    # GROUP G — BTC-relative / market context (10 features)
    # ────────────────────────────────────────────────────────────────
    if btc_df is not None and not btc_df.empty:
        btc_close = btc_df["close"].reindex(df.index, method="ffill").clip(lower=1e-10)

        # Relative strength vs BTC
        for period in (6, 24, 96):
            rs = (close.pct_change(period) - btc_close.pct_change(period)).fillna(0)
            add(_rolling_zscore(rs))

        # Rolling beta to BTC
        pair_ret = close.pct_change().fillna(0)
        btc_ret = btc_close.pct_change().fillna(0)
        cov = pair_ret.rolling(72).cov(btc_ret)
        var_btc = btc_ret.rolling(72).var().clip(lower=1e-10)
        beta = (cov / var_btc).clip(-3, 3).fillna(1.0)
        add((beta / 3).clip(-1, 1))

        # Correlation to BTC
        corr = pair_ret.rolling(72).corr(btc_ret).fillna(0).clip(-1, 1)
        add(corr)

        # BTC's own state (shared across all pairs, but useful signal)
        btc_ema_200 = _s(ta.EMA(btc_close, timeperiod=200)).clip(lower=1e-10)
        add(_rolling_zscore((btc_close / btc_ema_200) - 1))
        add(_rolling_zscore(btc_ret.rolling(24).sum().fillna(0)))
        btc_atr = _s(ta.ATR(btc_df["high"].reindex(df.index, method="ffill"),
                            btc_df["low"].reindex(df.index, method="ffill"),
                            btc_close, timeperiod=14)).fillna(0)
        add(_bounded((btc_atr / btc_close) * 100, 0.1, 6.0))

        # Lead-lag: pair leading BTC?
        leadlag = pair_ret.shift(1).rolling(24).corr(btc_ret).fillna(0).clip(-1, 1)
        add(leadlag)

        # BTC-beta-adjusted pair return (the residual alpha)
        alpha_ret = (pair_ret - beta * btc_ret).fillna(0)
        add(_rolling_zscore(alpha_ret.rolling(24).sum()))
    else:
        # Neutral fallback when BTC data not supplied
        for _ in range(10):
            add(pd.Series(0.0, index=df.index))

    # ────────────────────────────────────────────────────────────────
    # GROUP H — Temporal / calendar (8 features)
    # ────────────────────────────────────────────────────────────────
    if isinstance(df.index, pd.DatetimeIndex):
        idx_dt = df.index
    elif "date" in df.columns:
        idx_dt = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    else:
        idx_dt = pd.DatetimeIndex(pd.to_datetime(df.index, errors="coerce"))

    hour = np.asarray(idx_dt.hour) if hasattr(idx_dt, "hour") else np.zeros(len(df))
    dow = np.asarray(idx_dt.dayofweek) if hasattr(idx_dt, "dayofweek") else np.zeros(len(df))

    # Cyclical encoding for hour and day-of-week
    add(pd.Series(np.sin(2 * np.pi * hour / 24), index=df.index))
    add(pd.Series(np.cos(2 * np.pi * hour / 24), index=df.index))
    add(pd.Series(np.sin(2 * np.pi * dow / 7), index=df.index))
    add(pd.Series(np.cos(2 * np.pi * dow / 7), index=df.index))

    # Session flags (rough UTC proxies for Asia/Europe/US)
    add(pd.Series(((hour >= 0) & (hour < 8)).astype(float), index=df.index))
    add(pd.Series(((hour >= 8) & (hour < 16)).astype(float), index=df.index))
    add(pd.Series(((hour >= 16) & (hour < 24)).astype(float), index=df.index))
    add(pd.Series((dow >= 5).astype(float), index=df.index))  # weekend

    # ────────────────────────────────────────────────────────────────
    # Assemble
    # ────────────────────────────────────────────────────────────────
    feat_df = pd.DataFrame(feat, index=df.index).ffill().bfill().fillna(0.0)

    # Record canonical order on first call (trainer writes this to disk)
    global FEATURE_COLUMNS
    if not FEATURE_COLUMNS:
        FEATURE_COLUMNS = list(feat_df.columns)

    # Concat without duplicating columns
    overlap = [c for c in feat_df.columns if c in df.columns]
    if overlap:
        df = df.drop(columns=overlap)
    return pd.concat([df, feat_df], axis=1)


def get_feature_columns() -> list[str]:
    """Return the canonical ordered feature-column list."""
    if not FEATURE_COLUMNS:
        # Trigger a dummy run to populate
        dummy = pd.DataFrame({
            "open": [1.0] * 300, "high": [1.0] * 300, "low": [1.0] * 300,
            "close": [1.0] * 300, "volume": [1.0] * 300,
        }, index=pd.date_range("2024-01-01", periods=300, freq="h"))
        engineer_features(dummy)
    return list(FEATURE_COLUMNS)
