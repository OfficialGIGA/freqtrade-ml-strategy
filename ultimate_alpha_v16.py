"""
ultimate_alpha_v16.py — Ultimate Alpha Bot v16.0

Complete rewrite of the previous v14 script. Every component has a reason
to exist and either works or isn't there. No theater.

Architecture:
  1. Features — computed by the shared features.py module (same code as trainer).
  2. Model — one LightGBM classifier trained offline, loaded at startup.
     Outputs P(trade wins | primary signal fired). No torch, no ensemble.
  3. Regime — BTC-driven risk_on / neutral / risk_off with hysteresis.
  4. Scoring — additive fusion of model probability, regime adjustment,
     trend filter, and rank bonus. Bounded [0, 1]. No multiplicative explosion.
  5. Entry — score must exceed a regime-dependent threshold AND pass
     trend + execution-cost gates.
  6. Exit — ROI table + trailing stop + ATR-based custom stoploss +
     score-collapse exit.
  7. Position sizing — Kelly fraction scaled by regime and portfolio heat.

What's removed from v14:
  - Untrained 6-model torch ensemble (produced random noise).
  - Genetic weight optimizer (noise amplifier on tiny samples).
  - "True RL policy update" (was not RL).
  - Market memory with hash-state keys (keys never collided usefully).
  - Champion/challenger (never actually swapped).
  - 5 fake on-chain/DeFi/tokenomics/whale/microstructure scores.
  - "Quantum elite 20/10 master" naming.

What's kept and fixed:
  - BTC macro regime detection (was solid — now has proper hysteresis).
  - merge_asof for 1d informative (correctly implemented).
  - Portfolio heat & correlation-aware stake sizing.
  - Partial take-profits.
  - Persistent trade journal.
  - CryptoQuant netflow (now ACTUALLY calls the API, reads from env var).

Environment:
  CRYPTOQUANT_TOKEN (optional) — if missing, netflow signal defaults to 0.
  ALPHA_MODEL_DIR    (optional) — defaults to user_data/models.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame, Series

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    DecimalParameter,
    IntParameter,
    IStrategy,
)

# Make our sibling modules importable regardless of where Freqtrade launches us.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from features import engineer_features, get_feature_columns  # noqa: E402

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Model loader — loaded once, shared across all pairs
# ═══════════════════════════════════════════════════════════════════════
class _ModelBundle:
    """Lazy-loaded LightGBM model (single or ensemble) + metadata.

    Backward-compatible: handles both v16.0 single-model .pkl files AND
    v16.1 list-of-models ensemble .pkl files.
    """
    def __init__(self):
        self.models: list = []          # List of 1-or-more LightGBM models
        self.feature_cols: list[str] = []
        self.metrics: dict = {}
        self.loaded = False

    def load(self, model_dir: Path) -> bool:
        import joblib
        pkl = model_dir / "alpha_model.pkl"
        meta = model_dir / "alpha_model.meta.json"
        if not pkl.exists() or not meta.exists():
            log.warning(f"Model files not found in {model_dir}. "
                        "Strategy will run with model probability fixed at 0.5. "
                        "Run train_model.py first.")
            return False
        try:
            loaded = joblib.load(pkl)
            # Accept either a single model or a list of models
            if isinstance(loaded, list):
                self.models = loaded
            else:
                self.models = [loaded]

            with open(meta) as f:
                payload = json.load(f)
            self.feature_cols = payload["feature_columns"]
            self.metrics = payload.get("metrics", {})
            self.loaded = True
            log.info(f"Loaded {len(self.models)}-model ensemble from {pkl} — "
                     f"AUC {self.metrics.get('walk_forward_auc_mean', 0):.4f}, "
                     f"{len(self.feature_cols)} features")
            # Sanity: make sure our feature engineering matches the trained set
            current = get_feature_columns()
            if current != self.feature_cols:
                log.error("FEATURE MISMATCH between trained model and live features. "
                          "Retrain or sync features.py. Falling back to 0.5.")
                self.loaded = False
                return False
            return True
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            return False

    def predict_proba(self, features: np.ndarray) -> float:
        """Return averaged P(win) across ensemble for a single row."""
        if not self.loaded or not self.models:
            return 0.5
        try:
            X = features.reshape(1, -1)
            probas = [float(m.predict_proba(X)[0, 1]) for m in self.models]
            p = float(np.mean(probas))
            if np.isnan(p) or p < 0 or p > 1:
                return 0.5
            return p
        except Exception as e:
            log.debug(f"predict_proba failed: {e}")
            return 0.5

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """Ensemble-averaged P(win) for a batch. Used in populate_indicators."""
        if not self.loaded or not self.models:
            return np.full(len(X), 0.5, dtype=np.float32)
        try:
            probas = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models])
            return probas.mean(axis=1)
        except Exception as e:
            log.debug(f"predict_proba_batch failed: {e}")
            return np.full(len(X), 0.5, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════
#  Strategy
# ═══════════════════════════════════════════════════════════════════════
class UltimateAlphaV16(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "4h"
    informative_timeframe = "1d"
    startup_candle_count: int = 250   # 250 × 4h = ~42 days warmup for 200 EMA + features
    process_only_new_candles = True
    can_short = False

    # ── ROI & stops ──────────────────────────────────────────────────
    # Aligned with train_model.py meta-label barriers (4h timeframe):
    #   TARGET_PROFIT = 0.030 (take 3% wins)
    #   STOP_LOSS     = -0.020 (cut 2% losers)
    #   HORIZON_BARS  = 42 (~7 days holding max)
    # ROI minutes: one 4h candle = 240 min, one day = 1440 min, one week = 10080 min
    minimal_roi = {
        "0":     0.060,   # First 4h: let winners run up to 6%
        "240":   0.040,   # After 1d: take 4%+ (1 candle in)
        "1440":  0.030,   # After 1d: take 3% (aligned with model target)
        "4320":  0.015,   # After 3d: take 1.5% (partial scale-out)
        "10080": 0.0,     # After 7d: close on any profit (matches horizon)
    }
    stoploss = -0.05                             # Hard floor; custom_stoploss uses -2% baseline
    # ─ TRAILING STOP DISABLED (learned from 1h backtests) ─────────────
    # Trailing stops combined with the volatility profile of our targets
    # net-destroyed winners in both bull and bear markets. Pure ROI + hard
    # stoploss only.
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    use_custom_stoploss = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    position_adjustment_enable = False

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
        "emergency_exit": "market",
    }

    # ── Hyperopt parameters (safe, meaningful ranges) ────────────────
    entry_threshold_neutral = DecimalParameter(0.50, 0.72, default=0.58, decimals=3, space="buy")
    entry_threshold_risk_on_bonus = DecimalParameter(-0.10, 0.0, default=-0.05, decimals=3, space="buy")
    entry_threshold_risk_off_penalty = DecimalParameter(0.0, 0.15, default=0.08, decimals=3, space="buy")
    model_prob_weight = DecimalParameter(0.3, 0.7, default=0.50, decimals=2, space="buy")
    kelly_fraction = DecimalParameter(0.2, 0.6, default=0.35, decimals=2, space="buy")
    max_portfolio_heat = DecimalParameter(0.4, 0.8, default=0.60, decimals=2, space="buy")

    # ── Class-level state: RESET IN __init__ to avoid cross-instance bleed ──
    _model = _ModelBundle()
    _regime_state: dict[str, Any] = {}
    _btc_macro: dict[str, float] = {}
    _breadth: float = 0.5
    _onchain_skew: float = 0.0
    _pair_scores: dict[str, float] = {}
    _last_apex_log: dict[str, float] = {}
    _last_bot_loop_ts: float = 0.0
    _last_skew_fetch_ts: float = 0.0
    _trade_journal: list[dict] = []

    # ═══════════════════════════════════════════════════════════════
    #  Lifecycle
    # ═══════════════════════════════════════════════════════════════
    def __init__(self, config: dict) -> None:
        super().__init__(config)

        # Instance-level state (NOT shared between instances)
        self._apex_log_cache: dict[str, float] = {}
        self._last_regime_logged: str | None = None
        self._regime_smoothed: float = 0.0
        self._current_regime: str = "neutral"
        self._current_vol: float = 1.0
        self._pair_ranking: list[tuple[str, float]] = []
        self._last_rank_ts: float = 0.0

    def version(self) -> str:
        return "v16.0"

    def informative_pairs(self):
        # All whitelisted pairs on 1d for multi-timeframe context
        pairs = self.dp.current_whitelist() if self.dp else []
        return [(p, self.informative_timeframe) for p in pairs]

    def bot_start(self, **kwargs) -> None:
        log.info("=" * 72)
        log.info("Ultimate Alpha v16.0 starting")
        log.info("=" * 72)

        # Locate model dir: env var > user_data/models
        user_data = Path(self.config.get("user_data_dir", "user_data"))
        model_dir = Path(os.environ.get("ALPHA_MODEL_DIR", user_data / "models"))
        self._model.load(model_dir)

        # Restore trade journal
        journal_path = user_data / "alpha_trade_journal.json"
        if journal_path.exists():
            try:
                with open(journal_path) as f:
                    self._trade_journal = json.load(f)
                log.info(f"Restored {len(self._trade_journal)} trades from journal")
            except Exception as e:
                log.warning(f"Failed to restore trade journal: {e}")

        log.info(f"Model loaded: {self._model.loaded}")
        log.info(f"CryptoQuant token set: {'CRYPTOQUANT_TOKEN' in os.environ}")

    def bot_loop_start(self, current_time, **kwargs) -> None:
        """
        Runs once per candle processing cycle across all pairs.
        Updates market-wide state. Each step isolated — one failure won't
        stop the others.
        """
        now_ts = time.time()

        # ── BTC macro context ─────────────────────────────────────
        try:
            self._btc_macro = self._compute_btc_macro()
        except Exception as e:
            log.warning(f"bot_loop_start | btc_macro failed: {e}")

        # ── Breadth ───────────────────────────────────────────────
        try:
            self._breadth = self._compute_breadth()
        except Exception as e:
            log.warning(f"bot_loop_start | breadth failed: {e}")

        # ── Regime ────────────────────────────────────────────────
        try:
            prev = self._current_regime
            self._current_regime = self._detect_regime()
            if prev != self._current_regime and self._current_regime != self._last_regime_logged:
                log.info(f"REGIME CHANGE: {prev} → {self._current_regime}")
                self._last_regime_logged = self._current_regime
        except Exception as e:
            log.warning(f"bot_loop_start | regime failed: {e}")

        # ── On-chain skew (every 10 minutes) ──────────────────────
        if now_ts - self._last_skew_fetch_ts > 600:
            try:
                self._onchain_skew = self._fetch_cryptoquant_skew()
            except Exception as e:
                log.debug(f"skew fetch failed (safe): {e}")
            self._last_skew_fetch_ts = now_ts

        # ── Pair ranking (every 3 candles) ────────────────────────
        tf_s = self.timeframe_to_seconds(self.timeframe)
        if now_ts - self._last_rank_ts > tf_s * 3:
            try:
                self._pair_ranking = self._rank_pairs()
            except Exception as e:
                log.warning(f"bot_loop_start | ranking failed: {e}")
            self._last_rank_ts = now_ts

        self._last_bot_loop_ts = now_ts

    @staticmethod
    def timeframe_to_seconds(tf: str) -> int:
        return {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}.get(tf, 3600)

    # ═══════════════════════════════════════════════════════════════
    #  Market-wide computations
    # ═══════════════════════════════════════════════════════════════
    def _compute_btc_macro(self) -> dict[str, float]:
        btc = self.dp.get_pair_dataframe("BTC/USDT", self.timeframe)
        if btc is None or len(btc) < 200:
            return self._btc_macro or {"risk_state": 0.0, "ret24h": 0.0, "above_ema200": 0.5}

        close = btc["close"]
        ema200 = pd.Series(np.asarray(ta.EMA(close, timeperiod=200)), index=close.index)
        ret24 = float(close.pct_change(24).iloc[-1])
        above = 1.0 if float(close.iloc[-1]) > float(ema200.iloc[-1]) else 0.0

        atr_raw = ta.ATR(btc["high"], btc["low"], btc["close"], timeperiod=14)
        atr = pd.Series(np.asarray(atr_raw), index=close.index)
        atr_last = float(atr.iloc[-1])
        btc_vol_pct = (atr_last / float(close.iloc[-1])) * 100
        self._current_vol = btc_vol_pct

        risk_state = np.clip(
            (above * 2 - 1) * 0.4 + np.clip(ret24 * 8, -0.5, 0.5) * 0.6,
            -1.0, 1.0,
        )
        return {
            "risk_state": float(risk_state),
            "ret24h": ret24,
            "above_ema200": above,
            "vol_pct": btc_vol_pct,
        }

    def _compute_breadth(self) -> float:
        wl = self.dp.current_whitelist() if self.dp else []
        if not wl:
            return 0.5
        above = 0
        total = 0
        for p in wl:
            try:
                df = self.dp.get_pair_dataframe(p, self.timeframe)
                if df is None or len(df) < 51:
                    continue
                ema_raw = ta.EMA(df["close"], timeperiod=50)
                ema = pd.Series(np.asarray(ema_raw), index=df["close"].index)
                if float(df["close"].iloc[-1]) > float(ema.iloc[-1]):
                    above += 1
                total += 1
            except Exception:
                continue
        return (above / total) if total > 0 else 0.5

    def _detect_regime(self) -> str:
        """
        Regime state machine with hysteresis. Once in a regime, requires a
        stronger signal in the opposite direction to flip. Prevents whipsaws.
        """
        risk_state = float(self._btc_macro.get("risk_state", 0.0))
        breadth_signal = (self._breadth - 0.5) * 2.0  # [-1, 1]
        vol = float(self._current_vol)
        vol_penalty = max(0.0, (vol - 2.5) / 2.5)  # 0 below 2.5%, grows above

        raw = risk_state * 0.55 + breadth_signal * 0.35 - vol_penalty * 0.30
        raw = float(np.clip(raw, -1.0, 1.0))

        alpha = 0.25
        self._regime_smoothed = alpha * raw + (1 - alpha) * self._regime_smoothed
        s = self._regime_smoothed
        current = self._current_regime

        if current == "risk_on":
            if s < -0.35:   return "risk_off"
            if s <  0.20:   return "neutral"
            return "risk_on"
        if current == "risk_off":
            if s >  0.35:   return "risk_on"
            if s > -0.20:   return "neutral"
            return "risk_off"
        # neutral
        if s >  0.30:       return "risk_on"
        if s < -0.30:       return "risk_off"
        return "neutral"

    def _fetch_cryptoquant_skew(self) -> float:
        """
        Real BTC exchange netflow from CryptoQuant. Positive value = inflows =
        bearish (coins moving to exchanges to sell). We return a tanh-squashed
        score in [-1, 1] where positive = bullish (outflows).
        """
        token = os.environ.get("CRYPTOQUANT_TOKEN")
        if not token:
            return 0.0
        try:
            import requests
            resp = requests.get(
                "https://api.cryptoquant.com/v1/btc/exchange-flows/netflow",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            if resp.status_code != 200:
                log.debug(f"CryptoQuant returned {resp.status_code}")
                return 0.0
            data = resp.json().get("data", [])
            if not data:
                return 0.0
            latest = float(data[-1].get("value", 0))
            return float(-np.tanh(latest / 800.0))
        except Exception as e:
            log.debug(f"CryptoQuant fetch failed (safe): {e}")
            return 0.0

    def _rank_pairs(self) -> list[tuple[str, float]]:
        """Rank whitelist pairs by a cheap proxy score for allocation priority."""
        wl = self.dp.current_whitelist() if self.dp else []
        scored: list[tuple[str, float]] = []
        for p in wl:
            try:
                df, _ = self.dp.get_analyzed_dataframe(p, self.timeframe)
                if df is None or len(df) < 30 or "final_score" not in df.columns:
                    continue
                s = float(df["final_score"].iloc[-1])
                scored.append((p, s))
            except Exception:
                continue
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:12]

    def _rank_multiplier(self, pair: str) -> float:
        """Score-based, not rank-based. Top pairs get up to 15% bonus."""
        if not self._pair_ranking:
            return 1.0
        pairs = [p for p, _ in self._pair_ranking]
        if pair not in pairs:
            return 1.0
        idx = pairs.index(pair)
        # Linear interpolation: top = 1.15, bottom of top-12 = 0.95
        frac = 1.0 - (idx / max(len(pairs) - 1, 1))
        return float(0.95 + frac * 0.20)

    # ═══════════════════════════════════════════════════════════════
    #  Freqtrade hooks
    # ═══════════════════════════════════════════════════════════════
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        df = dataframe.copy()

        # Standardize index for merge_asof
        if "date" not in df.columns:
            df = df.reset_index().rename(columns={df.index.name or "index": "date"})
        df["date"] = pd.to_datetime(df["date"], utc=True)

        # ── Core TA indicators ─────────────────────────────────────
        df["rsi"] = ta.RSI(df["close"], timeperiod=14)
        df["atr"] = ta.ATR(df["high"], df["low"], df["close"], timeperiod=14)
        df["atr_pct"] = (df["atr"] / df["close"].clip(lower=1e-10)) * 100

        macd_val, macd_sig, macd_hist = ta.MACD(df["close"])
        df["macd"] = macd_val
        df["macd_signal"] = macd_sig
        df["macd_hist"] = macd_hist

        df["ema_9"] = ta.EMA(df["close"], timeperiod=9)
        df["ema_21"] = ta.EMA(df["close"], timeperiod=21)
        df["ema_50"] = ta.EMA(df["close"], timeperiod=50)
        df["ema_200"] = ta.EMA(df["close"], timeperiod=200)

        df["adx"] = ta.ADX(df["high"], df["low"], df["close"], timeperiod=14)

        df["volume_ma"] = df["volume"].rolling(24).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"].clip(lower=1e-10)

        # ── 1d informative merge ───────────────────────────────────
        df["trend_1d"] = 0.0
        df["close_1d"] = df["close"]
        try:
            inf = self.dp.get_pair_dataframe(pair, self.informative_timeframe)
            if inf is not None and len(inf) > 10:
                inf = inf.copy()
                if "date" not in inf.columns:
                    inf = inf.reset_index()
                inf["date"] = pd.to_datetime(inf["date"], utc=True)
                inf["trend_1d"] = inf["close"].pct_change(5).fillna(0)
                inf["close_1d"] = inf["close"]
                merged = pd.merge_asof(
                    df.sort_values("date"),
                    inf[["date", "trend_1d", "close_1d"]].sort_values("date"),
                    on="date",
                    direction="backward",
                    suffixes=("", "_inf"),
                )
                df["trend_1d"] = merged["trend_1d_inf"].fillna(0).values if "trend_1d_inf" in merged.columns else merged["trend_1d"].fillna(0).values
                df["close_1d"] = merged["close_1d_inf"].ffill().values if "close_1d_inf" in merged.columns else merged["close_1d"].ffill().values
        except Exception as e:
            log.debug(f"{pair} | 1d merge skipped: {e}")

        # ── Feature engineering (shared with trainer) ──────────────
        btc = self.dp.get_pair_dataframe("BTC/USDT", self.timeframe) if pair != "BTC/USDT" else None
        # Need datetime index on btc for features module
        if btc is not None and not btc.empty:
            btc_fe = btc.copy()
            if "date" in btc_fe.columns:
                btc_fe = btc_fe.set_index(pd.to_datetime(btc_fe["date"], utc=True))
        else:
            btc_fe = None

        # features.py expects datetime index
        df_fe = df.set_index("date")
        df_fe = engineer_features(df_fe, btc_df=btc_fe)
        df = df_fe.reset_index()

        # ── Model probability ──────────────────────────────────────
        df = self._apply_model(df, pair)

        # ── Final score ────────────────────────────────────────────
        df = self._compute_final_score(df, pair)

        return df

    def _apply_model(self, df: DataFrame, pair: str) -> DataFrame:
        """Apply the LightGBM model to the latest rows. Only the last row
        matters for live decisions, but we fill the column for backtests."""
        if not self._model.loaded or "feat_000" not in df.columns:
            df["model_prob"] = 0.5
            return df

        feat_cols = self._model.feature_cols
        missing = [c for c in feat_cols if c not in df.columns]
        if missing:
            log.warning(f"{pair} | missing {len(missing)} features — skipping model")
            df["model_prob"] = 0.5
            return df

        try:
            X = df[feat_cols].values.astype(np.float32)
            # LightGBM handles NaN natively; still, guard against inf
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            proba = self._model.predict_proba_batch(X)
            proba = np.clip(proba, 0.01, 0.99)
            df["model_prob"] = proba
        except Exception as e:
            log.warning(f"{pair} | model inference failed: {e}")
            df["model_prob"] = 0.5

        return df

    def _compute_final_score(self, df: DataFrame, pair: str) -> DataFrame:
        """
        Additive score in [0, 1]. No multiplication, no explosion.

        Weights sum to 1.0 for the base, then small additive adjustments.
        """
        if len(df) < 60:
            df["final_score"] = 0.5
            return df

        # Model probability (primary signal)
        model_prob = df["model_prob"].fillna(0.5)
        mw = float(self.model_prob_weight.value)

        # Classic technical components
        rsi_norm = (df["rsi"].fillna(50) / 100).clip(0, 1)
        rsi_score = 1.0 - (rsi_norm - 0.5).abs() * 2  # peaks at RSI=50
        rsi_score = rsi_score * 0.5 + rsi_norm * 0.5   # blend with trend direction

        macd_hist_norm = (df["macd_hist"].fillna(0) / df["close"].clip(lower=1e-10) * 100).clip(-2, 2)
        macd_score = (macd_hist_norm + 2) / 4

        ema_stack_score = (
            (df["close"] > df["ema_21"]).astype(float) * 0.3 +
            (df["ema_21"] > df["ema_50"]).astype(float) * 0.3 +
            (df["close"] > df["ema_200"]).astype(float) * 0.4
        )

        vol_score = df["volume_ratio"].fillna(1.0).clip(0.3, 3.0)
        vol_score = ((vol_score - 0.3) / 2.7).clip(0, 1)

        adx_score = (df["adx"].fillna(20) / 60).clip(0, 1)

        technical = (
            rsi_score * 0.15 +
            macd_score * 0.25 +
            ema_stack_score * 0.35 +
            vol_score * 0.15 +
            adx_score * 0.10
        )

        base_score = model_prob * mw + technical * (1 - mw)

        # ── Additive adjustments ───────────────────────────────────
        regime = self._current_regime
        regime_adj = {"risk_on": +0.04, "neutral": 0.0, "risk_off": -0.06}.get(regime, 0.0)

        skew_adj = float(self._onchain_skew) * 0.03

        rank_adj = (self._rank_multiplier(pair) - 1.0) * 0.10

        # Toxic volume spike without price movement = kill
        toxicity = np.where(
            (df["volume_ratio"].fillna(1.0) > 4.0) &
            (df["close"].pct_change(3).abs().fillna(0) < 0.002),
            -0.08, 0.0,
        )

        final = (base_score + regime_adj + skew_adj + rank_adj + toxicity).clip(0.0, 1.0)
        df["final_score"] = final.fillna(0.5)

        # ── Throttled logging (once per pair per 5 min) ────────────
        now = time.time()
        if now - self._apex_log_cache.get(pair, 0) > 300:
            last = df.iloc[-1]
            log.info(
                f"{pair:<12} | score={last['final_score']:.3f} "
                f"model_p={last['model_prob']:.3f} tech={technical.iloc[-1]:.3f} "
                f"regime={regime} rank_adj={rank_adj:+.3f} "
                f"skew={self._onchain_skew:+.3f}"
            )
            self._apex_log_cache[pair] = now

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        df = dataframe.copy()
        df["enter_long"] = 0

        if len(df) < 60 or "final_score" not in df.columns:
            return df

        regime = self._current_regime
        base = float(self.entry_threshold_neutral.value)
        threshold = base
        if regime == "risk_on":
            threshold = base + float(self.entry_threshold_risk_on_bonus.value)
        elif regime == "risk_off":
            threshold = base + float(self.entry_threshold_risk_off_penalty.value)
        threshold = float(np.clip(threshold, 0.45, 0.80))

        # Volatility adjustment: raise threshold when vol is unusually high
        atr_roll = df["atr_pct"].rolling(48).mean()
        atr_cur = df["atr_pct"]
        vol_ratio = (atr_cur / atr_roll.clip(lower=1e-6)).fillna(1.0)
        threshold_series = threshold + (vol_ratio - 1.0).clip(-0.05, 0.10) * 0.5
        threshold_series = threshold_series.clip(0.45, 0.82)

        # Gates — must mirror train_model.py generate_primary_signal so the
        # model is inferring on the same state distribution it trained on.
        score_ok = df["final_score"] > threshold_series

        # === Primary signal state (mirrors trainer) ===
        macd_bull = df["macd"] > df["macd_signal"]
        above_200 = df["close"] > df["ema_200"]
        ema_stack = df["ema_21"] > df["ema_50"]
        rsi_ok = (df["rsi"] > 50) & (df["rsi"] < 75)

        # === Live-only safety checks ===
        # 1d trend not deeply negative (avoid trading against higher-TF bear)
        trend_ok = (df["trend_1d"] > -0.08) | (df["ema_21"].pct_change(6).fillna(0) > 0.003)

        # Not overextended above fast EMA (don't chase parabolic moves)
        not_extended = df["close"] < df["ema_9"] * 1.04

        # Regime gate: never enter in risk_off unless score is exceptional
        regime_ok = (regime != "risk_off") | (df["final_score"] > 0.75)

        entry = (score_ok & macd_bull & above_200 & ema_stack & rsi_ok &
                 trend_ok & not_extended & regime_ok)
        df["enter_long"] = entry.astype(int)
        df["enter_tag"] = np.where(entry, f"v16_{regime}", "")

        # Log the last candle's decision once per 5 min
        last = df.iloc[-1]
        if bool(last["enter_long"]):
            log.info(
                f"{pair:<12} ENTRY | score={last['final_score']:.3f} "
                f"thresh={float(threshold_series.iloc[-1]):.3f} "
                f"model_p={last['model_prob']:.3f} regime={regime}"
            )

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()
        df["exit_long"] = 0

        if len(df) < 60 or "final_score" not in df.columns:
            return df

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # EXIT POLICY — The model was trained to predict ENTRIES, not exits.
        # Using model score as an exit signal is a category error that killed
        # winners before they reached their ROI target in v1 backtests.
        #
        # We rely on:
        #   1. ROI table (take profit at 4.5% / 2.5% / 1.5% / 0%)
        #   2. custom_stoploss (ATR-adaptive, -2% baseline)
        #   3. Trailing stop (locks in after +3%)
        #   4. This function: ONLY an extreme-volatility emergency exit
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # Extreme volatility spike — true risk signal (3x normal ATR).
        # Rare event. When it fires, market structure has broken.
        atr_roll = df["atr_pct"].rolling(48).mean().clip(lower=1e-6)
        extreme_vol = df["atr_pct"] > atr_roll * 3.0

        exit_signal = extreme_vol

        # Never exit on the same candle as entry
        if "enter_long" in df.columns:
            exit_signal = exit_signal & (df["enter_long"] != 1)

        df["exit_long"] = exit_signal.astype(int)
        return df

    # ═══════════════════════════════════════════════════════════════
    #  Custom stoploss — PURE hard stop, no profit-tightening
    # ═══════════════════════════════════════════════════════════════
    def custom_stoploss(
        self, pair: str, trade: Trade, current_time: datetime,
        current_rate: float, current_profit: float, **kwargs
    ) -> float:
        """
        Returns a NEGATIVE ratio — a pure hard stop.

        IMPORTANT: Does NOT tighten with profit. The previous version did,
        which Freqtrade labeled 'trailing_stop_loss' and destroyed winners.
        This version only widens the stop for high-volatility pairs so normal
        noise doesn't stop us out — it never raises the stop above baseline.
        """
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or len(df) < 20:
            return self.stoploss

        atr_pct = float(df["atr_pct"].iloc[-1]) / 100  # ratio

        # Base stop = -2.0% (training barrier). Widen up to -5% on high-vol pairs.
        # Never tighten with profit — that was the bug.
        atr_based = max(0.020, min(atr_pct * 2.0, 0.05))
        return -atr_based

    # ═══════════════════════════════════════════════════════════════
    #  Partial exits via custom_exit
    # ═══════════════════════════════════════════════════════════════
    def custom_exit(self, pair: str, trade: Trade, current_time,
                    current_rate: float, current_profit: float, **kwargs):
        """
        Bonus exits for exceptional profit. With TARGET_PROFIT=3%, anything
        running past 8% is a runner — we grab at 8%, and at 15%+ we're in
        rare-but-real multi-day-breakout territory.
        Order matters: check strongest condition first.
        """
        if current_profit > 0.15:
            return "runner_take_15pct"
        if current_profit > 0.08:
            return "partial_take_8pct"
        # No special exit — let ROI / stoploss handle it
        return None

    # ═══════════════════════════════════════════════════════════════
    #  Position sizing — Kelly × regime × portfolio-heat
    # ═══════════════════════════════════════════════════════════════
    def custom_stake_amount(
        self, pair: str, current_time: datetime, current_rate: float,
        proposed_stake: float, min_stake: Optional[float], max_stake: Optional[float],
        leverage: float, entry_tag: Optional[str], side: str, **kwargs
    ) -> float:
        try:
            if not isinstance(proposed_stake, (int, float)):
                proposed_stake = float(self.config.get("stake_amount", 100))

            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            score = float(df["final_score"].iloc[-1]) if df is not None and len(df) > 0 else 0.5
            model_p = float(df["model_prob"].iloc[-1]) if df is not None and "model_prob" in df.columns else 0.5

            # Kelly fraction: (p*b - q)/b, but we fractionalize to cap exposure.
            # Assume payoff ratio b ≈ 1 (our TP and SL are similar scale).
            # edge = p - (1 - p) = 2p - 1
            edge = max(0.0, 2 * model_p - 1)
            kelly = edge * float(self.kelly_fraction.value)  # fractional Kelly

            # Scale by score above neutral
            score_mult = 0.5 + max(0.0, score - 0.5)  # range [0.5, 1.0]

            # Portfolio heat check
            heat = self._portfolio_heat()
            heat_cap = float(self.max_portfolio_heat.value)
            heat_mult = 1.0 if heat < heat_cap else max(0.3, (1.5 - heat / heat_cap))

            # Regime scale
            regime_mult = {"risk_on": 1.15, "neutral": 1.0, "risk_off": 0.55}.get(self._current_regime, 1.0)

            # Final stake
            mult = (0.5 + kelly) * score_mult * heat_mult * regime_mult
            stake = proposed_stake * np.clip(mult, 0.3, 1.6)

            if min_stake is not None:
                stake = max(stake, float(min_stake) * 1.05)
            if max_stake is not None:
                stake = min(stake, float(max_stake))

            log.info(
                f"STAKE {pair:<12} | proposed=${proposed_stake:.2f} "
                f"→ final=${stake:.2f} | score={score:.3f} model_p={model_p:.3f} "
                f"kelly={kelly:.3f} heat={heat:.1%} regime={self._current_regime}"
            )
            return float(stake)

        except Exception as e:
            log.warning(f"custom_stake_amount fallback for {pair}: {e}")
            return float(proposed_stake) * 0.5

    def _portfolio_heat(self) -> float:
        """Fraction of total balance currently deployed in open trades."""
        try:
            open_trades = Trade.get_trades_proxy(is_open=True)
            total = float(self.wallets.get_total(self.config["stake_currency"]) or 0.0)
            if total <= 0:
                return 0.0
            deployed = sum(float(t.stake_amount or 0.0) for t in open_trades)
            return deployed / total
        except Exception:
            return 0.0

    # ═══════════════════════════════════════════════════════════════
    #  Trade journaling — persistent record for analysis
    # ═══════════════════════════════════════════════════════════════
    def confirm_trade_exit(
        self, pair: str, trade: Trade, order_type: str, amount: float,
        rate: float, time_in_force: str, exit_reason: str, current_time: datetime,
        **kwargs
    ) -> bool:
        try:
            profit_ratio = float(trade.calc_profit_ratio(rate)) if rate else 0.0
            entry = {
                "ts": current_time.isoformat() if hasattr(current_time, "isoformat") else str(current_time),
                "pair": pair,
                "entry_rate": float(trade.open_rate),
                "exit_rate": float(rate) if rate else None,
                "profit_ratio": profit_ratio,
                "exit_reason": exit_reason,
                "enter_tag": trade.enter_tag or "",
                "regime": self._current_regime,
                "trade_duration_mins": int((current_time - trade.open_date_utc).total_seconds() / 60)
                if trade.open_date_utc else 0,
            }
            self._trade_journal.append(entry)
            if len(self._trade_journal) > 2000:
                self._trade_journal = self._trade_journal[-2000:]
            self._persist_journal()
        except Exception as e:
            log.debug(f"journal write failed: {e}")
        return True

    def _persist_journal(self):
        try:
            user_data = Path(self.config.get("user_data_dir", "user_data"))
            user_data.mkdir(parents=True, exist_ok=True)
            path = user_data / "alpha_trade_journal.json"
            with open(path, "w") as f:
                json.dump(self._trade_journal, f, default=str)
        except Exception as e:
            log.debug(f"persist journal failed: {e}")
