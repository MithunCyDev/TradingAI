"""
Feature engineering for HQTS ML pipeline.

Computes: volatility (ATR), momentum (RSI), VWAP distance, time/session features,
and structural SMC primitives (swing highs/lows, FVG markers).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Trading session boundaries (UTC hour)
SESSION_ASIAN = (0, 8)
SESSION_LONDON = (8, 16)
SESSION_NEWYORK = (13, 21)
SESSION_OVERLAP_LN = (13, 16)  # London-New York overlap


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume-weighted average price (cumulative)."""
    typical = (high + low + close) / 3
    return (typical * volume).cumsum() / volume.cumsum().replace(0, np.nan)


def _swing_high(high: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    """Swing high: high is max among left+1+right bars."""
    return high.rolling(left + 1 + right, center=True).max()


def _swing_low(low: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    """Swing low: low is min among left+1+right bars."""
    return low.rolling(left + 1 + right, center=True).min()


def _is_swing_high(high: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    """Boolean: bar is a swing high."""
    sh = _swing_high(high, left, right)
    return (high >= sh) & (high.shift(-right) == sh)


def _is_swing_low(low: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    """Boolean: bar is a swing low."""
    sl = _swing_low(low, left, right)
    return (low <= sl) & (low.shift(-right) == sl)


def _fvg_bullish(high: pd.Series, low: pd.Series) -> pd.Series:
    """Fair Value Gap (bullish): gap between bar i-1 high and bar i+1 low."""
    gap = low.shift(-1) - high.shift(1)
    return gap.where(gap > 0, np.nan)


def _fvg_bearish(high: pd.Series, low: pd.Series) -> pd.Series:
    """Fair Value Gap (bearish): gap between bar i+1 high and bar i-1 low."""
    gap = high.shift(1) - low.shift(-1)
    return gap.where(gap > 0, np.nan)


def compute_features(
    df: pd.DataFrame,
    atr_period: int = 14,
    rsi_period: int = 14,
    swing_lookback: int = 2,
) -> pd.DataFrame:
    """
    Compute ML-ready features from OHLCV DataFrame.

    Args:
        df: DataFrame with columns: open, high, low, close, tick_volume, time.
        atr_period: ATR lookback period.
        rsi_period: RSI lookback period.
        swing_lookback: Bars left/right for swing high/low detection.

    Returns:
        DataFrame with original columns plus feature columns.
    """
    if df.empty:
        return df

    out = df.copy()
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["tick_volume"] if "tick_volume" in df.columns else pd.Series(1, index=df.index)

    # Volatility
    out["atr"] = _atr(high, low, close, atr_period)

    # Momentum
    out["rsi"] = _rsi(close, rsi_period)

    # VWAP distance (normalized by ATR to be scale-invariant)
    vwap = _vwap(high, low, close, volume)
    out["vwap_dist"] = (close - vwap) / out["atr"].replace(0, np.nan)

    # Time-based features
    if "time" in df.columns:
        dt = pd.to_datetime(df["time"])
        out["hour"] = dt.dt.hour
        out["dow"] = dt.dt.dayofweek
        out["session"] = 0  # 0=other, 1=asian, 2=london, 3=ny, 4=overlap
        out.loc[(out["hour"] >= SESSION_ASIAN[0]) & (out["hour"] < SESSION_ASIAN[1]), "session"] = 1
        out.loc[(out["hour"] >= SESSION_LONDON[0]) & (out["hour"] < SESSION_LONDON[1]), "session"] = 2
        out.loc[(out["hour"] >= SESSION_NEWYORK[0]) & (out["hour"] < SESSION_NEWYORK[1]), "session"] = 3
        out.loc[
            (out["hour"] >= SESSION_OVERLAP_LN[0]) & (out["hour"] < SESSION_OVERLAP_LN[1]),
            "session",
        ] = 4

    # Symbol encoding for multi-asset training (when symbol column exists)
    if "symbol" in df.columns:
        uniq = df["symbol"].unique()
        sym_map = {s: i for i, s in enumerate(sorted(uniq))}
        out["symbol_encoded"] = df["symbol"].map(sym_map).fillna(0).astype(int)

    # Timeframe encoding for multi-timeframe training (15m=0, 1h=1, 4h=2)
    if "timeframe" in df.columns:
        tf_map = {"15m": 0, "1h": 1, "4h": 2, "60m": 1, "m15": 0}
        out["timeframe_encoded"] = df["timeframe"].astype(str).str.lower().map(
            lambda x: tf_map.get(x, 0)
        ).fillna(0).astype(int)

    # Structural SMC primitives
    out["swing_high"] = _swing_high(high, swing_lookback, swing_lookback)
    out["swing_low"] = _swing_low(low, swing_lookback, swing_lookback)
    out["is_swing_high"] = _is_swing_high(high, swing_lookback, swing_lookback).astype(int)
    out["is_swing_low"] = _is_swing_low(low, swing_lookback, swing_lookback).astype(int)
    out["fvg_bullish"] = _fvg_bullish(high, low)
    out["fvg_bearish"] = _fvg_bearish(high, low)
    out["near_fvg_bull"] = (low <= out["fvg_bullish"].shift(1) + out["atr"] * 0.1).astype(int)
    out["near_fvg_bear"] = (high >= out["fvg_bearish"].shift(1) - out["atr"] * 0.1).astype(int)

    return out
