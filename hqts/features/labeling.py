"""
TP/SL-based labeling for HQTS ML pipeline.

Labels each bar based on whether TP (1:2 RR) or SL is hit first within N bars.
Classes: Up (TP hit first), Down (SL hit first), Ranging (neither within horizon).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LABEL_UP = 1
LABEL_DOWN = -1
LABEL_RANGE = 0


def compute_labels(
    df: pd.DataFrame,
    rr_ratio: float = 2.0,
    horizon_bars: int = 16,
    atr_mult_sl: float = 1.0,
    atr_mult_tp: Optional[float] = None,
) -> pd.Series:
    """
    Compute TP/SL-based labels for classification.

    For each bar, simulate a long entry at close. TP = close + atr * atr_mult_tp,
    SL = close - atr * atr_mult_sl. If rr_ratio=2, atr_mult_tp = 2 * atr_mult_sl.
    Look forward up to horizon_bars to see which level is hit first.

    Args:
        df: DataFrame with close, atr (or we compute from high/low/close).
        rr_ratio: Take-profit distance / stop-loss distance (e.g., 2 for 1:2 RR).
        horizon_bars: Max bars to look forward.
        atr_mult_sl: ATR multiplier for stop-loss distance.
        atr_mult_tp: ATR multiplier for TP. If None, uses atr_mult_sl * rr_ratio.

    Returns:
        Series with values: 1 (Up), -1 (Down), 0 (Ranging).
    """
    if df.empty:
        return pd.Series(dtype=int)

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    if "atr" in df.columns:
        atr = df["atr"].values
    else:
        # Fallback: simple ATR-14
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        atr = pd.Series(tr).rolling(14).mean().values

    tp_mult = atr_mult_tp if atr_mult_tp is not None else atr_mult_sl * rr_ratio
    n = len(close)
    labels = np.full(n, LABEL_RANGE, dtype=np.int32)

    for i in range(n - 1):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        entry = close[i]
        sl_dist = atr[i] * atr_mult_sl
        tp_dist = atr[i] * tp_mult
        sl_price = entry - sl_dist
        tp_price = entry + tp_dist

        for j in range(1, min(horizon_bars + 1, n - i)):
            idx = i + j
            bar_high = high[idx]
            bar_low = low[idx]

            hit_tp = bar_high >= tp_price
            hit_sl = bar_low <= sl_price

            if hit_tp and hit_sl:
                # Both hit in same bar: use bar close vs entry as tie-breaker
                labels[i] = LABEL_UP if close[idx] >= entry else LABEL_DOWN
                break
            elif hit_tp:
                labels[i] = LABEL_UP
                break
            elif hit_sl:
                labels[i] = LABEL_DOWN
                break

    return pd.Series(labels, index=df.index)


def compute_labels_pullback(
    df: pd.DataFrame,
    rr_ratio: float = 2.0,
    horizon_bars: int = 16,
    atr_mult_sl: float = 1.0,
    atr_mult_tp: Optional[float] = None,
    zone_width_atr: float = 0.75,
) -> pd.Series:
    """
    Pullback-aware labeling: only assign UP/DOWN when price is in a pullback zone.

    BUY (LABEL_UP): Only when in/near demand zone AND TP hits before SL.
    SELL (LABEL_DOWN): Only when in/near supply zone AND SL hits before TP (long fails = short wins).

    Bars not in a zone get LABEL_RANGE. This trains the model to recognize
    pullback entries that have higher success probability.
    """
    if df.empty:
        return pd.Series(dtype=int)

    base_labels = compute_labels(
        df, rr_ratio=rr_ratio, horizon_bars=horizon_bars,
        atr_mult_sl=atr_mult_sl, atr_mult_tp=atr_mult_tp,
    )

    close = df["close"].values
    atr = df["atr"].values if "atr" in df.columns else np.ones(len(df)) * np.nan
    swing_low = df["swing_low"].values if "swing_low" in df.columns else df["low"].values
    swing_high = df["swing_high"].values if "swing_high" in df.columns else df["high"].values

    n = len(close)
    labels = np.full(n, LABEL_RANGE, dtype=np.int32)

    for i in range(n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        in_demand = abs(close[i] - swing_low[i]) <= zone_width_atr * atr[i]
        in_supply = abs(swing_high[i] - close[i]) <= zone_width_atr * atr[i]

        base = int(base_labels.iloc[i])

        if base == LABEL_UP and in_demand:
            labels[i] = LABEL_UP
        elif base == LABEL_DOWN and in_supply:
            labels[i] = LABEL_DOWN
        else:
            labels[i] = LABEL_RANGE

    return pd.Series(labels, index=df.index)


def compute_labels_short(
    df: pd.DataFrame,
    rr_ratio: float = 2.0,
    horizon_bars: int = 16,
    atr_mult_sl: float = 1.0,
) -> pd.Series:
    """
    Same as compute_labels but for short entries: TP below, SL above.
    """
    if df.empty:
        return pd.Series(dtype=int)

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    if "atr" in df.columns:
        atr = df["atr"].values
    else:
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        atr = pd.Series(tr).rolling(14).mean().values

    tp_mult = atr_mult_sl * rr_ratio
    n = len(close)
    labels = np.full(n, LABEL_RANGE, dtype=np.int32)

    for i in range(n - 1):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        entry = close[i]
        sl_dist = atr[i] * atr_mult_sl
        tp_dist = atr[i] * tp_mult
        sl_price = entry + sl_dist
        tp_price = entry - tp_dist

        for j in range(1, min(horizon_bars + 1, n - i)):
            idx = i + j
            bar_high = high[idx]
            bar_low = low[idx]

            hit_tp = bar_low <= tp_price
            hit_sl = bar_high >= sl_price

            if hit_tp and hit_sl:
                dist_sl = abs(bar_high - sl_price)
                dist_tp = abs(bar_low - tp_price)
                labels[i] = LABEL_DOWN if dist_tp <= dist_sl else LABEL_UP
                break
            elif hit_tp:
                labels[i] = LABEL_DOWN
                break
            elif hit_sl:
                labels[i] = LABEL_UP
                break

    return pd.Series(labels, index=df.index)
