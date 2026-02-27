"""
Supply/demand zone detection and strength features.

Detects zones from swing points, order blocks, and FVGs; computes zone strength,
freshness, and distance for ML prediction.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _swing_low(low: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    """Swing low: low is min among left+1+right bars."""
    return low.rolling(left + 1 + right, center=True).min()


def _swing_high(high: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    """Swing high: high is max among left+1+right bars."""
    return high.rolling(left + 1 + right, center=True).max()


def _bullish_ob_zone(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Bullish OB: zone = low to open. Returns (zone_top, zone_bottom)."""
    down = close < open_
    strong_up = (close - open_).shift(-1) > (high - low).shift(-1) * 0.5
    ob = down & strong_up.shift(1)
    zone_top = open_.where(ob, np.nan)
    zone_bottom = low.where(ob, np.nan)
    return zone_top, zone_bottom


def _bearish_ob_zone(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Bearish OB: zone = open to high. Returns (zone_top, zone_bottom)."""
    up = close > open_
    strong_down = (open_ - close).shift(-1) > (high - low).shift(-1) * 0.5
    ob = up & strong_down.shift(1)
    zone_top = high.where(ob, np.nan)
    zone_bottom = open_.where(ob, np.nan)
    return zone_top, zone_bottom


def _zone_freshness_bars(idx: pd.Index, zone_formed: pd.Series) -> pd.Series:
    """Bars since zone formed at each bar."""
    n = len(idx)
    out = np.full(n, np.nan)
    for i in range(n):
        if pd.isna(zone_formed.iloc[i]) or not zone_formed.iloc[i]:
            continue
        out[i] = 0
        for j in range(i + 1, n):
            if pd.isna(out[j]):
                out[j] = j - i
    return pd.Series(out, index=idx)


def compute_supply_demand_features(
    df: pd.DataFrame,
    zone_width_atr: float = 0.5,
    freshness_bars: int = 20,
    swing_lookback: int = 2,
) -> pd.DataFrame:
    """
    Compute supply/demand zone features for ML.

    Args:
        df: DataFrame with open, high, low, close, atr, swing_high, swing_low.
        zone_width_atr: ATR multiplier for zone width.
        freshness_bars: Bars within which zone is considered "fresh".
        swing_lookback: For swing detection if not in df.

    Returns:
        DataFrame with new columns added (no copy of df; modifies in place).
    """
    if df.empty:
        return df

    high = df["high"]
    low = df["low"]
    close = df["close"]
    open_ = df["open"]

    if "atr" in df.columns:
        atr = df["atr"].replace(0, np.nan)
    else:
        atr = _atr(high, low, close)

    if "swing_low" in df.columns:
        swing_low = df["swing_low"]
    else:
        swing_low = _swing_low(low, swing_lookback, swing_lookback)
    if "swing_high" in df.columns:
        swing_high = df["swing_high"]
    else:
        swing_high = _swing_high(high, swing_lookback, swing_lookback)

    atr_safe = atr.replace(0, np.nan)
    zone_w = zone_width_atr * atr

    # Demand zone from swing low
    demand_bottom = swing_low
    demand_top = swing_low + zone_w
    in_demand_swing = (close >= demand_bottom) & (close <= demand_top) & demand_bottom.notna()
    dist_to_demand = (close - swing_low) / atr_safe
    dist_to_demand = dist_to_demand.fillna(0)

    # Supply zone from swing high
    supply_top = swing_high
    supply_bottom = swing_high - zone_w
    in_supply_swing = (close >= supply_bottom) & (close <= supply_top) & supply_top.notna()
    dist_to_supply = (swing_high - close) / atr_safe
    dist_to_supply = dist_to_supply.fillna(0)

    # Order block zones
    ob_bull_top, ob_bull_bot = _bullish_ob_zone(open_, high, low, close)
    ob_bear_top, ob_bear_bot = _bearish_ob_zone(open_, high, low, close)

    price_in_demand_ob = (
        (close >= ob_bull_bot) & (close <= ob_bull_top) & ob_bull_bot.notna()
    ).fillna(False).astype(int)
    price_in_supply_ob = (
        (close >= ob_bear_bot) & (close <= ob_bear_top) & ob_bear_bot.notna()
    ).fillna(False).astype(int)

    # Distance to nearest zones (signed: + when price above demand, + when price below supply)
    nearest_demand_dist = dist_to_demand
    nearest_supply_dist = dist_to_supply

    # Zone freshness: 1 if we were in zone within last N bars
    demand_zone_fresh = (in_demand_swing.astype(int).rolling(freshness_bars, min_periods=1).max() > 0).astype(int)
    supply_zone_fresh = (in_supply_swing.astype(int).rolling(freshness_bars, min_periods=1).max() > 0).astype(int)

    # Strength: 0-1, higher when in zone or closer to zone
    nearest_demand_strength = np.clip(1 - np.abs(nearest_demand_dist) / 3, 0, 1)
    nearest_demand_strength = np.where(in_demand_swing | (price_in_demand_ob == 1), np.maximum(nearest_demand_strength, 0.5), nearest_demand_strength)
    nearest_supply_strength = np.clip(1 - np.abs(nearest_supply_dist) / 3, 0, 1)
    nearest_supply_strength = np.where(in_supply_swing | (price_in_supply_ob == 1), np.maximum(nearest_supply_strength, 0.5), nearest_supply_strength)

    # sd_bias: +1 if in/near demand (bullish), -1 if in/near supply (bearish), 0 neutral
    in_demand = in_demand_swing | (price_in_demand_ob == 1)
    in_supply = in_supply_swing | (price_in_supply_ob == 1)
    sd_bias = np.where(in_demand & ~in_supply, 1, np.where(in_supply & ~in_demand, -1, 0))
    sd_bias = np.where(in_demand & in_supply, 0, sd_bias)
    neither = ~in_demand & ~in_supply
    sd_bias = np.where(neither, np.sign(nearest_demand_strength - nearest_supply_strength), sd_bias)

    df["nearest_demand_strength"] = nearest_demand_strength
    df["nearest_supply_strength"] = nearest_supply_strength
    df["price_in_demand_ob"] = price_in_demand_ob
    df["price_in_supply_ob"] = price_in_supply_ob
    df["dist_to_nearest_demand_atr"] = nearest_demand_dist
    df["dist_to_nearest_supply_atr"] = nearest_supply_dist
    df["demand_zone_fresh"] = demand_zone_fresh
    df["supply_zone_fresh"] = supply_zone_fresh
    df["sd_bias"] = sd_bias

    return df
