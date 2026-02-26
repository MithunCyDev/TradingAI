"""
Smart Money Concepts (SMC) filter for HQTS.

Detects order blocks, Fair Value Gaps, and liquidity sweeps to validate
entries. Used as Phase 2 filter after ML signal.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _swing_high_price(high: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    """Swing high price level: max among left+1+right bars."""
    return high.rolling(left + 1 + right, center=True).max()


def _swing_low_price(low: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    """Swing low price level: min among left+1+right bars."""
    return low.rolling(left + 1 + right, center=True).min()


def _bullish_order_block(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
) -> pd.Series:
    """
    Bullish OB: last down candle before a strong up move.
    Zone: low to open of that candle.
    """
    down = close < open_
    strong_up = (close - open_).shift(-1) > (high - low).shift(-1) * 0.5
    ob = down & strong_up.shift(1)
    return ob.astype(int)


def _bearish_order_block(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
) -> pd.Series:
    """Bearish OB: last up candle before a strong down move."""
    up = close > open_
    strong_down = (open_ - close).shift(-1) > (high - low).shift(-1) * 0.5
    ob = up & strong_down.shift(1)
    return ob.astype(int)


def _fvg_bullish_gap(high: pd.Series, low: pd.Series) -> pd.Series:
    """Bullish FVG gap size: low[i+1] - high[i-1]; NaN when no gap."""
    gap = low.shift(-1) - high.shift(1)
    return gap.where(gap > 0, np.nan)


def _fvg_bearish_gap(high: pd.Series, low: pd.Series) -> pd.Series:
    """Bearish FVG gap size: high[i+1] - low[i-1]; NaN when no gap."""
    gap = high.shift(1) - low.shift(-1)
    return gap.where(gap > 0, np.nan)


def _liquidity_sweep_bull(low: pd.Series, close: pd.Series, swing_low: pd.Series) -> pd.Series:
    """Price swept below swing low then reversed (bullish). Aligns with engineering."""
    prior_sl = swing_low.shift(1)
    swept_below = (low < prior_sl) & prior_sl.notna()
    reversed_up = close.shift(-1) > low
    return (swept_below & reversed_up.fillna(False)).astype(int)


def _liquidity_sweep_bear(high: pd.Series, close: pd.Series, swing_high: pd.Series) -> pd.Series:
    """Price swept above swing high then reversed (bearish). Aligns with engineering."""
    prior_sh = swing_high.shift(1)
    swept_above = (high > prior_sh) & prior_sh.notna()
    reversed_down = close.shift(-1) < high
    return (swept_above & reversed_down.fillna(False)).astype(int)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _ob_zone_strength(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, atr: pd.Series
) -> pd.Series:
    """OB zone strength (size/ATR) for bullish and bearish OBs combined."""
    ob_bull = _bullish_order_block(open_, high, low, close)
    ob_bear = _bearish_order_block(open_, high, low, close)
    size = (high - low).where(ob_bull.astype(bool) | ob_bear.astype(bool), np.nan)
    atr_safe = atr.replace(0, np.nan)
    strength = size / atr_safe
    return strength.fillna(0)


class SMCFilter:
    """
    Validates trade setups against SMC structure.

    Only allows BUY when price is in/near bullish OB or after bullish FVG/sweep.
    Only allows SELL when price is in/near bearish OB or after bearish FVG/sweep.
    """

    def __init__(
        self,
        require_order_block: bool = True,
        require_fvg: bool = False,
        require_liquidity_sweep: bool = False,
        ob_lookback: int = 20,
        fvg_min_size_atr: float = 0.3,
        min_ob_strength: float = 0.0,
        zone_width_atr: float = 0.5,
        require_price_in_zone: bool = False,
    ) -> None:
        self.require_order_block = require_order_block
        self.require_fvg = require_fvg
        self.require_liquidity_sweep = require_liquidity_sweep
        self.ob_lookback = ob_lookback
        self.fvg_min_size_atr = fvg_min_size_atr
        self.min_ob_strength = min_ob_strength
        self.zone_width_atr = zone_width_atr
        self.require_price_in_zone = require_price_in_zone

    def _last_atr(self, tail: pd.DataFrame) -> float:
        """Compute ATR from tail; use precomputed if available."""
        if "atr" in tail.columns:
            val = tail["atr"].iloc[-1]
            if not pd.isna(val) and val > 0:
                return float(val)
        atr = _atr(tail["high"], tail["low"], tail["close"])
        val = atr.iloc[-1]
        return float(val) if not pd.isna(val) and val > 0 else 1.0

    def _price_in_demand_zone(self, tail: pd.DataFrame, price: float) -> bool:
        """True if price is within zone_width_atr of swing low."""
        swing_low = _swing_low_price(tail["low"])
        atr_val = self._last_atr(tail)
        dist = abs(price - swing_low.iloc[-1])
        return dist <= self.zone_width_atr * atr_val

    def _price_in_supply_zone(self, tail: pd.DataFrame, price: float) -> bool:
        """True if price is within zone_width_atr of swing high."""
        swing_high = _swing_high_price(tail["high"])
        atr_val = self._last_atr(tail)
        dist = abs(swing_high.iloc[-1] - price)
        return dist <= self.zone_width_atr * atr_val

    def validate_buy(self, df: pd.DataFrame, current_price: Optional[float] = None) -> bool:
        """
        Check if current context allows a BUY per SMC rules.

        Args:
            df: Recent OHLCV DataFrame (last ob_lookback+ bars).
            current_price: Current price; if None, uses last close.

        Returns:
            True if SMC conditions for BUY are met.
        """
        if df.empty or len(df) < self.ob_lookback:
            return False
        tail = df.tail(self.ob_lookback)
        price = current_price if current_price is not None else float(tail["close"].iloc[-1])

        swing_low = _swing_low_price(tail["low"])
        ob_bull = _bullish_order_block(tail["open"], tail["high"], tail["low"], tail["close"])
        fvg_gap = _fvg_bullish_gap(tail["high"], tail["low"])
        sweep_bull = _liquidity_sweep_bull(tail["low"], tail["close"], swing_low)

        atr_val = self._last_atr(tail)
        has_ob = ob_bull.any()
        has_fvg = False
        if fvg_gap.notna().any():
            has_fvg = (fvg_gap.dropna() >= self.fvg_min_size_atr * atr_val).any()
        has_sweep = sweep_bull.any()

        if has_ob and self.min_ob_strength > 0:
            ob_strength = _ob_zone_strength(
                tail["open"], tail["high"], tail["low"], tail["close"],
                tail["atr"] if "atr" in tail.columns else _atr(tail["high"], tail["low"], tail["close"]),
            )
            max_strength = ob_strength[ob_bull.astype(bool)].max()
            if pd.isna(max_strength) or max_strength < self.min_ob_strength:
                has_ob = False

        if self.require_order_block and not has_ob:
            return False
        if self.require_fvg and not has_fvg:
            return False
        if self.require_liquidity_sweep and not has_sweep:
            return False

        if not (has_ob or has_fvg or has_sweep):
            return False

        if self.require_price_in_zone and not self._price_in_demand_zone(tail, price):
            return False

        return True

    def validate_sell(self, df: pd.DataFrame, current_price: Optional[float] = None) -> bool:
        """Check if current context allows a SELL per SMC rules."""
        if df.empty or len(df) < self.ob_lookback:
            return False
        tail = df.tail(self.ob_lookback)
        price = current_price if current_price is not None else float(tail["close"].iloc[-1])

        swing_high = _swing_high_price(tail["high"])
        ob_bear = _bearish_order_block(tail["open"], tail["high"], tail["low"], tail["close"])
        fvg_gap = _fvg_bearish_gap(tail["high"], tail["low"])
        sweep_bear = _liquidity_sweep_bear(tail["high"], tail["close"], swing_high)

        atr_val = self._last_atr(tail)
        has_ob = ob_bear.any()
        has_fvg = False
        if fvg_gap.notna().any():
            has_fvg = (fvg_gap.dropna() >= self.fvg_min_size_atr * atr_val).any()
        has_sweep = sweep_bear.any()

        if has_ob and self.min_ob_strength > 0:
            ob_strength = _ob_zone_strength(
                tail["open"], tail["high"], tail["low"], tail["close"],
                tail["atr"] if "atr" in tail.columns else _atr(tail["high"], tail["low"], tail["close"]),
            )
            max_strength = ob_strength[ob_bear.astype(bool)].max()
            if pd.isna(max_strength) or max_strength < self.min_ob_strength:
                has_ob = False

        if self.require_order_block and not has_ob:
            return False
        if self.require_fvg and not has_fvg:
            return False
        if self.require_liquidity_sweep and not has_sweep:
            return False

        if not (has_ob or has_fvg or has_sweep):
            return False

        if self.require_price_in_zone and not self._price_in_supply_zone(tail, price):
            return False

        return True
