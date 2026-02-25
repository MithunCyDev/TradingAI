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


def _swing_highs(high: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    """Identify swing high bars (high is max in window)."""
    roll = high.rolling(left + 1 + right, center=True)
    return (high >= roll.max()).astype(int)


def _swing_lows(low: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    """Identify swing low bars (low is min in window)."""
    roll = low.rolling(left + 1 + right, center=True)
    return (low <= roll.min()).astype(int)


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


def _fvg_bullish(high: pd.Series, low: pd.Series) -> pd.Series:
    """Bullish FVG: gap between bar i-1 high and bar i+1 low."""
    gap = low.shift(-1) - high.shift(1)
    return (gap > 0).astype(int)


def _fvg_bearish(high: pd.Series, low: pd.Series) -> pd.Series:
    """Bearish FVG: gap between bar i+1 high and bar i-1 low."""
    gap = high.shift(1) - low.shift(-1)
    return (gap > 0).astype(int)


def _liquidity_sweep_bull(low: pd.Series, swing_low: pd.Series) -> pd.Series:
    """Price swept below recent swing low then reversed (bullish)."""
    sl = low.rolling(10).min()
    sweep = (low < sl.shift(1)) & (low.shift(-1) > low)
    return sweep.astype(int)


def _liquidity_sweep_bear(high: pd.Series, swing_high: pd.Series) -> pd.Series:
    """Price swept above recent swing high then reversed (bearish)."""
    sh = high.rolling(10).max()
    sweep = (high > sh.shift(1)) & (high.shift(-1) < high)
    return sweep.astype(int)


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
    ) -> None:
        self.require_order_block = require_order_block
        self.require_fvg = require_fvg
        self.require_liquidity_sweep = require_liquidity_sweep
        self.ob_lookback = ob_lookback

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

        ob_bull = _bullish_order_block(tail["open"], tail["high"], tail["low"], tail["close"])
        fvg_bull = _fvg_bullish(tail["high"], tail["low"])
        sweep_bull = _liquidity_sweep_bull(tail["low"], _swing_lows(tail["low"]))

        has_ob = ob_bull.any()
        has_fvg = fvg_bull.any()
        has_sweep = sweep_bull.any()

        if self.require_order_block and not has_ob:
            return False
        if self.require_fvg and not has_fvg:
            return False
        if self.require_liquidity_sweep and not has_sweep:
            return False

        return has_ob or has_fvg or has_sweep

    def validate_sell(self, df: pd.DataFrame, current_price: Optional[float] = None) -> bool:
        """Check if current context allows a SELL per SMC rules."""
        if df.empty or len(df) < self.ob_lookback:
            return False
        tail = df.tail(self.ob_lookback)
        price = current_price if current_price is not None else float(tail["close"].iloc[-1])

        ob_bear = _bearish_order_block(tail["open"], tail["high"], tail["low"], tail["close"])
        fvg_bear = _fvg_bearish(tail["high"], tail["low"])
        sweep_bear = _liquidity_sweep_bear(tail["high"], _swing_highs(tail["high"]))

        has_ob = ob_bear.any()
        has_fvg = fvg_bear.any()
        has_sweep = sweep_bear.any()

        if self.require_order_block and not has_ob:
            return False
        if self.require_fvg and not has_fvg:
            return False
        if self.require_liquidity_sweep and not has_sweep:
            return False

        return has_ob or has_fvg or has_sweep
