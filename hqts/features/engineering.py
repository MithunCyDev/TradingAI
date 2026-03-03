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


def _bb_width(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Bollinger Bands width normalized by close."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    width = (upper - lower) / close.replace(0, np.nan)
    return width


def _bullish_ob_candle(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Last down candle before strong up move; returns zone size (high-low) or NaN."""
    down = close < open_
    strong_up = (close - open_).shift(-1) > (high - low).shift(-1) * 0.5
    ob = down & strong_up.shift(1)
    zone_size = (high - low).where(ob, np.nan)
    return zone_size


def _bearish_ob_candle(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Last up candle before strong down move; returns zone size or NaN."""
    up = close > open_
    strong_down = (open_ - close).shift(-1) > (high - low).shift(-1) * 0.5
    ob = up & strong_down.shift(1)
    zone_size = (high - low).where(ob, np.nan)
    return zone_size


def _liquidity_sweep_bull(low: pd.Series, close: pd.Series, swing_low: pd.Series) -> pd.Series:
    """Price swept below swing low then reversed (bullish liquidity sweep)."""
    prior_sl = swing_low.shift(1)
    swept_below = (low < prior_sl) & prior_sl.notna()
    reversed_up = close.shift(-1) > low
    return (swept_below & reversed_up.fillna(False)).astype(int)


def _liquidity_sweep_bear(high: pd.Series, close: pd.Series, swing_high: pd.Series) -> pd.Series:
    """Price swept above swing high then reversed (bearish liquidity sweep)."""
    prior_sh = swing_high.shift(1)
    swept_above = (high > prior_sh) & prior_sh.notna()
    reversed_down = close.shift(-1) < high
    return (swept_above & reversed_down.fillna(False)).astype(int)


def compute_features(
    df: pd.DataFrame,
    atr_period: int = 14,
    rsi_period: int = 14,
    swing_lookback: int = 2,
    zone_width_atr: float = 0.5,
    events: Optional[list] = None,
    news_minutes_before: int = 30,
    news_minutes_after: int = 30,
) -> pd.DataFrame:
    """
    Compute ML-ready features from OHLCV DataFrame.

    Args:
        df: DataFrame with columns: open, high, low, close, tick_volume, time.
        atr_period: ATR lookback period.
        rsi_period: RSI lookback period.
        swing_lookback: Bars left/right for swing high/low detection.
        zone_width_atr: ATR multiplier for demand/supply zone band width.
        events: Economic calendar events (from fetch_upcoming_events) for is_news_window.
        news_minutes_before: Minutes before event for news window.
        news_minutes_after: Minutes after event for news window.

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
    out["atr_pct"] = out["atr"] / close.replace(0, np.nan)
    out["bb_width"] = _bb_width(close)
    atr_pct_50 = out["atr_pct"].rolling(50, min_periods=20)
    p33 = atr_pct_50.quantile(0.33)
    p67 = atr_pct_50.quantile(0.67)
    out["volatility_regime"] = 0
    out.loc[out["atr_pct"] <= p33, "volatility_regime"] = 0
    out.loc[(out["atr_pct"] > p33) & (out["atr_pct"] <= p67), "volatility_regime"] = 1
    out.loc[out["atr_pct"] > p67, "volatility_regime"] = 2

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

    # Timeframe encoding for multi-timeframe training (1m=0, 3m=1, 5m=2, 45m=3, 1h=4, 2h=5, 4h=6, 1d=7, 1w=8)
    if "timeframe" in df.columns:
        tf_map = {
            "1m": 0, "3m": 1, "5m": 2, "45m": 3, "15m": 4,
            "1h": 5, "2h": 6, "4h": 7, "1d": 8, "1w": 9,
            "60m": 5, "m15": 4,
        }
        out["timeframe_encoded"] = df["timeframe"].astype(str).str.lower().map(
            lambda x: tf_map.get(x, 5)
        ).fillna(5).astype(int)

    # Structural SMC primitives
    out["swing_high"] = _swing_high(high, swing_lookback, swing_lookback)
    out["swing_low"] = _swing_low(low, swing_lookback, swing_lookback)
    out["is_swing_high"] = _is_swing_high(high, swing_lookback, swing_lookback).astype(int)
    out["is_swing_low"] = _is_swing_low(low, swing_lookback, swing_lookback).astype(int)
    out["fvg_bullish"] = _fvg_bullish(high, low)
    out["fvg_bearish"] = _fvg_bearish(high, low)
    out["near_fvg_bull"] = (low <= out["fvg_bullish"].shift(1) + out["atr"] * 0.1).astype(int)
    out["near_fvg_bear"] = (high >= out["fvg_bearish"].shift(1) - out["atr"] * 0.1).astype(int)

    # Supply/demand zone features
    ob_bull_size = _bullish_ob_candle(df["open"], high, low, close)
    ob_bear_size = _bearish_ob_candle(df["open"], high, low, close)
    out["ob_zone_strength"] = ob_bull_size.fillna(ob_bear_size).fillna(0) / out["atr"].replace(0, np.nan)
    out["ob_zone_strength"] = out["ob_zone_strength"].fillna(0)

    atr_safe = out["atr"].replace(0, np.nan)
    out["dist_to_demand"] = (close - out["swing_low"]) / atr_safe
    out["dist_to_supply"] = (out["swing_high"] - close) / atr_safe
    out["dist_to_demand"] = out["dist_to_demand"].fillna(0)
    out["dist_to_supply"] = out["dist_to_supply"].fillna(0)

    out["in_demand_zone"] = ((close - out["swing_low"]).abs() <= zone_width_atr * out["atr"]).astype(int)
    out["in_supply_zone"] = ((out["swing_high"] - close).abs() <= zone_width_atr * out["atr"]).astype(int)

    # Pullback features: price retraced into zone (came from above into demand, or from below into supply)
    out["pullback_bull"] = (
        (out["in_demand_zone"] == 1) & (close < close.shift(2))
    ).fillna(False).astype(int)
    out["pullback_bear"] = (
        (out["in_supply_zone"] == 1) & (close > close.shift(2))
    ).fillna(False).astype(int)

    # Liquidity sweep features
    out["is_liquidity_sweep_bull"] = _liquidity_sweep_bull(low, close, out["swing_low"])
    out["is_liquidity_sweep_bear"] = _liquidity_sweep_bear(high, close, out["swing_high"])
    out["near_liquidity_sweep_bull"] = out["is_liquidity_sweep_bull"].rolling(5, min_periods=1).max().astype(int)
    out["near_liquidity_sweep_bear"] = out["is_liquidity_sweep_bear"].rolling(5, min_periods=1).max().astype(int)

    # Supply/demand zone features
    from hqts.features.supply_demand import compute_supply_demand_features
    compute_supply_demand_features(out, zone_width_atr=zone_width_atr, freshness_bars=20)

    # Candlestick patterns (TA-Lib); -100/0/+100 -> -1/0/1 for ML
    try:
        import talib
        o, h, l_, c = df["open"].values, high.values, low.values, close.values
        patterns = [
            ("is_doji", talib.CDLDOJI),
            ("is_hammer", talib.CDLHAMMER),
            ("is_engulfing", talib.CDLENGULFING),
            ("is_harami", talib.CDLHARAMI),
            ("is_morning_star", talib.CDLMORNINGSTAR),
            ("is_evening_star", talib.CDLEVENINGSTAR),
            ("is_shooting_star", talib.CDLSHOOTINGSTAR),
            ("is_3_white_soldiers", talib.CDL3WHITESOLDIERS),
            ("is_3_black_crows", talib.CDL3BLACKCROWS),
        ]
        for name, func in patterns:
            arr = func(o, h, l_, c)
            out[name] = np.where(np.isnan(arr), 0, np.sign(arr)).astype(int)
        pat_cols = [p[0] for p in patterns]
        out["has_bullish_pattern"] = (out[pat_cols].rolling(5, min_periods=1).max().max(axis=1) > 0).astype(int)
        out["has_bearish_pattern"] = (out[pat_cols].rolling(5, min_periods=1).min().min(axis=1) < 0).astype(int)
    except Exception as e:
        logger.warning(
            "TA-Lib not available; candlestick pattern features set to 0. Error: %s. "
            "If using conda: conda install -c conda-forge ta-lib (in the same env as your script). "
            "Verify with: python -c \"import talib; print(talib.__file__)\"",
            e,
        )
        for name in [
            "is_doji", "is_hammer", "is_engulfing", "is_harami",
            "is_morning_star", "is_evening_star", "is_shooting_star",
            "is_3_white_soldiers", "is_3_black_crows",
            "has_bullish_pattern", "has_bearish_pattern",
        ]:
            out[name] = 0

    # News window feature
    if events and "time" in df.columns:
        try:
            from hqts.etl.economic_calendar import is_in_news_window

            def _check_news(t):
                dt = pd.Timestamp(t).to_pydatetime() if hasattr(t, "to_pydatetime") else t
                return 1 if is_in_news_window(dt, events, news_minutes_before, news_minutes_after) else 0

            out["is_news_window"] = df["time"].apply(_check_news)
        except Exception as ex:
            logger.debug("is_news_window computation failed: %s", ex)
            out["is_news_window"] = 0
    else:
        out["is_news_window"] = 0

    return out
