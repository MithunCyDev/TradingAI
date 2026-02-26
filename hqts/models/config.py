"""Model configuration and feature column definitions."""

# Columns used as model inputs (exclude identifiers and targets)
FEATURE_COLUMNS = [
    "symbol_encoded",
    "timeframe_encoded",
    "atr",
    "rsi",
    "vwap_dist",
    "hour",
    "dow",
    "session",
    "swing_high",
    "swing_low",
    "is_swing_high",
    "is_swing_low",
    "fvg_bullish",
    "fvg_bearish",
    "near_fvg_bull",
    "near_fvg_bear",
    "atr_pct",
    "bb_width",
    "volatility_regime",
    "ob_zone_strength",
    "dist_to_demand",
    "dist_to_supply",
    "in_demand_zone",
    "in_supply_zone",
    "is_liquidity_sweep_bull",
    "is_liquidity_sweep_bear",
    "near_liquidity_sweep_bull",
    "near_liquidity_sweep_bear",
    "is_news_window",
]

TARGET_COLUMN = "label"

# Class mapping: -1 -> 0 (Down), 0 -> 1 (Range), 1 -> 2 (Up) for sklearn compatibility
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {0: -1, 1: 0, 2: 1}
