"""Model configuration and feature column definitions."""

# Columns used as model inputs (exclude identifiers and targets)
FEATURE_COLUMNS = [
    "symbol_encoded",   # For multi-asset; omit if single symbol
    "timeframe_encoded",  # 15m=0, 1h=1, 4h=2
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
]

TARGET_COLUMN = "label"

# Class mapping: -1 -> 0 (Down), 0 -> 1 (Range), 1 -> 2 (Up) for sklearn compatibility
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {0: -1, 1: 0, 2: 1}
