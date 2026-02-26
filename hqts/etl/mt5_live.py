"""
Live data adapter for MetaTrader 5.

Fetches real-time OHLCV from MT5 for the inference pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # type: ignore

from hqts.etl.extract import _get_mt5_timeframe, initialize_mt5
from hqts.features.engineering import compute_features

logger = logging.getLogger(__name__)

# Common BTC symbol names across brokers
BTC_SYMBOL_ALIASES = ("BTCUSD", "BTCUSDm", "BTCUSD.a", "BTCUSDm.a", "BTCUSD#", "BTCUSDm#")

# Symbol -> MT5 broker aliases (first match wins)
MT5_SYMBOL_ALIASES: dict[str, tuple[str, ...]] = {
    "USTECH": ("US100", "NAS100", "USTEC", "USTECH", "US500", "NDX"),
    "USOIL": ("USOIL", "WTI", "XTIUSD", "CL", "USOILm", "WTI.a"),
}


def resolve_mt5_symbol_for_fetch(symbol: str) -> Optional[str]:
    """
    Find the first available MT5 symbol for historical fetch.

    Returns:
        MT5 symbol name (e.g. US100 for USTECH) or None if not found.
    """
    if mt5 is None:
        return None
    sym_upper = symbol.upper()
    symbols = mt5.symbols_get()
    if symbols is None:
        return None
    candidates = {s.name.upper(): s.name for s in symbols}

    aliases = MT5_SYMBOL_ALIASES.get(sym_upper)
    if aliases:
        for alias in aliases:
            if alias.upper() in candidates:
                return candidates[alias.upper()]

    if sym_upper in candidates:
        return candidates[sym_upper]
    for suffix in ("M", "m", ".A", "#"):
        cand = sym_upper + suffix
        if cand in candidates:
            return candidates[cand]
    return None


def resolve_btc_symbol() -> Optional[str]:
    """
    Find the first available BTC symbol in MT5.

    Returns:
        Symbol name (e.g. BTCUSD) or None if not found.
    """
    if mt5 is None:
        return None
    symbols = mt5.symbols_get()
    if symbols is None:
        return None
    for s in symbols:
        name = s.name.upper()
        if name in BTC_SYMBOL_ALIASES or (name.startswith("BTC") and "USD" in name):
            return s.name
    return None


def fetch_live_ohlcv(
    symbol: str,
    timeframe: str = "M15",
    count: int = 500,
) -> pd.DataFrame:
    """
    Fetch latest OHLCV from MT5 for live inference.

    Args:
        symbol: MT5 symbol (e.g. BTCUSD, XAUUSD).
        timeframe: Bar size (M15, H1, etc.).
        count: Number of bars to fetch.

    Returns:
        DataFrame with time, open, high, low, close, tick_volume, spread, symbol, timeframe.
    """
    if mt5 is None:
        raise RuntimeError("MetaTrader5 not installed")

    tf_val = _get_mt5_timeframe(timeframe)
    rates = mt5.copy_rates_from_pos(symbol, tf_val, 0, count)

    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        raise RuntimeError(f"MT5 copy_rates_from_pos failed: {err}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["symbol"] = symbol
    df["timeframe"] = timeframe
    cols = ["time", "open", "high", "low", "close", "tick_volume", "spread", "symbol", "timeframe"]
    df = df[[c for c in cols if c in df.columns]]
    return df


def get_account_info() -> dict:
    """Get account equity and balance from MT5."""
    if mt5 is None:
        return {"equity": 0.0, "balance": 0.0}
    info = mt5.account_info()
    if info is None:
        return {"equity": 0.0, "balance": 0.0}
    return {"equity": info.equity, "balance": info.balance}
