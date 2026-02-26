"""
Live data adapter for MetaTrader 5.

Fetches real-time OHLCV from MT5 for the inference pipeline.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # type: ignore

from hqts.etl.clean import clean_and_validate
from hqts.etl.extract import _get_mt5_timeframe, initialize_mt5

logger = logging.getLogger(__name__)

# Common BTC symbol names across brokers
BTC_SYMBOL_ALIASES = ("BTCUSD", "BTCUSDm", "BTCUSD.a", "BTCUSDm.a", "BTCUSD#", "BTCUSDm#")

# Symbol -> MT5 broker aliases (first match wins)
MT5_SYMBOL_ALIASES: dict[str, tuple[str, ...]] = {
    "XAUUSD": ("XAUUSD", "XAUUSDm", "GOLD", "GOLDm", "XAUUSD.a"),
    "XAGUSD": ("XAGUSD", "XAGUSDm", "SILVER", "SILVERm", "XAGUSD.a"),
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


def resolve_mt5_symbol(symbol: str) -> Optional[str]:
    """
    Find MT5 symbol for our symbol (e.g. EURUSD -> EURUSD or EURUSDm).

    Returns:
        MT5 symbol name or None if not found.
    """
    if mt5 is None:
        return None
    try:
        symbols = mt5.symbols_get()
        if symbols is None:
            return None
        candidates = [s.name for s in symbols]
        target = symbol.upper()

        if target == "BTCUSD":
            return resolve_btc_symbol()

        for name in candidates:
            if name.upper() == target:
                return name
        for name in candidates:
            if name.upper() in (target + "M", target + ".A", target + "#"):
                return name
        for alias in MT5_SYMBOL_ALIASES.get(target, ()):
            for name in candidates:
                if name.upper() == alias:
                    return name
        for name in candidates:
            name_upper = name.upper()
            if target in name_upper and len(name_upper) <= len(target) + 3:
                return name
        return None
    except Exception:
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
    # MT5 returns newest-first; sort ascending so last row = newest (matches yfinance convention)
    df = df.sort_values("time", ascending=True).reset_index(drop=True)
    return df


def get_account_info() -> dict:
    """Get account equity and balance from MT5."""
    if mt5 is None:
        return {"equity": 0.0, "balance": 0.0}
    info = mt5.account_info()
    if info is None:
        return {"equity": 0.0, "balance": 0.0}
    return {"equity": info.equity, "balance": info.balance}


def fetch_data_mt5_first(
    symbol: str,
    timeframe: str = "15m",
    count: int = 500,
) -> tuple[pd.DataFrame, str]:
    """
    Fetch OHLCV: try MT5 first (real-time broker data), fallback to yfinance.

    Args:
        symbol: Symbol (e.g. XAUUSD, EURUSD).
        timeframe: Bar size (15m, 1h).
        count: Number of bars to fetch.

    Returns:
        Tuple of (DataFrame, data_source) where data_source is "MT5" or "yfinance".
    """
    mt5_enabled = os.getenv("MT5_ENABLED", "true").lower() in ("true", "1", "yes")
    if mt5_enabled and mt5 is not None:
        try:
            if not mt5.initialize():
                raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
            mt5_sym = resolve_mt5_symbol(symbol)
            if mt5_sym:
                mt5.symbol_select(mt5_sym, True)
                mt5_tf = "M15" if timeframe == "15m" else ("H1" if timeframe == "1h" else timeframe)
                df = fetch_live_ohlcv(symbol=mt5_sym, timeframe=mt5_tf, count=count)
                df["symbol"] = symbol.upper()
                df["timeframe"] = timeframe
                logger.info("%s: fetched %d bars from MT5 (%s)", symbol, len(df), mt5_sym)
                return df, "MT5"
        except Exception as e:
            logger.debug("MT5 fetch for %s failed: %s", symbol, e)

    from hqts.etl.yfinance_fetch import fetch_yfinance

    period = os.getenv("YFINANCE_PERIOD", "60d")
    df = fetch_yfinance(symbol, interval=timeframe, period=period, force_fresh=True)
    df = clean_and_validate(df)
    logger.info("%s: fetched %d bars from yfinance (MT5 fallback)", symbol, len(df))
    return df, "yfinance"
