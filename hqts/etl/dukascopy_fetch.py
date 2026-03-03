"""
Fetch historical OHLCV from Dukascopy for walk-forward training.

Supports long date ranges (e.g. 2004-2026) with chunked requests.
Returns DataFrame compatible with HQTS pipeline: time, open, high, low, close, tick_volume, symbol, timeframe.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from hqts.etl.clean import clean_and_validate

logger = logging.getLogger(__name__)

# Symbols not in Dukascopy - use fallback (yfinance/MT5)
DUKASCOPY_UNSUPPORTED = {"USTECH"}

# Interval mapping: our format -> dukascopy constant
INTERVAL_MAP = {
    "1m": "1MIN",
    "5m": "5MIN",
    "10m": "10MIN",
    "15m": "15MIN",
    "30m": "30MIN",
    "1h": "1HOUR",
    "4h": "4HOUR",
    "1d": "1DAY",
    "1w": "1WEEK",
}


def _get_instrument(symbol: str) -> Optional[str]:
    """
    Map HQTS symbol to Dukascopy instrument string.

    Returns None for unsupported symbols (USTECH, etc.) or when dukascopy-python is not installed.
    """
    try:
        from dukascopy_python.instruments import (
            INSTRUMENT_CMD_ENERGY_E_LIGHT,
            INSTRUMENT_FX_MAJORS_AUD_USD,
            INSTRUMENT_FX_MAJORS_EUR_USD,
            INSTRUMENT_FX_MAJORS_GBP_USD,
            INSTRUMENT_FX_MAJORS_USD_JPY,
            INSTRUMENT_FX_METALS_XAG_USD,
            INSTRUMENT_FX_METALS_XAU_USD,
            INSTRUMENT_VCCY_BTC_USD,
        )
    except ImportError:
        return None

    mapping = {
        "EURUSD": INSTRUMENT_FX_MAJORS_EUR_USD,
        "GBPUSD": INSTRUMENT_FX_MAJORS_GBP_USD,
        "USDJPY": INSTRUMENT_FX_MAJORS_USD_JPY,
        "AUDUSD": INSTRUMENT_FX_MAJORS_AUD_USD,
        "XAUUSD": INSTRUMENT_FX_METALS_XAU_USD,
        "XAGUSD": INSTRUMENT_FX_METALS_XAG_USD,
        "BTCUSD": INSTRUMENT_VCCY_BTC_USD,
        "USOIL": INSTRUMENT_CMD_ENERGY_E_LIGHT,
    }
    return mapping.get(symbol.upper())


def _get_interval(interval: str) -> Optional[str]:
    """Map our interval string to dukascopy interval constant."""
    return INTERVAL_MAP.get(interval.lower())


def _chunk_date_range(
    start: datetime,
    end: datetime,
    chunk_months: int = 6,
) -> list[tuple[datetime, datetime]]:
    """Split date range into chunks to avoid Dukascopy limit (30k bars per request)."""
    chunks = []
    current = start
    while current < end:
        chunk_end = min(
            current + timedelta(days=chunk_months * 31),
            end,
        )
        chunks.append((current, chunk_end))
        current = chunk_end
    return chunks


def fetch_dukascopy(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    max_retries: int = 7,
) -> pd.DataFrame:
    """
    Fetch OHLCV from Dukascopy for a symbol and date range.

    Args:
        symbol: HQTS symbol (e.g. EURUSD, XAUUSD).
        interval: Bar size (15m, 1h, 4h, 1d, 1w).
        start: Start datetime (UTC).
        end: End datetime (UTC).
        max_retries: Retries per chunk on failure.

    Returns:
        DataFrame with columns: time, open, high, low, close, tick_volume, symbol, timeframe.
    """
    instrument = _get_instrument(symbol)
    if instrument is None:
        try:
            import dukascopy_python  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "dukascopy-python not installed. Run: pip install dukascopy-python"
            ) from None
        raise ValueError(
            f"Symbol {symbol} not supported by Dukascopy. "
            f"Supported: EURUSD, GBPUSD, USDJPY, AUDUSD, XAUUSD, XAGUSD, BTCUSD, USOIL. "
            f"For USTECH use yfinance/MT5 fallback."
        )

    duka_interval = _get_interval(interval)
    if duka_interval is None:
        raise ValueError(
            f"Interval {interval} not supported. Supported: {list(INTERVAL_MAP.keys())}"
        )

    try:
        import dukascopy_python
    except ImportError:
        raise RuntimeError("dukascopy-python not installed. Run: pip install dukascopy-python")

    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    chunks = _chunk_date_range(start, end)
    all_dfs = []

    for chunk_start, chunk_end in chunks:
        try:
            df = dukascopy_python.fetch(
                instrument,
                duka_interval,
                dukascopy_python.OFFER_SIDE_BID,
                start=chunk_start,
                end=chunk_end,
                max_retries=max_retries,
            )
            if df.empty:
                continue
            all_dfs.append(df)
        except Exception as e:
            logger.warning("Dukascopy fetch failed for %s %s [%s - %s]: %s", symbol, interval, chunk_start, chunk_end, e)
            raise

    if not all_dfs:
        return pd.DataFrame(
            columns=["time", "open", "high", "low", "close", "tick_volume", "symbol", "timeframe"]
        )

    combined = pd.concat(all_dfs, axis=0)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()

    out = pd.DataFrame(
        {
            "time": combined.index,
            "open": combined["open"].values,
            "high": combined["high"].values,
            "low": combined["low"].values,
            "close": combined["close"].values,
            "tick_volume": combined["volume"].values if "volume" in combined.columns else 0,
        }
    )
    out["symbol"] = symbol.upper()
    out["timeframe"] = interval
    if "spread" not in out.columns:
        out["spread"] = 0.0

    out = clean_and_validate(out)
    logger.info("Fetched %d rows for %s %s from Dukascopy [%s - %s]", len(out), symbol, interval, start.date(), end.date())
    return out


def fetch_dukascopy_multi_timeframe(
    symbol: str,
    intervals: list[str],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    Fetch data for multiple timeframes and concatenate (stacked by symbol+timeframe).

    Returns DataFrame with symbol, timeframe columns for pipeline compatibility.
    """
    dfs = []
    for interval in intervals:
        try:
            df = fetch_dukascopy(symbol, interval, start, end)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            logger.warning("Skipping %s %s: %s", symbol, interval, e)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, axis=0, ignore_index=True)
