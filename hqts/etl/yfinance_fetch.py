"""
Fetch historical OHLCV from yfinance for symbols not available via MT5.

Supports BTCUSD (BTC-USD), XAUUSD (GC=F gold futures proxy), and other
yfinance tickers. Uses 15m interval; limited to last ~60 days per yfinance.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None  # type: ignore

from hqts.etl.clean import clean_and_validate

logger = logging.getLogger(__name__)

# yfinance ticker -> output symbol name
SYMBOL_MAP = {
    "BTCUSD": "BTC-USD",
    "BTCUSDm": "BTC-USD",
    "XAUUSD": "GC=F",  # Gold futures proxy for spot
    "XAGUSD": "SI=F",  # Silver futures proxy
}


def fetch_yfinance(
    symbol: str,
    interval: str = "15m",
    period: str = "2mo",
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV from yfinance.

    Args:
        symbol: Output symbol name (e.g., BTCUSD, XAUUSD) or yfinance ticker.
        interval: Bar size (1m, 5m, 15m, 1h, 1d). 15m limited to ~60 days.
        period: Valid period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y) or 2mo for 2 months.
        end: End datetime; if None, uses now.

    Returns:
        DataFrame with time, open, high, low, close, tick_volume, spread, symbol, timeframe.
    """
    if yf is None:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    ticker = SYMBOL_MAP.get(symbol.upper(), symbol)
    tf = "15m" if interval == "15m" else interval

    # yfinance period: 2mo -> 60d for intraday
    period = "60d" if period in ("2mo", "2 months") else period

    df = yf.download(
        ticker,
        interval=interval,
        period=period,
        end=end,
        progress=False,
        auto_adjust=False,
        prepost=False,
        threads=False,
    )

    if df.empty or len(df) < 10:
        raise RuntimeError(f"yfinance returned empty or insufficient data for {ticker}")

    # Reshape: yfinance returns MultiIndex (Ticker, PriceType) for single ticker too
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
    df = df.reset_index()
    # Handle index column name (Date, Datetime, etc.)
    date_cols = [c for c in df.columns if str(c).lower() in ("date", "datetime")]
    if date_cols and "time" not in df.columns:
        df = df.rename(columns={date_cols[0]: "time"})
    col_map = {
        "open": "open", "high": "high", "low": "low", "close": "close",
        "volume": "tick_volume", "vol": "tick_volume",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "time" not in df.columns and len(df.columns) > 0:
        first = df.columns[0]
        if "date" in str(first).lower() or "time" in str(first).lower():
            df = df.rename(columns={first: "time"})
    if "tick_volume" not in df.columns:
        df["tick_volume"] = 0
    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC", ambiguous="infer")
    else:
        df["time"] = df["time"].dt.tz_convert("UTC")

    df["spread"] = 0  # yfinance does not provide spread
    df["tick_volume"] = df["tick_volume"].fillna(0).astype("int64")
    df["symbol"] = symbol.upper()
    df["timeframe"] = tf

    cols = ["time", "open", "high", "low", "close", "tick_volume", "spread", "symbol", "timeframe"]
    df = df[[c for c in cols if c in df.columns]]
    logger.info("Fetched %d rows for %s from yfinance", len(df), symbol)
    return df


def _resample_to_4h(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Resample 1h OHLCV to 4h bars for a single symbol."""
    if df.empty or "time" not in df.columns:
        return pd.DataFrame()
    df = df.set_index("time").sort_index()
    resampled = df.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "tick_volume": "sum",
        "spread": "mean",
    }).dropna()
    resampled = resampled.reset_index()
    resampled["symbol"] = symbol
    resampled["timeframe"] = "4h"
    return resampled


def fetch_multi_symbol(
    symbols: list[str],
    interval: str = "15m",
    period: str = "60d",
    output_dir: str | None = "data/clean",
    use_mt5: bool = True,
) -> pd.DataFrame:
    """
    Fetch data for multiple symbols, combining MT5 and yfinance as needed.

    Args:
        symbols: List of symbols (e.g., ["BTCUSD", "XAUUSD"]).
        interval: Bar interval.
        period: Period (60d for 2 months).
        output_dir: Where to save CSVs; None to skip.
        use_mt5: If True, try MT5 first for XAUUSD.

    Returns:
        Combined DataFrame with all symbols.
    """
    all_dfs = []

    for sym in symbols:
        try:
            if use_mt5 and sym.upper() in ("XAUUSD", "XAGUSD"):
                try:
                    from hqts.etl.extract import extract_historical_data, initialize_mt5
                    import MetaTrader5 as mt5
                    if initialize_mt5():
                        # 2 months M15 ≈ 60*24*4 = 5760 bars
                        count = 5760 if interval == "15m" else 2000
                        df = extract_historical_data(symbol=sym, timeframe="M15", count=count)
                        df = clean_and_validate(df)
                        try:
                            mt5.shutdown()
                        except Exception:
                            pass
                        all_dfs.append(df)
                        continue
                except Exception as e:
                    logger.warning("MT5 fetch failed for %s: %s, falling back to yfinance", sym, e)

            df = fetch_yfinance(sym, interval=interval, period=period)
            df = clean_and_validate(df)
            all_dfs.append(df)
        except Exception as e:
            logger.error("Failed to fetch %s: %s", sym, e)

    if not all_dfs:
        raise RuntimeError("No data fetched for any symbol")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values("time").reset_index(drop=True)

    if output_dir:
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / "multi_symbol_2mo.csv"
        combined.to_csv(path, index=False)
        logger.info("Saved combined data to %s (%d rows)", path, len(combined))

    return combined


def fetch_multi_symbol_multi_timeframe(
    symbols: list[str],
    intervals: list[str] | None = None,
    period: str = "60d",
    output_dir: str | None = "data/clean",
    use_mt5: bool = False,
) -> pd.DataFrame:
    """
    Fetch data for multiple symbols and multiple timeframes (15m, 1h, 4h).

    Args:
        symbols: List of symbols (e.g., ["BTCUSD", "XAUUSD"]).
        intervals: List of intervals ["15m", "1h", "4h"]. 4h is resampled from 1h.
        period: Period (60d for 2 months).
        output_dir: Where to save CSV; None to skip.
        use_mt5: If True, try MT5 first for XAUUSD.

    Returns:
        Combined DataFrame with all symbols and timeframes.
    """
    intervals = intervals or ["15m", "1h", "4h"]
    all_dfs = []

    for interval in intervals:
        if interval == "4h":
            # Fetch 1h and resample to 4h
            df_1h = fetch_multi_symbol(
                symbols=symbols,
                interval="1h",
                period=period,
                output_dir=None,
                use_mt5=use_mt5,
            )
            for sym in symbols:
                sym_df = df_1h[df_1h["symbol"] == sym].copy()
                if sym_df.empty:
                    continue
                sym_4h = _resample_to_4h(sym_df, sym)
                if not sym_4h.empty:
                    all_dfs.append(sym_4h)
        else:
            df = fetch_multi_symbol(
                symbols=symbols,
                interval=interval,
                period=period,
                output_dir=None,
                use_mt5=use_mt5,
            )
            df["timeframe"] = interval
            all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No data fetched")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values("time").reset_index(drop=True)

    if output_dir:
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / "multi_symbol_multi_tf_2mo.csv"
        combined.to_csv(path, index=False)
        logger.info("Saved multi-timeframe data to %s (%d rows)", path, len(combined))

    return combined
