"""
Fetch historical OHLCV from yfinance for symbols not available via MT5.

Supports BTCUSD (BTC-USD), XAUUSD (GC=F gold futures proxy), and other
yfinance tickers. Uses 15m interval; limited to last ~60 days per yfinance.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None  # type: ignore

from hqts.etl.clean import clean_and_validate

logger = logging.getLogger(__name__)

# yfinance ticker -> output symbol name (primary)
SYMBOL_MAP = {
    "BTCUSD": "BTC-USD",
    "BTCUSDm": "BTC-USD",
    "XAUUSD": "GC=F",  # Gold futures
    "XAGUSD": "SI=F",  # Silver futures
    "EURUSD": "EURUSD=X",
    "USDJPY": "JPY=X",
    "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X",
    "USTECH": "NQ=F",  # Nasdaq 100 futures
    "USOIL": "CL=F",   # WTI Crude futures
}

def fetch_yfinance(
    symbol: str,
    interval: str = "15m",
    period: str = "2mo",
    end: Optional[datetime] = None,
    force_fresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV from yfinance.

    Args:
        symbol: Output symbol name (e.g., BTCUSD, XAUUSD) or yfinance ticker.
        interval: Bar size (1m, 5m, 15m, 1h, 1d). 15m limited to ~60 days.
        period: Valid period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y) or 2mo for 2 months.
        end: End datetime; if None, yfinance uses default. Omitted when force_fresh (avoids GC=F empty data).
        force_fresh: If True, omit end param so symbols like GC=F return data reliably.

    Returns:
        DataFrame with time, open, high, low, close, tick_volume, spread, symbol, timeframe.
    """
    if yf is None:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    ticker = SYMBOL_MAP.get(symbol.upper(), symbol)
    tf = "15m" if interval == "15m" else interval

    # yfinance period mapping
    if period in ("2mo", "2 months"):
        period = "60d"
    elif period in ("365d", "1y"):
        period = "1y"
    elif period in ("6mo", "180d", "6 months"):
        period = "6mo"

    # Some symbols (e.g. GC=F) return empty when end is set; omit end for reliability
    download_kwargs = {
        "progress": False,
        "auto_adjust": False,
        "prepost": False,
        "threads": False,
    }
    if end is not None and not force_fresh:
        download_kwargs["end"] = end

    df = yf.download(ticker, interval=interval, period=period, **download_kwargs)

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


def _period_to_min_bars(period: str, interval: str) -> int:
    """Return minimum bars expected for period+interval (for fallback threshold)."""
    p = period.lower()
    if "1y" in p or "365" in p:
        if interval == "15m":
            return 25_000
        if interval == "1h":
            return 7_000
    if "6mo" in p or "180" in p:
        if interval == "15m":
            return 15_000
        if interval == "1h":
            return 4_000
    if "60d" in p or "2mo" in p:
        if interval == "15m":
            return 4_000
        if interval == "1h":
            return 1_000
    return 100


def _period_to_count(period: str, interval: str) -> int:
    """Return bar count for MT5 extract (1y M15 ≈ 35k, 6mo M15 ≈ 17k)."""
    p = period.lower()
    if "1y" in p or "365" in p:
        return 35_040 if interval == "15m" else 8_760
    if "6mo" in p or "180" in p:
        return 17_280 if interval == "15m" else 4_320
    if "60d" in p or "2mo" in p:
        return 5_760 if interval == "15m" else 2_000
    return 5_760 if interval == "15m" else 2_000


def fetch_symbol_with_fallback(
    symbol: str,
    interval: str = "15m",
    period: str = "1y",
    min_bars: int | None = None,
    use_mt5: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV: try yfinance first, fallback to MT5 if insufficient data.

    For 1y 15m, yfinance caps at ~60d; fallback to MT5 for full year.
    """
    min_bars = min_bars or _period_to_min_bars(period, interval)
    try:
        df = fetch_yfinance(symbol, interval=interval, period=period)
        df = clean_and_validate(df)
        if len(df) >= min_bars:
            return df
        logger.info(
            "%s %s: yfinance returned %d bars (need %d), falling back to MT5",
            symbol, interval, len(df), min_bars,
        )
    except Exception as e:
        logger.info("%s %s: yfinance failed (%s), trying MT5", symbol, interval, e)

    if not use_mt5:
        raise RuntimeError(f"No data for {symbol} (yfinance insufficient, MT5 disabled)")

    from hqts.etl.mt5_live import resolve_mt5_symbol_for_fetch, resolve_btc_symbol
    from hqts.etl.extract import extract_historical_data, initialize_mt5

    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise RuntimeError("MT5 not installed; cannot fallback from yfinance")

    if not initialize_mt5():
        raise RuntimeError("MT5 init failed; cannot fetch historical data")

    if symbol.upper() == "BTCUSD":
        mt5_sym = resolve_btc_symbol()
    else:
        mt5_sym = resolve_mt5_symbol_for_fetch(symbol)
    if not mt5_sym:
        mt5_sym = symbol.upper()

    try:
        tf = "M15" if interval == "15m" else ("H1" if interval == "1h" else "M15")
        count = _period_to_count(period, interval)
        df = extract_historical_data(symbol=mt5_sym, timeframe=tf, count=count)
        df = clean_and_validate(df)
        df["symbol"] = symbol.upper()
        df["timeframe"] = interval
        logger.info("Fetched %d rows for %s from MT5 (%s)", len(df), symbol, mt5_sym)
        return df
    except Exception as mt5_err:
        logger.warning("MT5 fetch failed for %s: %s; falling back to yfinance 60d", symbol, mt5_err)
        df = fetch_yfinance(symbol, interval=interval, period="60d")
        df = clean_and_validate(df)
        return df
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


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

    For period 1y/365d, uses fetch_symbol_with_fallback (yfinance first, MT5 if insufficient).
    """
    all_dfs = []
    use_fallback = use_mt5 and (
        period in ("1y", "365d", "6mo", "180d")
        or "1y" in period.lower()
        or "365" in period
        or "6mo" in period.lower()
        or "180" in period
    )

    for sym in symbols:
        try:
            if use_fallback:
                df = fetch_symbol_with_fallback(
                    symbol=sym,
                    interval=interval,
                    period=period,
                    use_mt5=use_mt5,
                )
                all_dfs.append(df)
                continue

            if use_mt5 and sym.upper() in ("XAUUSD", "XAGUSD"):
                try:
                    from hqts.etl.extract import extract_historical_data, initialize_mt5
                    import MetaTrader5 as mt5
                    if initialize_mt5():
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
        p = period.lower()
        suffix = "1y" if ("1y" in p or "365" in p) else ("6mo" if ("6mo" in p or "180" in p) else "2mo")
        path = Path(output_dir) / f"multi_symbol_{suffix}.csv"
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
