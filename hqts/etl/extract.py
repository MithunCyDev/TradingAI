"""
MT5 historical data extraction pipeline.

Extracts OHLCV + spread from MetaTrader 5, transforms to DataFrame,
and persists to CSV/SQLite for ML training.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # type: ignore

from hqts.etl.clean import clean_and_validate

logger = logging.getLogger(__name__)

# Default extraction parameters (SRS: ~100k M15 candles ≈ 3–4 years)
DEFAULT_SYMBOL = "XAUUSD"
DEFAULT_TIMEFRAME = "M15"
DEFAULT_COUNT = 100_000
MAX_RETRIES = 5
RETRY_DELAY_SEC = 2

# MT5 timeframe mapping (fallback when mt5 not imported)
TIMEFRAME_MAP = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 16385,
    "H4": 16388,
    "D1": 16408,
    "W1": 32769,
    "MN1": 49153,
}


def _get_mt5_timeframe(tf: str) -> int:
    """Resolve timeframe string to MT5 constant."""
    if mt5 is not None and hasattr(mt5, f"TIMEFRAME_{tf}"):
        return getattr(mt5, f"TIMEFRAME_{tf}")
    if tf in TIMEFRAME_MAP:
        return TIMEFRAME_MAP[tf]
    raise ValueError(f"Unknown timeframe: {tf}. Use one of {list(TIMEFRAME_MAP.keys())}")


def initialize_mt5(path: Optional[str] = None) -> bool:
    """
    Initialize MT5 terminal with retry logic.

    Returns:
        True if initialization succeeded, False otherwise.
    """
    if mt5 is None:
        logger.error("MetaTrader5 package not installed. Run: pip install MetaTrader5")
        return False

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = {} if path is None else {"path": path}
            if mt5.initialize(**kwargs):
                logger.info("MT5 initialized successfully")
                return True
            err = mt5.last_error()
            logger.warning("MT5 init attempt %d failed: %s", attempt, err)
        except Exception as e:
            logger.warning("MT5 init attempt %d raised: %s", attempt, e)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SEC)

    logger.error("MT5 initialization failed after %d attempts", MAX_RETRIES)
    return False


def extract_historical_data(
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    count: int = DEFAULT_COUNT,
    start_pos: int = 0,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from MT5.

    Args:
        symbol: Instrument (e.g., XAUUSD, XAGUSD).
        timeframe: Bar size (M15, H1, etc.).
        count: Number of bars to fetch.
        start_pos: Starting bar index (0 = current bar).

    Returns:
        DataFrame with columns: time, open, high, low, close, tick_volume, spread.
    """
    if mt5 is None:
        raise RuntimeError("MetaTrader5 not installed")

    if not mt5.symbol_select(symbol, True):
        logger.warning("symbol_select(%s) returned False; symbol may not be in Market Watch", symbol)

    tf_val = _get_mt5_timeframe(timeframe)
    rates = mt5.copy_rates_from_pos(symbol, tf_val, start_pos, count)

    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        raise RuntimeError(f"MT5 copy_rates_from_pos failed for {symbol}: {err}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"time": "time"})

    # Select schema columns (SRS); drop real_volume if present for consistency
    cols = ["time", "open", "high", "low", "close", "tick_volume", "spread"]
    df = df[[c for c in cols if c in df.columns]]

    df["symbol"] = symbol
    df["timeframe"] = timeframe
    logger.info("Extracted %d rows for %s %s", len(df), symbol, timeframe)
    return df


def run_extraction_pipeline(
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    count: int = DEFAULT_COUNT,
    output_dir: Path | str = "data",
    save_csv: bool = True,
    save_sqlite: bool = True,
    mt5_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Full ETL: connect to MT5, extract, clean, validate, and persist.

    Args:
        symbol: Instrument symbol.
        timeframe: Bar timeframe.
        count: Number of bars.
        output_dir: Base directory for data/raw and data/clean.
        save_csv: Whether to save cleaned CSV.
        save_sqlite: Whether to save to SQLite.
        mt5_path: Optional path to MT5 terminal executable.

    Returns:
        Cleaned DataFrame.
    """
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    clean_dir = output_dir / "clean"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    if not initialize_mt5(mt5_path):
        raise RuntimeError("Cannot connect to MT5. Ensure the terminal is running.")

    try:
        df = extract_historical_data(symbol=symbol, timeframe=timeframe, count=count)
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass

    df = clean_and_validate(df)

    base_name = f"{symbol}_{timeframe}"
    if save_csv:
        csv_path = clean_dir / f"{base_name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved CSV: %s", csv_path)

    if save_sqlite:
        db_path = clean_dir / "hqts_data.db"
        import sqlite3

        conn = sqlite3.connect(db_path)
        table = f"ohlcv_{symbol}_{timeframe}".replace(".", "_")
        df.to_sql(table, conn, if_exists="replace", index=False)
        conn.close()
        logger.info("Saved SQLite table %s: %s", table, db_path)

    return df


def main() -> None:
    """CLI entry point for extraction pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="HQTS MT5 historical data extraction")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Instrument (e.g., XAUUSD)")
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME, help="Bar size (M15, H1, etc.)")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of bars")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV output")
    parser.add_argument("--no-sqlite", action="store_true", help="Skip SQLite output")
    parser.add_argument("--mt5-path", default=None, help="Path to MT5 terminal executable")
    args = parser.parse_args()

    run_extraction_pipeline(
        symbol=args.symbol,
        timeframe=args.timeframe,
        count=args.count,
        output_dir=args.output_dir,
        save_csv=not args.no_csv,
        save_sqlite=not args.no_sqlite,
        mt5_path=args.mt5_path,
    )


if __name__ == "__main__":
    main()
