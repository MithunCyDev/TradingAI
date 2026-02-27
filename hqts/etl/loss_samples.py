"""
Extract loss samples from closed losing trades for model fine-tuning.

For each losing trade, fetches OHLCV at entry time, computes features,
and assigns inverted label (BUY lost -> DOWN, SELL lost -> UP).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from hqts.features.engineering import compute_features
from hqts.features.labeling import LABEL_DOWN, LABEL_UP

logger = logging.getLogger(__name__)


def _load_loss_trades(path: Path) -> list[dict]:
    """Load loss trades from JSONL."""
    if not path.exists():
        return []
    trades = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                trades.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return trades


def _fetch_data_for_symbol(
    symbol: str,
    data_dir: Path,
    period: str = "1y",
) -> pd.DataFrame:
    """Fetch or load OHLCV data for symbol."""
    suffix = "1y" if "1y" in period.lower() else "6mo"
    raw_path = data_dir / f"{symbol.lower()}_{suffix}.csv"
    if raw_path.exists():
        df = pd.read_csv(raw_path)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        return df

    try:
        from hqts.etl.yfinance_fetch import fetch_multi_symbol_multi_timeframe
        df = fetch_multi_symbol_multi_timeframe(
            symbols=[symbol],
            intervals=["15m", "1h", "4h"],
            period=period,
            output_dir=None,
            use_mt5=True,
        )
        return df
    except Exception as e:
        logger.warning("Could not fetch data for %s: %s", symbol, e)
        return pd.DataFrame()


def extract_loss_samples(
    loss_trades_path: str | Path,
    data_dir: str | Path = "data/clean",
    context_bars: int = 100,
    period: str = "1y",
) -> pd.DataFrame:
    """
    Extract loss samples for fine-tuning.

    For each losing trade in loss_trades.jsonl:
    - Fetches OHLCV for symbol at entry_time
    - Takes context_bars before entry
    - Computes features
    - Assigns inverted label: BUY lost -> DOWN (-1), SELL lost -> UP (1)

    Args:
        loss_trades_path: Path to loss_trades.jsonl
        data_dir: Directory for raw data or fetch
        context_bars: Bars before entry for feature context
        period: Data period when fetching

    Returns:
        DataFrame with features and label column (same schema as training data).
    """
    path = Path(loss_trades_path)
    trades = _load_loss_trades(path)
    if not trades:
        logger.info("No loss trades in %s", path)
        return pd.DataFrame()

    data_dir = Path(data_dir)
    samples = []
    symbol_data_cache: dict[str, pd.DataFrame] = {}

    for t in trades:
        symbol = t.get("symbol", "").upper()
        if not symbol:
            continue
        entry_time_str = t.get("entry_time")
        direction = t.get("direction", "").lower()
        if not entry_time_str or direction not in ("buy", "sell"):
            continue

        entry_time = pd.to_datetime(entry_time_str)
        if symbol not in symbol_data_cache:
            symbol_data_cache[symbol] = _fetch_data_for_symbol(symbol, data_dir, period)

        df = symbol_data_cache[symbol]
        if df.empty or "time" not in df.columns:
            continue

        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

        # Filter to 15m for consistency with training
        if "timeframe" in df.columns:
            df = df[df["timeframe"].astype(str).str.lower().isin(["15m", "m15"])].copy()
        if df.empty:
            continue

        # Find bar closest to entry_time
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
        entry_naive = entry_time.tz_localize(None) if entry_time.tzinfo else entry_time
        idx = (df["time"] - entry_naive).abs().argmin()
        if idx < context_bars:
            continue
        window = df.iloc[idx - context_bars : idx + 1].copy().reset_index(drop=True)
        if len(window) < 50:
            continue

        featured = compute_features(window, atr_period=14, rsi_period=14)
        featured = featured.dropna(subset=["atr", "rsi"]).reset_index(drop=True)
        if featured.empty:
            continue

        last_row = featured.iloc[-1:].copy()
        last_row = last_row.reset_index(drop=True)
        last_row["symbol"] = symbol
        if direction == "buy":
            last_row["label"] = LABEL_DOWN
        else:
            last_row["label"] = LABEL_UP
        samples.append(last_row)

    if not samples:
        return pd.DataFrame()
    return pd.concat(samples, ignore_index=True)
