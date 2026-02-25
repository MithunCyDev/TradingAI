"""
Data cleaning and validation for MT5 OHLCV datasets.

Implements: duplicate removal, NaN handling (forward-fill), timezone alignment,
and basic sanity checks per SRS.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate OHLCV DataFrame per SRS requirements.

    - Drop duplicates (by timestamp)
    - Forward-fill missing close (and derived) values
    - Align timezone to UTC
    - Basic sanity checks on price/volume

    Args:
        df: Raw DataFrame with time, open, high, low, close, tick_volume, spread.

    Returns:
        Cleaned DataFrame.
    """
    if df.empty:
        return df

    df = df.copy()

    # Ensure time is datetime and UTC
    if "time" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], utc=True)
        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize("UTC", ambiguous="infer")
        else:
            df["time"] = df["time"].dt.tz_convert("UTC")

    # Drop duplicates by timestamp (keep first)
    before = len(df)
    df = df.drop_duplicates(subset=["time"], keep="first")
    if len(df) < before:
        logger.info("Dropped %d duplicate timestamps", before - len(df))

    # Sort by time
    df = df.sort_values("time").reset_index(drop=True)

    # Forward-fill missing close (and other price columns) to prevent training errors
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    for col in price_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            df[col] = df[col].ffill()
            logger.info("Forward-filled %d missing values in %s", missing, col)

    # Fill any remaining NaN in tick_volume/spread with 0 (or median for spread)
    if "tick_volume" in df.columns:
        df["tick_volume"] = df["tick_volume"].fillna(0).astype("int64")
    if "spread" in df.columns:
        df["spread"] = df["spread"].fillna(df["spread"].median()).astype("int64")

    # Sanity checks
    _validate_sanity(df)

    return df


def _validate_sanity(df: pd.DataFrame) -> None:
    """Run basic sanity assertions on price and volume."""
    if df.empty:
        return

    # OHLC consistency: high >= low, high >= open/close, low <= open/close
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        invalid_hl = (df["high"] < df["low"]).sum()
        if invalid_hl > 0:
            logger.warning("Found %d rows with high < low", invalid_hl)

        invalid_high = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
        invalid_low = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
        if invalid_high > 0 or invalid_low > 0:
            logger.warning("Found %d high/ %d low inconsistencies", invalid_high, invalid_low)

    # Non-negative volume
    if "tick_volume" in df.columns:
        neg = (df["tick_volume"] < 0).sum()
        if neg > 0:
            logger.warning("Found %d rows with negative tick_volume", neg)

    # Non-negative spread
    if "spread" in df.columns:
        neg = (df["spread"] < 0).sum()
        if neg > 0:
            logger.warning("Found %d rows with negative spread", neg)
