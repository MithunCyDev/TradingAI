"""
Shared pytest fixtures for HQTS trading system tests.

Provides sample OHLCV data and common test utilities.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))


def generate_sample_ohlcv(
    n_bars: int = 500,
    start_price: float = 1900.0,
    volatility: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=n_bars, freq="15min", tz="UTC")
    close = start_price + np.cumsum(rng.standard_normal(n_bars) * volatility)
    high = close + np.abs(rng.standard_normal(n_bars) * volatility * 0.5)
    low = close - np.abs(rng.standard_normal(n_bars) * volatility * 0.5)
    open_ = np.roll(close, 1)
    open_[0] = start_price
    return pd.DataFrame({
        "time": times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "tick_volume": rng.integers(1000, 10000, n_bars),
        "spread": rng.integers(10, 20, n_bars),
        "symbol": "XAUUSD",
        "timeframe": "M15",
    })


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Sample OHLCV DataFrame with 500 bars."""
    return generate_sample_ohlcv(n_bars=500)


@pytest.fixture
def sample_ohlcv_small() -> pd.DataFrame:
    """Small OHLCV DataFrame with 50 bars."""
    return generate_sample_ohlcv(n_bars=50)


@pytest.fixture
def sample_ohlcv_empty() -> pd.DataFrame:
    """Empty OHLCV DataFrame with correct columns."""
    return pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume"])
