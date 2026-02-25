#!/usr/bin/env python3
"""Generate sample OHLCV data for testing HQTS pipeline without MT5."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_sample_ohlcv(
    n_bars: int = 10_000,
    start_price: float = 1900.0,
    volatility: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2021-01-01", periods=n_bars, freq="15min", tz="UTC")
    close = start_price + np.cumsum(rng.standard_normal(n_bars) * volatility)
    high = close + np.abs(rng.standard_normal(n_bars) * volatility * 0.5)
    low = close - np.abs(rng.standard_normal(n_bars) * volatility * 0.5)
    open_ = np.roll(close, 1)
    open_[0] = start_price
    df = pd.DataFrame({
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
    return df


def main() -> None:
    out = Path("data/clean")
    out.mkdir(parents=True, exist_ok=True)
    df = generate_sample_ohlcv(5000)
    path = out / "XAUUSD_M15_sample.csv"
    df.to_csv(path, index=False)
    print(f"Generated {path} with {len(df)} rows")


if __name__ == "__main__":
    main()
