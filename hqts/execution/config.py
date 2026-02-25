"""
Execution configuration for HQTS.

Centralizes all tunable parameters for risk, execution, and SMC filters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RiskConfig:
    """Risk management parameters."""

    risk_per_trade_pct: float = 0.01  # 1% of equity per trade (SRS)
    daily_drawdown_limit_pct: float = 0.03  # 3% kill switch (SRS)
    max_slippage_pips: float = 2.0
    rr_ratio: float = 2.0  # 1:2 risk-reward
    trailing_stop_at_rr: float = 1.0  # Move to breakeven at 1:1
    max_open_trades_per_symbol: int = 1
    max_total_open_trades: int = 5


@dataclass
class OrderConfig:
    """Order execution parameters."""

    symbol: str = "XAUUSD"
    timeframe: str = "M15"
    lot_size: Optional[float] = None  # Override auto-sizing if set
    max_slippage_points: int = 20  # ~2 pips for 5-digit broker
    use_limit_orders: bool = True  # Place at order blocks when possible
    paper_trade: bool = True  # Sandbox mode, no real orders
    # Risk sizing: pip_value_per_lot ($ per pip per lot), sl_pips_multiplier (atr->pips)
    pip_value_per_lot: float = 10.0  # XAUUSD ~$10; BTC use 1.0 (1 lot = 1 BTC)
    sl_pips_multiplier: float = 100.0  # atr * this = pips; BTC use 1.0


@dataclass
class NewsFilterConfig:
    """Economic calendar filter parameters."""

    enabled: bool = True
    api_url: Optional[str] = None  # e.g., ForexFactory API
    window_minutes_before: int = 30
    window_minutes_after: int = 30
    high_impact_only: bool = True  # Red folder events


@dataclass
class MarketHoursConfig:
    """
    Market close / trading hours filter.

    When enabled, blocks new trades outside trading hours.
    Forex/metals: typically closed Friday evening–Sunday evening (UTC).
    Crypto (BTCUSD): 24/7; set weekend_closed=False for crypto-only.
    """

    enabled: bool = True
    weekend_closed: bool = True  # Forex/metals: no trading Fri eve–Sun eve
    friday_close_utc_hour: int = 21  # Friday: stop trading at this hour (UTC)
    sunday_open_utc_hour: int = 21  # Sunday: resume trading at this hour (UTC)
    # Optional: restrict to specific hours (e.g. London+NY overlap only)
    trading_start_utc_hour: Optional[int] = None  # None = no start limit
    trading_end_utc_hour: Optional[int] = None   # None = no end limit


@dataclass
class SMCConfig:
    """Smart Money Concepts filter parameters."""

    require_order_block: bool = True
    require_fvg: bool = False  # Alternative: FVG confirmation
    require_liquidity_sweep: bool = False
    ob_lookback_bars: int = 20
    fvg_min_size_atr: float = 0.3


@dataclass
class ExecutionConfig:
    """Full execution configuration."""

    risk: RiskConfig = field(default_factory=RiskConfig)
    order: OrderConfig = field(default_factory=OrderConfig)
    news: NewsFilterConfig = field(default_factory=NewsFilterConfig)
    market_hours: MarketHoursConfig = field(default_factory=MarketHoursConfig)
    smc: SMCConfig = field(default_factory=SMCConfig)
    model_dir: Path | str = "models"
    data_buffer_bars: int = 500  # Candles to keep for feature computation
