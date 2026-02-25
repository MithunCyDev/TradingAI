"""
Market hours filter for HQTS.

Blocks trading when markets are closed (e.g. Forex weekend).
Uses UTC for consistency with MT5 and data feeds.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from hqts.execution.config import MarketHoursConfig

logger = logging.getLogger(__name__)


def is_market_open(
    dt: Optional[datetime] = None,
    weekend_closed: bool = True,
    friday_close_utc_hour: int = 21,
    sunday_open_utc_hour: int = 21,
    trading_start_utc_hour: Optional[int] = None,
    trading_end_utc_hour: Optional[int] = None,
) -> bool:
    """
    Return True if market is open for trading, False if closed.

    Forex/metals convention: closed Friday evening–Sunday evening (UTC).
    - Friday: close at friday_close_utc_hour (e.g. 21:00 UTC)
    - Saturday: all day closed
    - Sunday: open at sunday_open_utc_hour (e.g. 21:00 UTC)

    Args:
        dt: Time to check; if None, uses now(UTC).
        weekend_closed: If True, apply weekend closure rules.
        friday_close_utc_hour: Hour (0–23) when trading stops on Friday.
        sunday_open_utc_hour: Hour (0–23) when trading resumes on Sunday.
        trading_start_utc_hour: Optional daily start hour; None = no limit.
        trading_end_utc_hour: Optional daily end hour; None = no limit.

    Returns:
        True if trading is allowed, False if market is closed.
    """
    dt = dt or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    dow = dt.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
    hour = dt.hour

    if weekend_closed:
        if dow == 4 and hour >= friday_close_utc_hour:  # Friday evening
            return False
        if dow == 5:  # Saturday
            return False
        if dow == 6 and hour < sunday_open_utc_hour:  # Sunday before open
            return False

    if trading_start_utc_hour is not None and hour < trading_start_utc_hour:
        return False
    if trading_end_utc_hour is not None and hour >= trading_end_utc_hour:
        return False

    return True


class MarketHoursFilter:
    """
    Filter that blocks trading when market is closed.

    Integrates with ExecutionConfig.market_hours.
    Forex/metals: closed Friday 21:00 UTC–Sunday 21:00 UTC by default.
    Crypto: set weekend_closed=False for 24/7 trading.
    """

    def __init__(self, config: MarketHoursConfig) -> None:
        self.config = config

    def is_trading_allowed(self, dt: Optional[datetime] = None) -> bool:
        """Return True if trading is allowed at the given time."""
        if not self.config.enabled:
            return True
        return is_market_open(
            dt=dt,
            weekend_closed=self.config.weekend_closed,
            friday_close_utc_hour=self.config.friday_close_utc_hour,
            sunday_open_utc_hour=self.config.sunday_open_utc_hour,
            trading_start_utc_hour=self.config.trading_start_utc_hour,
            trading_end_utc_hour=self.config.trading_end_utc_hour,
        )
