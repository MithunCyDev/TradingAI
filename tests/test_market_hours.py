"""
Tests for market hours filter.

Validates weekend closure and trading window logic.
"""

from datetime import datetime, timezone

import pytest

from hqts.execution.config import MarketHoursConfig
from hqts.execution.market_hours import MarketHoursFilter, is_market_open


class TestIsMarketOpen:
    """Test is_market_open function."""

    def test_weekday_midday_open(self):
        """Tuesday midday UTC should be open."""
        dt = datetime(2024, 2, 6, 12, 0, 0, tzinfo=timezone.utc)  # Tuesday
        assert is_market_open(dt=dt, weekend_closed=True) is True

    def test_saturday_closed(self):
        """Saturday should be closed when weekend_closed=True."""
        dt = datetime(2024, 2, 10, 12, 0, 0, tzinfo=timezone.utc)  # Saturday
        assert is_market_open(dt=dt, weekend_closed=True) is False

    def test_friday_evening_after_close(self):
        """Friday after friday_close_utc_hour should be closed."""
        dt = datetime(2024, 2, 9, 22, 0, 0, tzinfo=timezone.utc)  # Fri 22:00 UTC
        assert is_market_open(
            dt=dt,
            weekend_closed=True,
            friday_close_utc_hour=21,
        ) is False

    def test_sunday_before_open_closed(self):
        """Sunday before sunday_open_utc_hour should be closed."""
        dt = datetime(2024, 2, 11, 20, 0, 0, tzinfo=timezone.utc)  # Sun 20:00 UTC
        assert is_market_open(
            dt=dt,
            weekend_closed=True,
            sunday_open_utc_hour=21,
        ) is False

    def test_sunday_after_open(self):
        """Sunday after sunday_open_utc_hour should be open."""
        dt = datetime(2024, 2, 11, 22, 0, 0, tzinfo=timezone.utc)  # Sun 22:00 UTC
        assert is_market_open(
            dt=dt,
            weekend_closed=True,
            sunday_open_utc_hour=21,
        ) is True

    def test_weekend_closed_false_always_open(self):
        """When weekend_closed=False, Saturday should be open."""
        dt = datetime(2024, 2, 10, 12, 0, 0, tzinfo=timezone.utc)
        assert is_market_open(dt=dt, weekend_closed=False) is True


class TestMarketHoursFilter:
    """Test MarketHoursFilter class."""

    def test_disabled_filter_allows_all(self):
        """When enabled=False, is_trading_allowed should always return True."""
        config = MarketHoursConfig(enabled=False, weekend_closed=True)
        f = MarketHoursFilter(config)
        sat = datetime(2024, 2, 10, 12, 0, 0, tzinfo=timezone.utc)
        assert f.is_trading_allowed(dt=sat) is True

    def test_crypto_style_24_7(self):
        """Crypto config (weekend_closed=False) allows weekend trading."""
        config = MarketHoursConfig(enabled=True, weekend_closed=False)
        f = MarketHoursFilter(config)
        sat = datetime(2024, 2, 10, 12, 0, 0, tzinfo=timezone.utc)
        assert f.is_trading_allowed(dt=sat) is True
