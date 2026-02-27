"""
Tests for order executor.

Validates paper-trade mode and order placement logic.
"""

import pytest

from hqts.execution.executor import OrderExecutor, OrderResult, OrderType


class TestOrderExecutor:
    """Test OrderExecutor in paper-trade mode."""

    def test_paper_trade_returns_success(self):
        """Paper-trade mode should always return success."""
        ex = OrderExecutor(symbol="XAUUSD", paper_trade=True)
        res = ex.place_market_order(
            order_type=OrderType.BUY,
            lot_size=0.1,
            sl_price=1900.0,
            tp_price=1950.0,
        )
        assert res.success is True
        assert res.ticket == -1
        assert res.message == "Paper trade"

    def test_paper_trade_sell(self):
        """Paper-trade SELL should succeed."""
        ex = OrderExecutor(symbol="XAUUSD", paper_trade=True)
        res = ex.place_market_order(
            order_type=OrderType.SELL,
            lot_size=0.05,
            sl_price=2000.0,
            tp_price=1920.0,
        )
        assert res.success is True

    def test_modify_trailing_stop_paper(self):
        """modify_trailing_stop in paper mode should return True."""
        ex = OrderExecutor(symbol="XAUUSD", paper_trade=True)
        assert ex.modify_trailing_stop(ticket=123, new_sl=1950.0) is True


class TestOrderResult:
    """Test OrderResult dataclass."""

    def test_order_result_defaults(self):
        """OrderResult should have sensible defaults."""
        r = OrderResult(success=True)
        assert r.success is True
        assert r.ticket is None
        assert r.price is None
        assert r.message == ""
