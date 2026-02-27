"""
Tests for risk management module.

Validates lot sizing, kill switch, and trade limits.
"""

from datetime import date
from unittest.mock import patch

import pytest

from hqts.execution.risk import DailyState, RiskManager


class TestDailyState:
    """Test DailyState dataclass."""

    def test_drawdown_pct_zero_when_equal(self):
        """Drawdown should be 0 when current equals starting."""
        state = DailyState(date=date.today(), starting_equity=10000, current_equity=10000)
        assert state.drawdown_pct == 0.0

    def test_drawdown_pct_calculation(self):
        """Drawdown should be (start - current) / start."""
        state = DailyState(date=date.today(), starting_equity=10000, current_equity=9700)
        assert abs(state.drawdown_pct - 0.03) < 1e-6


class TestRiskManager:
    """Test RiskManager."""

    def test_calculate_lot_size_basic(self):
        """Lot size should follow fixed fractional formula."""
        rm = RiskManager(risk_per_trade_pct=0.01)
        # equity=10000, risk 1% = 100, sl=50 pips, pip_val=10 -> lot = 100/(50*10) = 0.2
        lot = rm.calculate_lot_size(
            equity=10000,
            sl_distance_pips=50,
            pip_value_per_lot=10,
        )
        assert 0.19 <= lot <= 0.21

    def test_calculate_lot_size_respects_min_max(self):
        """Lot size should be clamped to min_lot and max_lot."""
        rm = RiskManager(risk_per_trade_pct=0.5)  # High risk
        lot = rm.calculate_lot_size(
            equity=100000,
            sl_distance_pips=10,
            pip_value_per_lot=10,
            min_lot=0.01,
            max_lot=1.0,
        )
        assert 0.01 <= lot <= 1.0

    def test_calculate_lot_size_invalid_inputs(self):
        """Zero or negative sl/pip_value should return min_lot."""
        rm = RiskManager()
        assert rm.calculate_lot_size(10000, 0, 10) == 0.01
        assert rm.calculate_lot_size(10000, 50, 0) == 0.01

    def test_can_open_trade_respects_limits(self):
        """can_open_trade should respect per-symbol and total limits."""
        rm = RiskManager(max_open_per_symbol=1, max_total_open=2)
        assert rm.can_open_trade("XAUUSD", {}, 0) is True
        assert rm.can_open_trade("XAUUSD", {"XAUUSD": 1}, 1) is False
        assert rm.can_open_trade("EURUSD", {"XAUUSD": 1}, 2) is False

    @patch("hqts.execution.risk.date")
    def test_kill_switch_triggers_on_drawdown(self, mock_date):
        """Kill switch should trigger when daily drawdown exceeds limit."""
        mock_date.today.return_value = date(2024, 2, 6)
        rm = RiskManager(daily_drawdown_limit_pct=0.03)
        rm.reset_daily_if_new_day(10000)
        rm.update_equity(9690)  # 3.1% drawdown
        assert rm.is_trading_allowed(9690) is False
