"""
Risk management module for HQTS.

Implements: fixed fractional position sizing (1% per trade), daily drawdown
kill switch (3%), per-symbol and total exposure caps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DailyState:
    """Tracks daily PnL for kill switch."""

    date: date
    starting_equity: float
    current_equity: float
    trades_count: int = 0

    @property
    def drawdown_pct(self) -> float:
        if self.starting_equity <= 0:
            return 0.0
        return (self.starting_equity - self.current_equity) / self.starting_equity


class RiskManager:
    """
    Manages position sizing and drawdown limits.

    - Fixed fractional: lot size = (equity * risk_pct) / (sl_distance * pip_value)
    - Kill switch: halt trading if daily drawdown exceeds limit
    """

    def __init__(
        self,
        risk_per_trade_pct: float = 0.01,
        daily_drawdown_limit_pct: float = 0.03,
        max_open_per_symbol: int = 1,
        max_total_open: int = 5,
    ) -> None:
        self.risk_per_trade_pct = risk_per_trade_pct
        self.daily_drawdown_limit_pct = daily_drawdown_limit_pct
        self.max_open_per_symbol = max_open_per_symbol
        self.max_total_open = max_total_open
        self._daily: Optional[DailyState] = None
        self._kill_switch: bool = False
        self._open_trades: dict = {}

    def reset_daily_if_new_day(self, equity: float) -> None:
        """Reset daily state when server date changes."""
        today = date.today()
        if self._daily is None or self._daily.date != today:
            self._daily = DailyState(
                date=today,
                starting_equity=equity,
                current_equity=equity,
            )
            if self._kill_switch:
                logger.info("New trading day: kill switch reset")
            self._kill_switch = False

    def update_equity(self, equity: float) -> None:
        """Update current equity and check kill switch."""
        self.reset_daily_if_new_day(equity)
        if self._daily:
            self._daily.current_equity = equity
            if self._daily.drawdown_pct >= self.daily_drawdown_limit_pct:
                self._kill_switch = True
                logger.warning(
                    "Kill switch triggered: daily drawdown %.2f%% >= limit %.2f%%",
                    self._daily.drawdown_pct * 100,
                    self.daily_drawdown_limit_pct * 100,
                )

    def is_trading_allowed(self, equity: float) -> bool:
        """Return False if kill switch is active."""
        self.update_equity(equity)
        return not self._kill_switch

    def calculate_lot_size(
        self,
        equity: float,
        sl_distance_pips: float,
        pip_value_per_lot: float,
        min_lot: float = 0.01,
        max_lot: float = 10.0,
    ) -> float:
        """
        Compute lot size for fixed fractional risk.

        risk_amount = equity * risk_per_trade_pct
        lot_size = risk_amount / (sl_distance_pips * pip_value_per_lot)

        Args:
            equity: Account equity.
            sl_distance_pips: Stop-loss distance in pips.
            pip_value_per_lot: Value of 1 pip per standard lot (e.g., $10 for XAUUSD).
            min_lot: Minimum lot size.
            max_lot: Maximum lot size.

        Returns:
            Lot size clamped to [min_lot, max_lot].
        """
        if sl_distance_pips <= 0 or pip_value_per_lot <= 0:
            return min_lot
        risk_amount = equity * self.risk_per_trade_pct
        lot = risk_amount / (sl_distance_pips * pip_value_per_lot)
        return max(min_lot, min(max_lot, round(lot, 2)))

    def can_open_trade(self, symbol: str, open_count_by_symbol: dict, total_open: int) -> bool:
        """Check if opening another trade is allowed."""
        if total_open >= self.max_total_open:
            return False
        if open_count_by_symbol.get(symbol, 0) >= self.max_open_per_symbol:
            return False
        return True
