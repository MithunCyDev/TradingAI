"""
Order execution module for HQTS.

Encapsulates MT5 order placement with slippage control, TP/SL, and trailing stops.
Supports paper-trading mode when MT5 is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # type: ignore


class OrderType(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderResult:
    """Result of an order placement attempt."""

    success: bool
    ticket: Optional[int] = None
    price: Optional[float] = None
    message: str = ""


class OrderExecutor:
    """
    Executes orders via MT5 with slippage and risk controls.

    In paper-trade mode, logs orders without sending to MT5.
    """

    def __init__(
        self,
        symbol: str = "XAUUSD",
        max_slippage_points: int = 20,
        paper_trade: bool = True,
    ) -> None:
        self.symbol = symbol
        self.max_slippage_points = max_slippage_points
        self.paper_trade = paper_trade

    def place_market_order(
        self,
        order_type: OrderType,
        lot_size: float,
        sl_price: float,
        tp_price: float,
    ) -> OrderResult:
        """
        Place a market order with TP/SL.

        Args:
            order_type: BUY or SELL.
            lot_size: Volume in lots.
            sl_price: Stop-loss price.
            tp_price: Take-profit price.

        Returns:
            OrderResult with success status and ticket/price if filled.
        """
        if self.paper_trade:
            logger.info(
                "PAPER: %s %.2f lots %s SL=%.2f TP=%.2f",
                order_type.value,
                lot_size,
                self.symbol,
                sl_price,
                tp_price,
            )
            return OrderResult(success=True, ticket=-1, price=0.0, message="Paper trade")

        if mt5 is None:
            return OrderResult(success=False, message="MetaTrader5 not available")

        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return OrderResult(success=False, message=f"Symbol {self.symbol} not found")

        # Normalize lot to broker's volume step
        step = getattr(symbol_info, "volume_step", 0.01) or 0.01
        min_vol = getattr(symbol_info, "volume_min", 0.01) or 0.01
        max_vol = getattr(symbol_info, "volume_max", 100.0) or 100.0
        lot_size = max(min_vol, min(max_vol, round(lot_size / step) * step))

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return OrderResult(success=False, message="No tick data")

        price = tick.ask if order_type == OrderType.BUY else tick.bid
        trade_type = mt5.TRADE_ACTION_DEAL
        request = {
            "action": trade_type,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY if order_type == OrderType.BUY else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": self.max_slippage_points,
        }
        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            return OrderResult(success=False, message=str(err))
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(success=False, message=result.comment or str(result.retcode))
        return OrderResult(
            success=True,
            ticket=result.order,
            price=result.price,
            message="Filled",
        )

    def modify_trailing_stop(self, ticket: int, new_sl: float) -> bool:
        """Modify an open position's stop-loss (e.g., trail to breakeven)."""
        if self.paper_trade:
            logger.info("PAPER: Modify ticket %d SL to %.2f", ticket, new_sl)
            return True
        if mt5 is None:
            return False
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
        pos = position[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": new_sl,
            "tp": pos.tp,
        }
        result = mt5.order_send(request)
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
