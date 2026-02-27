"""
MT5 trade history fetcher for closed deals.

Fetches closed trades from MT5 via history_deals_get, pairs entry+exit deals
by position_id, and maps broker symbols to internal symbols.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # type: ignore

from hqts.etl.mt5_live import MT5_SYMBOL_ALIASES

logger = logging.getLogger(__name__)

# Reverse map: broker symbol -> internal symbol (e.g. USTECm -> USTECH)
_BROKER_TO_INTERNAL: dict[str, str] = {}
for internal, aliases in MT5_SYMBOL_ALIASES.items():
    for alias in aliases:
        _BROKER_TO_INTERNAL[alias.upper()] = internal
# Add common forex/indices that may not be in aliases
for sym in ("XAUUSD", "XAGUSD", "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USOIL", "BTCUSD", "USTECH"):
    _BROKER_TO_INTERNAL[sym.upper()] = sym
    _BROKER_TO_INTERNAL[sym.upper() + "M"] = sym


def _broker_to_internal(broker_symbol: str) -> str:
    """Map MT5 broker symbol to internal symbol."""
    key = broker_symbol.upper()
    return _BROKER_TO_INTERNAL.get(key, broker_symbol)


@dataclass
class ClosedTrade:
    """Single closed trade record."""

    position_id: int
    order_id: int  # Order ticket (what we store when placing)
    symbol: str  # internal symbol
    broker_symbol: str
    direction: str  # buy, sell
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    volume: float
    profit: float
    commission: float = 0.0
    swap: float = 0.0


def fetch_closed_deals(
    from_dt: datetime,
    to_dt: datetime,
    group: str = "*",
) -> list[ClosedTrade]:
    """
    Fetch closed trades from MT5 within the given time range.

    Pairs entry (type=IN) and exit (type=OUT) deals by position_id to build
    complete trade records.

    Args:
        from_dt: Start of range (UTC).
        to_dt: End of range (UTC).
        group: Symbol group filter; "*" for all.

    Returns:
        List of ClosedTrade records.
    """
    if mt5 is None:
        logger.warning("MetaTrader5 not installed; cannot fetch trade history")
        return []

    if from_dt.tzinfo is None:
        from_dt = from_dt.replace(tzinfo=timezone.utc)
    if to_dt.tzinfo is None:
        to_dt = to_dt.replace(tzinfo=timezone.utc)

    deals = mt5.history_deals_get(from_dt, to_dt, group=group)
    if deals is None:
        err = mt5.last_error()
        logger.warning("history_deals_get failed: %s", err)
        return []

    # Group by position_id: {position_id: [deals]}
    by_position: dict[int, list] = {}
    for d in deals:
        pid = getattr(d, "position_id", 0)
        if pid not in by_position:
            by_position[pid] = []
        by_position[pid].append(d)

    entry_in = getattr(mt5, "DEAL_ENTRY_IN", 0)
    entry_out = getattr(mt5, "DEAL_ENTRY_OUT", 1)

    trades: list[ClosedTrade] = []
    for position_id, deal_list in by_position.items():
        if len(deal_list) < 2:
            continue  # Need entry + exit
        entry_deal = None
        exit_deal = None
        for d in deal_list:
            entry_type = getattr(d, "entry", -1)
            if entry_type == entry_in:
                entry_deal = d
            elif entry_type == entry_out:
                exit_deal = d
        if entry_deal is None or exit_deal is None:
            continue

        entry_time = datetime.fromtimestamp(entry_deal.time, tz=timezone.utc)
        exit_time = datetime.fromtimestamp(exit_deal.time, tz=timezone.utc)
        entry_price = float(getattr(entry_deal, "price", 0))
        exit_price = float(getattr(exit_deal, "price", 0))
        volume = float(getattr(entry_deal, "volume", 0))
        profit = float(getattr(exit_deal, "profit", 0))
        commission = float(getattr(exit_deal, "commission", 0))
        swap = float(getattr(exit_deal, "swap", 0))
        broker_symbol = getattr(entry_deal, "symbol", "")
        internal_symbol = _broker_to_internal(broker_symbol)
        direction = "buy" if getattr(entry_deal, "type", 0) == getattr(mt5, "DEAL_TYPE_BUY", 0) else "sell"

        order_id = int(getattr(entry_deal, "order", 0))

        trades.append(
            ClosedTrade(
                position_id=position_id,
                order_id=order_id,
                symbol=internal_symbol,
                broker_symbol=broker_symbol,
                direction=direction,
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=exit_time,
                exit_price=exit_price,
                volume=volume,
                profit=profit,
                commission=commission,
                swap=swap,
            )
        )

    return trades
