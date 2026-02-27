#!/usr/bin/env python3
"""
Sync closed trades from MT5 to local store.

Fetches history_deals from MT5, matches to trades_live.jsonl by ticket,
updates records with exit_price/pnl/outcome, and persists loss trades
to data/loss_trades.jsonl for fine-tuning.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from hqts.etl.extract import initialize_mt5
from hqts.etl.trade_history import fetch_closed_deals

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_trades(path: Path) -> list[dict]:
    """Load trades from JSONL file."""
    if not path.exists():
        return []
    trades = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                trades.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return trades


def _save_trades(path: Path, trades: list[dict]) -> None:
    """Overwrite JSONL file with trades."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")


def _outcome_from_trade(ct, sl_price: float | None, tp_price: float | None) -> str:
    """Infer outcome (tp/sl/manual) from exit price vs sl/tp."""
    if sl_price is None or tp_price is None:
        return "manual"
    exit_p = ct.exit_price
    if ct.direction == "buy":
        if exit_p <= sl_price:
            return "sl"
        if exit_p >= tp_price:
            return "tp"
    else:
        if exit_p >= sl_price:
            return "sl"
        if exit_p <= tp_price:
            return "tp"
    return "manual"


def sync_trade_outcomes(
    trades_path: Path,
    loss_trades_path: Path,
    days_back: int = 30,
) -> tuple[int, int]:
    """
    Sync MT5 closed deals to trades file and persist loss trades.

    Returns:
        (updated_count, loss_count)
    """
    if not initialize_mt5():
        logger.error("MT5 init failed")
        return 0, 0

    to_dt = datetime.now(timezone.utc)
    from_dt = to_dt - timedelta(days=days_back)
    closed = fetch_closed_deals(from_dt, to_dt)
    logger.info("Fetched %d closed trades from MT5", len(closed))

    trades = _load_trades(trades_path)
    ticket_to_idx = {t.get("ticket"): i for i, t in enumerate(trades) if t.get("ticket") not in (None, -1)}

    updated = 0
    loss_records = []

    for ct in closed:
        idx = ticket_to_idx.get(ct.order_id)
        if idx is None:
            continue
        t = trades[idx]
        if t.get("pnl") is not None:
            continue  # Already updated
        sl = t.get("sl_price")
        tp = t.get("tp_price")
        outcome = _outcome_from_trade(ct, sl, tp)
        trades[idx]["exit_price"] = ct.exit_price
        trades[idx]["pnl"] = ct.profit
        trades[idx]["outcome"] = outcome
        updated += 1

        if ct.profit < 0:
            loss_records.append({
                "symbol": ct.symbol,
                "direction": ct.direction,
                "entry_time": ct.entry_time.isoformat(),
                "entry_price": ct.entry_price,
                "exit_time": ct.exit_time.isoformat(),
                "exit_price": ct.exit_price,
                "pnl": ct.profit,
                "outcome": outcome,
                "volume": ct.volume,
                "position_id": ct.position_id,
                "order_id": ct.order_id,
            })

    if updated > 0:
        _save_trades(trades_path, trades)
        logger.info("Updated %d trades in %s", updated, trades_path)

    if loss_records:
        loss_trades_path.parent.mkdir(parents=True, exist_ok=True)
        with open(loss_trades_path, "a") as f:
            for r in loss_records:
                f.write(json.dumps(r) + "\n")
        logger.info("Persisted %d loss trades to %s", len(loss_records), loss_trades_path)

    return updated, len(loss_records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync MT5 closed trades to local store")
    parser.add_argument(
        "--trades-path",
        default=None,
        help="Path to trades JSONL (default: TRADES_LOG_PATH env or logs/trades_live.jsonl)",
    )
    parser.add_argument(
        "--loss-path",
        default="data/loss_trades.jsonl",
        help="Path for loss trades output",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of history to fetch from MT5",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    _trades = args.trades_path or os.getenv("TRADES_LOG_PATH", "logs/trades_live.jsonl")
    _loss = args.loss_path
    trades_path = Path(_trades) if Path(_trades).is_absolute() else project_root / _trades
    loss_path = Path(_loss) if Path(_loss).is_absolute() else project_root / _loss

    updated, loss_count = sync_trade_outcomes(trades_path, loss_path, days_back=args.days)
    logger.info("Done: %d trades updated, %d loss trades persisted", updated, loss_count)


if __name__ == "__main__":
    main()
