"""
Trade and prediction reporting for HQTS.

Persists predictions vs. outcomes for continuous learning and provides
simple summary reports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Single prediction log entry."""

    timestamp: str
    symbol: str
    label: int  # -1, 0, 1
    prob_up: float
    prob_down: float
    prob_range: float
    outcome: Optional[int] = None  # Filled when trade closes
    pnl: Optional[float] = None


@dataclass
class TradeRecord:
    """Single trade log entry."""

    timestamp: str
    symbol: str
    side: str  # buy, sell
    lot_size: float
    entry_price: float
    sl_price: float
    tp_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    outcome: Optional[str] = None  # tp, sl, manual


class PredictionLogger:
    """
    Logs model predictions to a JSONL file for future retraining.

    Each line is a JSON object with timestamp, symbol, label, probabilities.
    """

    def __init__(self, log_path: Path | str = "logs/predictions.jsonl") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, symbol: str, result: dict) -> None:
        """Append a prediction to the log file."""
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "symbol": symbol,
            "label": result.get("label", 0),
            "prob_up": result.get("prob_up", 0),
            "prob_down": result.get("prob_down", 0),
            "prob_range": result.get("prob_range", 0),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def update_outcome(self, timestamp: str, outcome: int, pnl: Optional[float] = None) -> None:
        """
        Update a prior prediction with outcome (for continuous learning).
        In a full implementation, this would update a DB row by timestamp.
        """
        logger.info("Prediction outcome: %s -> %d, pnl=%s", timestamp, outcome, pnl)


class TradeReporter:
    """
    Logs trades and generates simple summary reports.
    """

    def __init__(self, log_path: Path | str = "logs/trades.jsonl") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._trades: list[dict] = []

    def log_trade(
        self,
        symbol: str,
        side: str,
        lot_size: float,
        entry: float,
        sl: float,
        tp: float,
        ticket: Optional[int] = None,
    ) -> None:
        """Log an opened trade."""
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "symbol": symbol,
            "side": side,
            "lot_size": lot_size,
            "entry_price": entry,
            "sl_price": sl,
            "tp_price": tp,
            "ticket": ticket,
            "exit_price": None,
            "pnl": None,
            "outcome": None,
        }
        self._trades.append(record)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_close(self, ticket: int, exit_price: float, pnl: float, outcome: str) -> None:
        """Log a closed trade (update last matching trade)."""
        for t in reversed(self._trades):
            if t.get("ticket") == ticket:
                t["exit_price"] = exit_price
                t["pnl"] = pnl
                t["outcome"] = outcome
                break

    def summary(self) -> dict[str, Any]:
        """Return summary stats for logged trades."""
        closed = [t for t in self._trades if t.get("pnl") is not None]
        if not closed:
            return {"total_trades": len(self._trades), "closed_trades": 0}

        pnls = [t["pnl"] for t in closed]
        wins = [p for p in pnls if p > 0]
        return {
            "total_trades": len(self._trades),
            "closed_trades": len(closed),
            "win_count": len(wins),
            "loss_count": len(closed) - len(wins),
            "win_rate": len(wins) / len(closed) if closed else 0,
            "total_pnl": sum(pnls),
            "avg_pnl": sum(pnls) / len(closed),
        }

    def report(self) -> str:
        """Generate a human-readable report string."""
        s = self.summary()
        lines = [
            "=== HQTS Trade Report ===",
            f"Total trades: {s['total_trades']}",
            f"Closed trades: {s['closed_trades']}",
        ]
        if s["closed_trades"] > 0:
            lines.extend([
                f"Wins: {s['win_count']}, Losses: {s['loss_count']}",
                f"Win rate: {s['win_rate']:.1%}",
                f"Total PnL: {s['total_pnl']:.2f}",
                f"Avg PnL: {s['avg_pnl']:.2f}",
            ])
        return "\n".join(lines)
