#!/usr/bin/env python3
"""
Run HQTS trading bot (paper mode by default).

Demonstrates full pipeline: load data -> inference -> SMC filter -> risk check.
Use with MT5 data adapter for live trading.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from hqts.execution.config import ExecutionConfig
from hqts.execution.orchestrator import TradingOrchestrator
from hqts.logging.setup import configure_logging
from hqts.logging.reporter import PredictionLogger, TradeReporter


def main() -> None:
    configure_logging(log_file="logs/hqts.log")
    config = ExecutionConfig()
    config.smc.require_order_block = False  # Relax for demo with sample data
    orch = TradingOrchestrator(config)
    pred_logger = PredictionLogger()
    trade_reporter = TradeReporter()

    # Load sample data as buffer (replace with MT5 live data in production)
    df = pd.read_csv("data/clean/XAUUSD_M15_sample.csv")
    df["time"] = pd.to_datetime(df["time"])
    orch.update_buffer(df.tail(1000))

    # Evaluate signal
    equity = 20_000.0  # Example: 20,000 BDT
    signal = orch.evaluate_signal(equity)
    if signal:
        result = orch.inference.run(orch._buffer)
        pred_logger.log(config.order.symbol, result)
        success = orch.execute_signal(signal, equity)
        if success:
            close = float(orch._buffer["close"].iloc[-1])
            atr = orch._last_atr()
            rr = config.risk.rr_ratio
            sl = close - atr if signal.value == "buy" else close + atr
            tp = close + atr * rr if signal.value == "buy" else close - atr * rr
            lot = orch.risk.calculate_lot_size(equity, atr * 100, 10.0)
            trade_reporter.log_trade(config.order.symbol, signal.value, lot, close, sl, tp)
        print(f"Signal: {signal.value}")
    else:
        print("No signal (ML/SMC/risk filter did not pass)")

    print(trade_reporter.report())


if __name__ == "__main__":
    main()
