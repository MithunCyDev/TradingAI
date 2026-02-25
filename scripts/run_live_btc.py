#!/usr/bin/env python3
"""
Live BTC trading via MetaTrader 5.

Connects to MT5, fetches live BTC data, runs inference, and places real trades
when the model predicts direction and filters pass.

Ensure:
- MT5 terminal is running and logged in
- pip install MetaTrader5
- BTC symbol available (BTCUSD, BTCUSDm, etc.)
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from hqts.etl.mt5_live import fetch_live_ohlcv, get_account_info, initialize_mt5, resolve_btc_symbol
from hqts.execution.config import ExecutionConfig
from hqts.execution.orchestrator import TradingOrchestrator
from hqts.logging.reporter import PredictionLogger, TradeReporter
from hqts.logging.setup import configure_logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live BTC trading via MT5")
    parser.add_argument("--paper", action="store_true", help="Paper trade (no real orders)")
    parser.add_argument("--symbol", default=None, help="MT5 symbol (default: auto-detect BTC)")
    parser.add_argument("--timeframe", default="M15", help="Bar timeframe (M15, H1, etc.)")
    parser.add_argument("--once", action="store_true", help="Run once and exit (no loop)")
    args = parser.parse_args()

    configure_logging(log_file="logs/hqts_live.log")

    if not initialize_mt5():
        logger.error(
            "Failed to connect to MT5. Ensure:\n"
            "  1. MT5 terminal is running and logged in\n"
            "  2. pip install MetaTrader5 (Python 3.9-3.11)"
        )
        sys.exit(1)

    symbol = args.symbol or resolve_btc_symbol()
    if not symbol:
        logger.error("BTC symbol not found in MT5. Check Market Watch or specify --symbol")
        sys.exit(1)

    logger.info("Using symbol: %s", symbol)

    # Enable symbol if needed
    try:
        import MetaTrader5 as mt5
        if not mt5.symbol_select(symbol, True):
            logger.warning("Could not select symbol %s; it may already be visible", symbol)
    except Exception as e:
        logger.warning("Symbol select: %s", e)

    config = ExecutionConfig()
    config.order.symbol = symbol
    config.order.timeframe = args.timeframe
    config.order.paper_trade = args.paper
    config.order.pip_value_per_lot = 1.0  # BTC: 1 lot = 1 BTC, $1 move = $1
    config.order.sl_pips_multiplier = 1.0  # BTC: ATR in price = "pips"
    config.market_hours.weekend_closed = False  # Crypto 24/7
    config.smc.require_order_block = False  # Relax for crypto
    config.data_buffer_bars = 500

    orch = TradingOrchestrator(config)
    pred_logger = PredictionLogger(log_path="logs/predictions_live.jsonl")
    trade_reporter = TradeReporter(log_path="logs/trades_live.jsonl")

    # Fetch live data
    df = fetch_live_ohlcv(symbol=symbol, timeframe=args.timeframe, count=500)
    orch.update_buffer(df)

    account = get_account_info()
    equity = account.get("equity", 0.0) or account.get("balance", 0.0)
    if equity <= 0:
        logger.warning("Account equity not available; using 10000 for sizing")
        equity = 10000.0

    logger.info("Account equity: %.2f", equity)

    signal = orch.evaluate_signal(equity)
    if signal:
        result = orch.inference.run(orch._buffer)
        pred_logger.log(symbol, result)
        success = orch.execute_signal(
            signal,
            equity,
            pip_value_per_lot=1.0,
            sl_pips_multiplier=1.0,
        )
        if success:
            close = float(orch._buffer["close"].iloc[-1])
            atr = orch._last_atr()
            rr = config.risk.rr_ratio
            sl = close - atr if signal.value == "buy" else close + atr
            tp = close + atr * rr if signal.value == "buy" else close - atr * rr
            lot = orch.risk.calculate_lot_size(equity, atr * 1.0, 1.0)
            trade_reporter.log_trade(symbol, signal.value, lot, close, sl, tp)
        logger.info("Signal: %s | prob_up=%.2f prob_down=%.2f", signal.value, result["prob_up"], result["prob_down"])
    else:
        result = orch.inference.run(orch._buffer)
        logger.info("No signal | prob_up=%.2f prob_down=%.2f prob_range=%.2f", result["prob_up"], result["prob_down"], result["prob_range"])

    try:
        import MetaTrader5 as mt5
        mt5.shutdown()
    except Exception:
        pass

    print(trade_reporter.report())


if __name__ == "__main__":
    main()
