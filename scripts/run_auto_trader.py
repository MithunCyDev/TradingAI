#!/usr/bin/env python3
"""
Auto-trader: runs every 3 minutes, predicts for all symbols, executes trades via MT5.

Fetches data (MT5 or yfinance), runs inference, and places real orders when
a trade signal is found (prob >= 60%). Requires MT5 terminal running on Windows.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np

from hqts.etl.economic_calendar import fetch_upcoming_events
from hqts.etl.mt5_live import fetch_data_mt5_first
from hqts.execution.config import ExecutionConfig, MarketHoursConfig
from hqts.execution.executor import OrderExecutor, OrderType
from hqts.execution.market_hours import MarketHoursFilter
from hqts.execution.risk import RiskManager
from hqts.execution.smc import SMCFilter
from hqts.logging.setup import configure_logging

# ANSI colors for terminal (Windows 10+ supports these)
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"


def _enable_windows_ansi():
    """Enable ANSI escape codes on Windows."""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to log levels in terminal."""

    LEVEL_COLORS = {
        logging.DEBUG: C.DIM,
        logging.INFO: C.GREEN,
        logging.WARNING: C.YELLOW,
        logging.ERROR: C.RED,
        logging.CRITICAL: C.RED + C.BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, C.RESET)
        record.levelname = f"{color}{record.levelname}{C.RESET}"
        return super().format(record)


def cprint(msg: str, color: str = C.RESET) -> None:
    """Print to stdout with color."""
    print(f"{color}{msg}{C.RESET}", flush=True)


def log_symbol_direction(symbol: str, direction: str, prob_up: float, prob_down: float, prob_range: float) -> None:
    """Log symbol prediction with colored direction."""
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    if direction == "up":
        color = C.GREEN
        arrow = "^"
        label = "UP"
    elif direction == "down":
        color = C.RED
        arrow = "v"
        label = "DOWN"
    else:
        color = C.YELLOW
        arrow = "-"
        label = "RANGE"
    msg = f"  {C.CYAN}{symbol}{C.RESET} {color}{arrow} {label}{C.RESET}  up={prob_up:.2f} down={prob_down:.2f} range={prob_range:.2f}"
    cprint(f"[{ts}] {msg}")


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INTERVAL_SEC = int(os.getenv("AUTO_TRADER_INTERVAL_SEC", "60"))
MODELS_BASE = Path(os.getenv("MODELS_BASE_DIR", "models"))
DATA_BUFFER_BARS = 500
RR_RATIO = 2.0
TRADE_PROB_THRESHOLD = float(os.getenv("TRADE_PROB_THRESHOLD", "0.6"))

SYMBOLS = [
    ("BTCUSD", "btcusd"),
    ("XAUUSD", "xauusd"),
    ("XAGUSD", "xagusd"),
    ("EURUSD", "eurusd"),
    ("USDJPY", "usdjpy"),
    ("GBPUSD", "gbpusd"),
    ("AUDUSD", "audusd"),
    ("USDCHF", "usdchf"),
    ("USTECH", "ustech"),
    ("USOIL", "usoil"),
]

# (pip_value_per_lot, sl_pips_multiplier) per symbol
SYMBOL_RISK_CONFIG = {
    "BTCUSD": (1.0, 1.0),
    "XAUUSD": (10.0, 100.0),
    "XAGUSD": (50.0, 100.0),
    "EURUSD": (10.0, 10000.0),
    "USDJPY": (9.0, 100.0),
    "GBPUSD": (10.0, 10000.0),
    "AUDUSD": (10.0, 10000.0),
    "USDCHF": (10.0, 10000.0),
    "USTECH": (1.0, 1.0),
    "USOIL": (10.0, 100.0),
}

# Max spread (points) per symbol - skip trade if broker spread exceeds this
# Only rejects when spread is too high; smaller spreads are always accepted
SYMBOL_MAX_SPREAD_POINTS = {
    "BTCUSD": 500,
    "XAUUSD": 50,
    "XAGUSD": 60,
    "EURUSD": 25,
    "USDJPY": 25,
    "GBPUSD": 25,
    "AUDUSD": 25,
    "USDCHF": 30,  # Often wider than other majors
    "USTECH": 50,
    "USOIL": 40,
}

# Crypto trades 24/7; forex/metals closed Fri eve–Sun eve
CRYPTO_SYMBOLS = {"BTCUSD"}


# Broker-specific aliases (e.g. GOLD for XAUUSD, SILVER for XAGUSD)
SYMBOL_MT5_ALIASES = {
    "XAUUSD": ("XAUUSD", "XAUUSDm", "GOLD", "GOLDm", "XAUUSD.a"),
    "XAGUSD": ("XAGUSD", "XAGUSDm", "SILVER", "SILVERm", "XAGUSD.a"),
    "USTECH": ("US100", "NAS100", "USTEC", "USTECH", "US500", "NDX"),
    "USOIL": ("USOIL", "WTI", "XTIUSD", "CL", "USOILm", "WTI.a"),
}


def get_mt5_spread_points(mt5_sym: str) -> int | None:
    """Get current broker spread in points for the MT5 symbol. Returns None if unavailable."""
    try:
        import MetaTrader5 as mt5
        mt5.symbol_select(mt5_sym, True)
        info = mt5.symbol_info(mt5_sym)
        if info is None:
            return None
        return getattr(info, "spread", None)
    except Exception:
        return None


def resolve_mt5_symbol(symbol: str):
    """Find MT5 symbol for our symbol (e.g. EURUSD -> EURUSD or EURUSDm)."""
    try:
        import MetaTrader5 as mt5
        from hqts.etl.mt5_live import resolve_btc_symbol

        if symbol.upper() == "BTCUSD":
            return resolve_btc_symbol()

        symbols = mt5.symbols_get()
        if symbols is None:
            return None

        target = symbol.upper()
        candidates = [s.name for s in symbols]

        # Exact match
        for name in candidates:
            if name.upper() == target:
                return name
        # Common suffix (e.g. EURUSDm)
        for name in candidates:
            if name.upper() in (target + "M", target + ".A", target + "#"):
                return name
        # Broker aliases for metals
        for alias in SYMBOL_MT5_ALIASES.get(target, ()):
            for name in candidates:
                if name.upper() == alias:
                    return name
        # Partial match for forex (e.g. EURUSD in EURUSDm)
        for name in candidates:
            name_upper = name.upper()
            if target in name_upper and len(name_upper) <= len(target) + 3:
                return name
        return None
    except Exception:
        return None


def fetch_data(symbol: str, timeframe: str = "15m"):
    """Fetch OHLCV: try MT5 first, fallback to yfinance. Uses shared logic with API."""
    df, _ = fetch_data_mt5_first(symbol, timeframe=timeframe, count=DATA_BUFFER_BARS)
    return df


def _compute_atr(high, low, close, period=14):
    """Compute ATR from arrays."""
    n = len(close)
    if n < period + 1:
        return float(close[-1] * 0.01) if n > 0 else 1.0
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.convolve(tr, np.ones(period) / period, mode="valid")
    return float(atr[-1]) if len(atr) > 0 else 1.0


def run_cycle(paper: bool) -> None:
    """Run one prediction and trade cycle for all symbols."""
    try:
        from hqts.etl.mt5_live import get_account_info

        account = get_account_info()
        equity = account.get("equity", 0.0) or account.get("balance", 0.0)
        if equity <= 0:
            equity = 10000.0
    except Exception:
        equity = 10000.0

    events = []
    try:
        from datetime import datetime, timedelta, timezone
        events = fetch_upcoming_events(
            from_dt=datetime.now(timezone.utc),
            to_dt=datetime.now(timezone.utc) + timedelta(days=1),
            high_impact_only=True,
        )
    except Exception:
        pass

    config = ExecutionConfig()
    market_hours_forex = MarketHoursFilter(config.market_hours)
    market_hours_crypto = MarketHoursFilter(
        MarketHoursConfig(weekend_closed=False, enabled=config.market_hours.enabled)
    )

    from datetime import datetime
    cprint(f"\n{C.DIM}{'-' * 50}{C.RESET}")
    cprint(f"{C.BOLD}{C.BLUE}  Predictions @ {datetime.now().strftime('%H:%M:%S')}{C.RESET}")
    processed = 0
    for symbol, model_dir_name in SYMBOLS:
        try:
            model_dir = MODELS_BASE / model_dir_name
            if not (model_dir / "model.joblib").exists():
                logger.debug("Skipping %s: model not trained", symbol)
                continue

            df = fetch_data(symbol, "15m")
            if len(df) < 100:
                logger.warning("%s: insufficient data (%d bars)", symbol, len(df))
                continue

            df = df.tail(DATA_BUFFER_BARS).reset_index(drop=True)
            processed += 1

            from hqts.models.inference import InferenceEngine

            engine = InferenceEngine(model_dir=model_dir)
            result = engine.run(
                df,
                events=events if events else None,
                zone_width_atr=config.smc.zone_width_atr,
            )

            prob_up = result["prob_up"]
            prob_down = result["prob_down"]
            prob_range = result.get("prob_range", 1.0 - prob_up - prob_down)
            last_close = float(df["close"].iloc[-1])

            direction_str = "up" if prob_up >= prob_down and prob_up > prob_range else (
                "down" if prob_down > prob_up and prob_down > prob_range else "range"
            )
            log_symbol_direction(symbol, direction_str, prob_up, prob_down, prob_range)

            meets_threshold = (
                (prob_up >= TRADE_PROB_THRESHOLD and prob_up > prob_down)
                or (prob_down >= TRADE_PROB_THRESHOLD and prob_down > prob_up)
            )

            if not meets_threshold:
                logger.debug("%s: no signal (up=%.2f down=%.2f)", symbol, prob_up, prob_down)
                continue

            direction = "buy" if prob_up >= prob_down else "sell"

            market_hours = market_hours_crypto if symbol in CRYPTO_SYMBOLS else market_hours_forex
            if not market_hours.is_trading_allowed():
                logger.debug("%s: market closed, skipping", symbol)
                continue

            smc = SMCFilter(
                require_order_block=False if symbol in CRYPTO_SYMBOLS else config.smc.require_order_block,
                require_fvg=config.smc.require_fvg,
                require_liquidity_sweep=config.smc.require_liquidity_sweep,
                ob_lookback=config.smc.ob_lookback_bars,
                fvg_min_size_atr=config.smc.fvg_min_size_atr,
                min_ob_strength=config.smc.min_ob_strength,
                zone_width_atr=config.smc.zone_width_atr,
                require_price_in_zone=config.smc.require_price_in_zone,
            )
            if direction == "buy" and not smc.validate_buy(df):
                logger.info("%s: SMC filter rejected BUY (no order block/FVG/sweep in last %d bars)", symbol, smc.ob_lookback)
                continue
            if direction == "sell" and not smc.validate_sell(df):
                logger.info("%s: SMC filter rejected SELL (no order block/FVG/sweep in last %d bars)", symbol, smc.ob_lookback)
                continue

            mt5_sym = resolve_mt5_symbol(symbol)
            if not mt5_sym:
                logger.warning("%s: MT5 symbol not found, cannot execute", symbol)
                continue

            if not paper:
                try:
                    import MetaTrader5 as mt5
                    positions = mt5.positions_get(symbol=mt5_sym)
                    if positions is not None and len(positions) >= 1:
                        logger.info("%s: already have open position (%d), skipping to avoid stacking", symbol, len(positions))
                        continue
                except Exception:
                    pass

            max_spread = int(os.getenv("MAX_SPREAD_POINTS", "0")) or SYMBOL_MAX_SPREAD_POINTS.get(symbol, 30)
            current_spread = get_mt5_spread_points(mt5_sym)
            if current_spread is not None and current_spread > max_spread:
                logger.info(
                    "%s: spread too high (%d pts > max %d), skipping until spread improves",
                    symbol, current_spread, max_spread,
                )
                continue

            pip_val, sl_mult = SYMBOL_RISK_CONFIG.get(symbol, (10.0, 100.0))
            high_arr = df["high"].values
            low_arr = df["low"].values
            close_arr = df["close"].values
            atr = _compute_atr(high_arr, low_arr, close_arr)

            sl_dist = atr * 1.0
            tp_dist = atr * RR_RATIO
            sl_pips = sl_dist * sl_mult
            risk_mgr = RiskManager(risk_per_trade_pct=0.01)
            lot = risk_mgr.calculate_lot_size(equity, sl_pips, pip_val)

            if direction == "buy":
                sl_price = last_close - sl_dist
                tp_price = last_close + tp_dist
                order_type = OrderType.BUY
            else:
                sl_price = last_close + sl_dist
                tp_price = last_close - tp_dist
                order_type = OrderType.SELL

            executor = OrderExecutor(symbol=mt5_sym, paper_trade=paper)
            res = executor.place_market_order(order_type, lot, sl_price, tp_price)

            if res.success:
                arrow = "^" if direction == "buy" else "v"
                color = C.GREEN if direction == "buy" else C.RED
                cprint(
                    f"  {C.BOLD}{color}{arrow} {direction.upper()}{C.RESET} {symbol} "
                    f"{lot:.2f} lots @ {last_close:.5f} SL={sl_price:.5f} TP={tp_price:.5f}"
                )
            else:
                logger.warning("%s order failed: %s", symbol, res.message)

        except Exception as e:
            logger.exception("Error processing %s: %s", symbol, e)

    cprint(f"{C.BOLD}{C.GREEN}[OK] Cycle complete: {processed}/{len(SYMBOLS)} symbols processed{C.RESET}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-trader: predict and trade every 3 min")
    parser.add_argument("--paper", action="store_true", help="Paper trade (no real orders)")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    _enable_windows_ansi()
    configure_logging(log_file="logs/auto_trader.log")

    # Apply colored formatter to console (file stays plain)
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and h.stream in (sys.stdout, sys.stderr):
            h.setFormatter(ColoredFormatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
            break

    try:
        from hqts.etl.extract import initialize_mt5
        if not initialize_mt5():
            logger.error("MT5 init failed. Ensure terminal is running and logged in.")
            sys.exit(1)
    except Exception as e:
        logger.error("MT5 not available: %s", e)
        sys.exit(1)

    mode = "PAPER" if args.paper else "LIVE"
    mode_color = C.YELLOW if args.paper else C.GREEN
    cprint(f"\n{C.BOLD}Auto-trader started{C.RESET} ({mode_color}{mode}{C.RESET} mode) | interval={INTERVAL_SEC}s\n")

    while True:
        try:
            run_cycle(paper=args.paper)
        except Exception as e:
            logger.exception("Cycle error: %s", e)

        if args.once:
            break
        time.sleep(INTERVAL_SEC)

    try:
        import MetaTrader5 as mt5
        mt5.shutdown()
    except Exception:
        pass

    cprint(f"\n{C.DIM}Auto-trader stopped{C.RESET}\n")


if __name__ == "__main__":
    main()
