#!/usr/bin/env python3
"""
Historical backtest for HQTS models.

Evaluates model signals on past bars, simulates TP/SL outcomes, and writes
results per symbol to logs/backtest_results/. Uses model-only by default;
set BACKTEST_USE_SMC=true in .env for model + SMC filter.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import pandas as pd

from hqts.etl.clean import clean_and_validate
from hqts.etl.mt5_live import fetch_data_mt5_first
from hqts.execution.config import ExecutionConfig, MarketHoursConfig
from hqts.execution.market_hours import MarketHoursFilter
from hqts.execution.smc import SMCFilter
from hqts.models.inference import InferenceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Constants ---

SYMBOLS = [
    ("BTCUSD", "btcusd"),
    ("XAUUSD", "xauusd"),
    ("XAGUSD", "xagusd"),
    ("EURUSD", "eurusd"),
    ("USDJPY", "usdjpy"),
    ("GBPUSD", "gbpusd"),
    ("AUDUSD", "audusd"),
    ("USTECH", "ustech"),
    ("USOIL", "usoil"),
]

CRYPTO_SYMBOLS = frozenset({"BTCUSD"})
DATA_BUFFER_BARS = 500
HORIZON_BARS = 16
RR_RATIO = 2.5
ATR_PERIOD = 14
FETCH_BAR_COUNT = 6_000


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return str(val).lower() in ("true", "1", "yes")


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


# --- Configuration ---


@dataclass
class BacktestConfig:
    """Backtest parameters from environment and overrides."""

    models_dir: Path = field(default_factory=lambda: Path("models"))
    results_dir: Path = field(default_factory=lambda: Path("logs/backtest_results"))
    timeframe: str = "15m"
    period_days: int = field(default_factory=lambda: _env_int("BACKTEST_PERIOD_DAYS", 60))
  
    prob_threshold: float = field(default_factory=lambda: _env_float("BACKTEST_PROB_THRESHOLD", 0.005))
    use_smc: bool = field(default_factory=lambda: _env_bool("BACKTEST_USE_SMC", False))
    use_market_hours: bool = field(default_factory=lambda: _env_bool("BACKTEST_USE_MARKET_HOURS", False))
    rr_ratio: float = RR_RATIO
    horizon_bars: int = HORIZON_BARS
    data_buffer_bars: int = DATA_BUFFER_BARS

    def __post_init__(self) -> None:
        proj = Path(__file__).resolve().parent.parent
        models_base = os.getenv("MODELS_BASE_DIR", "models")
        self.models_dir = proj / models_base if not Path(models_base).is_absolute() else Path(models_base)
        self.results_dir = proj / self.results_dir if not self.results_dir.is_absolute() else self.results_dir


# --- Data ---


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = ATR_PERIOD) -> np.ndarray:
    """Compute ATR array."""
    n = len(close)
    tr = np.maximum(high - low, np.maximum(
        np.abs(high - np.roll(close, 1)),
        np.abs(low - np.roll(close, 1)),
    ))
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan, dtype=float)
    for i in range(period - 1, n):
        atr[i] = np.mean(tr[i - period + 1 : i + 1])
    return atr


def fetch_ohlcv(symbol: str, timeframe: str, count: int) -> pd.DataFrame | None:
    """Fetch OHLCV from MT5 or yfinance. Returns None on failure."""
    try:
        df, _ = fetch_data_mt5_first(symbol, timeframe=timeframe, count=count)
    except Exception as e:
        logger.warning("%s: MT5 failed (%s), trying yfinance", symbol, e)
        try:
            from hqts.etl.yfinance_fetch import fetch_yfinance
            df = fetch_yfinance(symbol, interval=timeframe, period="60d")
            df = clean_and_validate(df)
        except Exception as yf_err:
            logger.error("%s: fetch failed: %s", symbol, yf_err)
            return None

    if df.empty or len(df) < DATA_BUFFER_BARS + HORIZON_BARS:
        logger.warning("%s: insufficient data (%d bars)", symbol, len(df) if not df.empty else 0)
        return None

    return df


def _ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure time column has UTC timezone."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC", ambiguous="infer")
    else:
        df["time"] = df["time"].dt.tz_convert("UTC")
    return df


# --- TP/SL Simulation ---


def simulate_tp_sl(
    df: pd.DataFrame,
    entry_idx: int,
    direction: str,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    horizon: int,
) -> tuple[str, float]:
    """
    Simulate TP/SL hit using future bars.
    Returns (outcome, pnl) where outcome is "tp", "sl", or "range".
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    for j in range(entry_idx + 1, min(entry_idx + horizon + 1, len(df))):
        bar_high, bar_low, bar_close = high[j], low[j], close[j]

        if direction == "buy":
            hit_tp = bar_high >= tp_price
            hit_sl = bar_low <= sl_price
        else:
            hit_tp = bar_low <= tp_price
            hit_sl = bar_high >= sl_price

        if hit_tp and hit_sl:
            tp_first = (bar_close >= entry_price) if direction == "buy" else (bar_close <= entry_price)
            if tp_first:
                pnl = (tp_price - entry_price) if direction == "buy" else (entry_price - tp_price)
                return ("tp", pnl)
            pnl = (sl_price - entry_price) if direction == "buy" else (entry_price - sl_price)
            return ("sl", pnl)

        if hit_tp:
            pnl = (tp_price - entry_price) if direction == "buy" else (entry_price - tp_price)
            return ("tp", pnl)
        if hit_sl:
            pnl = (sl_price - entry_price) if direction == "buy" else (entry_price - sl_price)
            return ("sl", pnl)

    return ("range", 0.0)


# --- Diagnostic ---


def _diagnose_no_trades(
    df: pd.DataFrame,
    engine: InferenceEngine,
    config: BacktestConfig,
    exec_config: ExecutionConfig,
    symbol: str,
    buf_size: int,
    n: int,
    horizon: int,
    window_start: datetime,
    window_end: datetime,
    market_hours: MarketHoursFilter,
) -> dict:
    """
    Sample bars in the backtest window, run inference, and collect prob stats
    to explain why no trades were taken.
    """
    prob_up_list: list[float] = []
    prob_down_list: list[float] = []
    prob_range_list: list[float] = []
    sample_size = 0
    max_samples = min(500, max(1, n - buf_size - horizon))

    for i in range(buf_size, n - horizon, max(1, (n - buf_size - horizon) // max_samples)):
        bar_time = df["time"].iloc[i]
        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=timezone.utc)
        if bar_time < window_start or bar_time > window_end:
            continue
        if config.use_market_hours and not market_hours.is_trading_allowed(bar_time):
            continue

        buffer = df.iloc[i - buf_size : i].copy()
        buffer["symbol"] = symbol
        buffer["timeframe"] = config.timeframe
        try:
            result = engine.run(buffer, zone_width_atr=exec_config.smc.zone_width_atr)
        except Exception:
            continue
        prob_up_list.append(result["prob_up"])
        prob_down_list.append(result["prob_down"])
        prob_range_list.append(result.get("prob_range", 1.0 - result["prob_up"] - result["prob_down"]))
        sample_size += 1
        if sample_size >= max_samples:
            break

    th = float(config.prob_threshold)
    pu = np.array(prob_up_list) if prob_up_list else np.array([])
    pd_arr = np.array(prob_down_list) if prob_down_list else np.array([])

    def count_above(arr: np.ndarray, t: float) -> int:
        return int(np.sum(arr >= t)) if arr.size else 0

    pu_max = float(np.max(pu)) if pu.size else 0.0
    pd_max = float(np.max(pd_arr)) if pd_arr.size else 0.0
    return {
        "sample_bars": sample_size,
        "prob_up_mean": float(np.mean(pu)) if pu.size else 0.0,
        "prob_up_max": pu_max,
        "prob_down_mean": float(np.mean(pd_arr)) if pd_arr.size else 0.0,
        "prob_down_max": pd_max,
        "above_threshold_up": count_above(pu, th),
        "above_threshold_down": count_above(pd_arr, th),
        "threshold": th,
        "suggestion": f" prob_up max={pu_max:.3f}, prob_down max={pd_max:.3f}. Use --threshold {max(0.001, min(pu_max, pd_max) * 0.9):.3f} to get trades.",
    }


# --- Backtest Engine ---


def run_backtest(
    symbol: str,
    model_dir_name: str,
    config: BacktestConfig,
    exec_config: ExecutionConfig,
) -> dict:
    """
    Run backtest for one symbol.

    Returns metrics dict with keys: symbol, period, total_trades, win_count,
    loss_count, range_count, win_rate, total_pnl, avg_pnl, trades, use_smc.
    """
    model_dir = config.models_dir / model_dir_name
    if not (model_dir / "model.joblib").exists():
        return {"error": "model not found", "symbol": symbol, "model_dir": str(model_dir)}

    df = fetch_ohlcv(symbol, config.timeframe, FETCH_BAR_COUNT)
    if df is None:
        return {"error": "fetch failed", "symbol": symbol}

    df = _ensure_utc(df.reset_index(drop=True))
    close = df["close"].values
    n = len(df)

    window_end = df["time"].iloc[-1]
    window_start = window_end - timedelta(days=config.period_days)
    if window_start.tzinfo is None:
        window_start = window_start.replace(tzinfo=timezone.utc)

    high = df["high"].values
    low = df["low"].values
    atr_arr = _compute_atr(high, low, close)

    market_hours = MarketHoursFilter(
        MarketHoursConfig(weekend_closed=False, enabled=True)
    ) if symbol in CRYPTO_SYMBOLS else MarketHoursFilter(exec_config.market_hours)

    smc: SMCFilter | None = None
    if config.use_smc:
        smc = SMCFilter(
            require_order_block=exec_config.smc.require_order_block,
            require_fvg=exec_config.smc.require_fvg,
            require_liquidity_sweep=exec_config.smc.require_liquidity_sweep,
            require_any=exec_config.smc.require_any,
            ob_lookback=exec_config.smc.ob_lookback_bars,
            fvg_min_size_atr=exec_config.smc.fvg_min_size_atr,
            min_ob_strength=exec_config.smc.min_ob_strength,
            zone_width_atr=exec_config.smc.zone_width_atr,
            require_price_in_zone=exec_config.smc.require_price_in_zone,
        )

    engine = InferenceEngine(model_dir=model_dir)
    threshold = float(config.prob_threshold)
    buf_size = config.data_buffer_bars
    horizon = config.horizon_bars

    trades: list[dict] = []
    next_eligible_bar = buf_size

    for i in range(buf_size, n - horizon):
        bar_time = df["time"].iloc[i]
        if bar_time.tzinfo is None:
            bar_time = bar_time.tz_localize("UTC")

        if bar_time < window_start or bar_time > window_end:
            continue
        if i < next_eligible_bar:
            continue
        if config.use_market_hours and not market_hours.is_trading_allowed(bar_time):
            continue

        buffer = df.iloc[i - buf_size : i].copy()
        # Ensure symbol/timeframe for feature encoding (model expects these)
        buffer["symbol"] = symbol
        buffer["timeframe"] = config.timeframe

        try:
            result = engine.run(buffer, zone_width_atr=exec_config.smc.zone_width_atr)
        except Exception as e:
            logger.debug("%s bar %d: inference error: %s", symbol, i, e)
            continue

        prob_up = result["prob_up"]
        prob_down = result["prob_down"]
        prob_range = result.get("prob_range", 1.0 - prob_up - prob_down)

        if prob_up >= threshold and prob_up > prob_down:
            direction = "buy"
        elif prob_down >= threshold and prob_down > prob_up:
            direction = "sell"
        else:
            continue

        if smc is not None:
            if direction == "buy" and not smc.validate_buy(buffer):
                continue
            if direction == "sell" and not smc.validate_sell(buffer):
                continue

        entry_price = float(close[i])
        atr_val = float(atr_arr[i]) if not np.isnan(atr_arr[i]) and atr_arr[i] > 0 else entry_price * 0.01
        sl_dist = atr_val * 1.0
        tp_dist = atr_val * config.rr_ratio

        if direction == "buy":
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        outcome, pnl = simulate_tp_sl(df, i, direction, entry_price, sl_price, tp_price, horizon)

        trades.append({
            "time": str(bar_time),
            "direction": direction,
            "entry": entry_price,
            "sl": sl_price,
            "tp": tp_price,
            "outcome": outcome,
            "pnl": pnl,
        })
        next_eligible_bar = i + horizon + 1

    if not trades:
        diag = _diagnose_no_trades(
            df, engine, config, exec_config, symbol, buf_size, n, horizon,
            window_start, window_end, market_hours,
        )
        return {
            "symbol": symbol,
            "period": window_start.strftime("%Y-%m-%d") + " to " + window_end.strftime("%Y-%m-%d"),
            "total_trades": 0,
            "win_count": 0,
            "loss_count": 0,
            "range_count": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "trades": [],
            "use_smc": config.use_smc,
            "diagnostic": diag,
        }

    wins = [t for t in trades if t["outcome"] == "tp"]
    losses = [t for t in trades if t["outcome"] == "sl"]
    ranged = [t for t in trades if t["outcome"] == "range"]
    closed = wins + losses
    win_rate = len(wins) / len(closed) if closed else 0.0
    total_pnl = sum(t["pnl"] for t in trades)
    avg_pnl = total_pnl / len(trades) if trades else 0.0

    period_str = window_start.strftime("%Y-%m-%d") + " to " + window_end.strftime("%Y-%m-%d")

    return {
        "symbol": symbol,
        "period": period_str,
        "total_trades": len(trades),
        "win_count": len(wins),
        "loss_count": len(losses),
        "range_count": len(ranged),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "trades": trades,
        "use_smc": config.use_smc,
    }


# --- Report ---


def format_report(metrics: dict) -> str:
    """Format backtest metrics as human-readable text."""
    if metrics.get("error"):
        return f"Error: {metrics['error']}\nSymbol: {metrics.get('symbol', '?')}\n"

    smc_note = " (model + SMC)" if metrics.get("use_smc") else " (model only)"
    lines = [
        "=" * 60,
        f"Backtest: {metrics['symbol']}{smc_note}",
        f"Period:  {metrics['period']}",
        "=" * 60,
        "",
        f"Total trades:   {metrics['total_trades']}",
        f"Wins (TP):      {metrics['win_count']}",
        f"Losses (SL):    {metrics['loss_count']}",
        f"Range (no hit): {metrics.get('range_count', 0)}",
        "",
        f"Win rate:       {metrics['win_rate']:.1%}" if metrics["total_trades"] > 0 else "Win rate:       N/A",
        f"Total PnL:      {metrics['total_pnl']:.4f}",
        f"Avg PnL/trade:  {metrics['avg_pnl']:.4f}",
        "",
        "--- Trade log (last 20) ---",
    ]

    for t in metrics.get("trades", [])[-20:]:
        lines.append(f"  {t['time'][:19]} {t['direction']:4} entry={t['entry']:.4f} outcome={t['outcome']:6} pnl={t['pnl']:+.4f}")

    if not metrics.get("trades"):
        lines.append("  (no trades)")

    diag = metrics.get("diagnostic")
    if diag:
        lines.append("")
        lines.append("--- Diagnostic (0 trades) ---")
        lines.append(f"  Sampled bars: {diag['sample_bars']}")
        lines.append(f"  prob_up: mean={diag['prob_up_mean']:.3f} max={diag['prob_up_max']:.3f}")
        lines.append(f"  prob_down: mean={diag['prob_down_mean']:.3f} max={diag['prob_down_max']:.3f}")
        lines.append(f"  Above threshold {diag['threshold']}: up={diag['above_threshold_up']} down={diag['above_threshold_down']}")
        lines.append(f"  Suggestion: {diag['suggestion']}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def save_report(symbol: str, metrics: dict, results_dir: Path) -> Path:
    """Write backtest report to file. Returns path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    path = results_dir / f"{symbol}_{month}.txt"
    path.write_text(format_report(metrics), encoding="utf-8")
    return path


# --- CLI ---


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest HQTS models on historical data (model-only by default).",
    )
    parser.add_argument("--symbol", help="Single symbol only (e.g. BTCUSD)")
    parser.add_argument("--results-dir", help="Override results directory")
    parser.add_argument("--threshold", type=float, help="Override TRADE_PROB_THRESHOLD")
    parser.add_argument("--period-days", type=int, help="Override BACKTEST_PERIOD_DAYS")
    parser.add_argument("--use-smc", action="store_true", help="Enable SMC filter (overrides BACKTEST_USE_SMC)")
    parser.add_argument("--use-market-hours", action="store_true", help="Enable market hours filter")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config = BacktestConfig()
    if args.results_dir:
        config.results_dir = Path(args.results_dir)
    if args.threshold is not None:
        config.prob_threshold = args.threshold
    if args.period_days is not None:
        config.period_days = args.period_days
    if args.use_smc:
        config.use_smc = True
    if args.use_market_hours:
        config.use_market_hours = True

    exec_config = ExecutionConfig()
    symbols_to_run = SYMBOLS
    if args.symbol:
        sym_upper = args.symbol.upper()
        symbols_to_run = [(s, m) for s, m in SYMBOLS if s.upper() == sym_upper]
        if not symbols_to_run:
            logger.error("Symbol %s not in list", args.symbol)
            sys.exit(1)

    mode = "model + SMC" if config.use_smc else "model only"
    logger.info("Backtest mode: %s | threshold=%.3f | period=%dd", mode, float(config.prob_threshold), config.period_days)
    logger.info("Symbols: %s", [s[0] for s in symbols_to_run])

    for symbol, model_dir_name in symbols_to_run:
        logger.info("Backtesting %s...", symbol)
        try:
            metrics = run_backtest(symbol, model_dir_name, config, exec_config)
            path = save_report(symbol, metrics, config.results_dir)
            win_pct = (metrics.get("win_rate") or 0) * 100
            logger.info("%s: %d trades, %.1f%% win rate -> %s", symbol, metrics.get("total_trades", 0), win_pct, path)
        except Exception as e:
            logger.exception("Backtest failed for %s: %s", symbol, e)
            save_report(symbol, {"error": str(e), "symbol": symbol}, config.results_dir)

    logger.info("Done. Results in %s", config.results_dir)


if __name__ == "__main__":
    main()
