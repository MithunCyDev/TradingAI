"""
FastAPI application for HQTS prediction endpoints.

Provides prediction endpoints for BTC, gold, silver, and major forex pairs.
Fetches real-time candle data from MT5 terminal (yfinance fallback when MT5 unavailable).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# Load .env before accessing os.environ
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hqts.etl.mt5_live import fetch_data_mt5_first
from hqts.etl.economic_calendar import fetch_upcoming_events
from hqts.models.inference import InferenceEngine
from hqts.execution.risk import RiskManager

logger = logging.getLogger(__name__)

# Track MT5 connection status (set at startup)
_mt5_ok: bool = False


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Initialize MT5 at startup so API uses broker data (same as run_auto_trader)."""
    global _mt5_ok
    _mt5_ok = False
    if os.getenv("MT5_ENABLED", "true").lower() in ("true", "1", "yes"):
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                _mt5_ok = True
                logger.info("API startup: MT5 initialized successfully")
            else:
                logger.warning("API startup: MT5 init failed: %s", mt5.last_error())
        except Exception as e:
            logger.warning("API startup: MT5 not available: %s", e)
    yield
    try:
        import MetaTrader5 as mt5
        mt5.shutdown()
    except Exception:
        pass


app = FastAPI(
    title="HQTS Prediction API",
    description="ML-based directional predictions for crypto, metals, and forex",
    version="1.0.0",
    lifespan=_lifespan,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_BASE = PROJECT_ROOT / os.getenv("MODELS_BASE_DIR", "models")
DATA_BUFFER_BARS = 500
RR_RATIO = 2.0
TRADE_PROB_THRESHOLD = float(os.getenv("TRADE_PROB_THRESHOLD", "0.6"))

# Endpoint -> (symbol, model_dir_name)
ENDPOINT_SYMBOLS = {
    "btc": ("BTCUSD", "btcusd"),
    "gold": ("XAUUSD", "xauusd"),
    "silver": ("XAGUSD", "xagusd"),
    "eurusd": ("EURUSD", "eurusd"),
    "usdjpy": ("USDJPY", "usdjpy"),
    "gbpusd": ("GBPUSD", "gbpusd"),
    "audusd": ("AUDUSD", "audusd"),
    "usdchf": ("USDCHF", "usdchf"),
}

# (pip_value_per_lot, sl_pips_multiplier) per symbol for lot sizing
SYMBOL_RISK_CONFIG = {
    "BTCUSD": (1.0, 1.0),
    "XAUUSD": (10.0, 100.0),
    "XAGUSD": (50.0, 100.0),
    "EURUSD": (10.0, 10000.0),
    "USDJPY": (9.0, 100.0),
    "GBPUSD": (10.0, 10000.0),
    "AUDUSD": (10.0, 10000.0),
    "USDCHF": (10.0, 10000.0),
}

# Cache for InferenceEngine instances
_engine_cache: dict[str, InferenceEngine] = {}


def _get_engine(model_dir_name: str) -> InferenceEngine:
    """Get or create InferenceEngine for a symbol (cached)."""
    if model_dir_name not in _engine_cache:
        model_dir = MODELS_BASE / model_dir_name
        if not (model_dir / "model.joblib").exists():
            raise FileNotFoundError(
                f"Model not trained for {model_dir_name}. "
                "Run: python scripts/train_all_symbols.py"
            )
        _engine_cache[model_dir_name] = InferenceEngine(model_dir=model_dir)
    return _engine_cache[model_dir_name]


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoints."""

    symbol: str
    label: int
    direction: str
    prob_up: float
    prob_down: float
    prob_range: float
    timeframe: str
    last_close: float
    position: str
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    lot_size: Optional[float] = None
    data_fetched_at: str
    data_source: str = "MT5"  # MT5 or yfinance


def _direction_from_label(label: int) -> str:
    """Map label (-1, 0, 1) to direction string."""
    return {-1: "down", 0: "range", 1: "up"}.get(label, "range")


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """Compute ATR from OHLC arrays; returns last value."""
    n = len(close)
    if n < period + 1:
        return float(close[-1] * 0.01) if len(close) > 0 else 1.0
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.convolve(tr, np.ones(period) / period, mode="valid")
    return float(atr[-1]) if len(atr) > 0 else 1.0


def _predict_for_symbol(
    symbol: str,
    model_dir_name: str,
    timeframe: str = "15m",
    equity: Optional[float] = None,
) -> PredictionResponse:
    """
    Fetch data, run inference, and return prediction response.

    Raises:
        HTTPException: If model not found or data fetch fails.
    """
    try:
        engine = _get_engine(model_dir_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    try:
        df, data_source = fetch_data_mt5_first(
            symbol,
            timeframe=timeframe,
            count=DATA_BUFFER_BARS,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Data fetch failed: {e}") from e

    if len(df) < 100:
        raise HTTPException(
            status_code=503,
            detail=f"Insufficient data: {len(df)} bars (need >= 100)",
        )

    df = df.tail(DATA_BUFFER_BARS).reset_index(drop=True)
    events = []
    try:
        events = fetch_upcoming_events(
            from_dt=datetime.now(timezone.utc),
            to_dt=datetime.now(timezone.utc) + timedelta(days=1),
            high_impact_only=True,
        )
    except Exception:
        pass
    zone_width = float(os.getenv("SMC_ZONE_WIDTH_ATR", "0.5"))
    result = engine.run(
        df,
        events=events if events else None,
        zone_width_atr=zone_width,
    )
    last_close = float(df["close"].iloc[-1])
    direction = _direction_from_label(result["label"])

    position = "hold"
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    lot_size: Optional[float] = None

    prob_up = result["prob_up"]
    prob_down = result["prob_down"]
    meets_threshold = (
        (direction == "up" and prob_up >= TRADE_PROB_THRESHOLD and prob_up > prob_down)
        or (direction == "down" and prob_down >= TRADE_PROB_THRESHOLD and prob_down > prob_up)
    )

    if direction in ("up", "down") and meets_threshold:
        high_arr = df["high"].values
        low_arr = df["low"].values
        close_arr = df["close"].values
        atr = _compute_atr(high_arr, low_arr, close_arr)
        pip_val, sl_mult = SYMBOL_RISK_CONFIG.get(symbol, (10.0, 100.0))

        sl_dist = atr * 1.0
        tp_dist = atr * RR_RATIO
        sl_pips = sl_dist * sl_mult

        balance = equity if (equity is not None and equity > 0) else 1000.0
        risk_mgr = RiskManager(risk_per_trade_pct=0.01)
        lot_size = risk_mgr.calculate_lot_size(balance, sl_pips, pip_val)

        if direction == "up":
            position = "buy"
            take_profit = last_close + tp_dist
            stop_loss = last_close - sl_dist
        else:
            position = "sell"
            take_profit = last_close - tp_dist
            stop_loss = last_close + sl_dist

    return PredictionResponse(
        symbol=symbol,
        label=result["label"],
        direction=direction,
        prob_up=round(result["prob_up"], 2),
        prob_down=round(result["prob_down"], 2),
        prob_range=round(result["prob_range"], 2),
        timeframe=timeframe,
        last_close=last_close,
        position=position,
        take_profit=take_profit,
        stop_loss=stop_loss,
        lot_size=lot_size,
        data_fetched_at=datetime.now(timezone.utc).isoformat(),
        data_source=data_source,
    )


@app.get("/api/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict:
    """Show MT5 connection and data source. Use MT5 for predictions to match run_auto_trader."""
    return {
        "mt5_initialized": _mt5_ok,
        "data_source_note": "MT5" if _mt5_ok else "yfinance (MT5 unavailable - predictions may differ from run_auto_trader)",
    }


@app.get("/api/symbols")
def list_symbols() -> dict:
    """List supported symbols and their endpoints."""
    return {
        "symbols": [
            {"endpoint": f"/api/predict/{key}", "symbol": val[0]}
            for key, val in ENDPOINT_SYMBOLS.items()
        ]
    }


@app.get("/api/predict/btc", response_model=PredictionResponse)
def predict_btc(
    timeframe: str = "15m",
    equity: Optional[float] = Query(default=None, description="Account balance for lot size (default 10000 if not provided)"),
) -> PredictionResponse:
    """Get BTC/USD prediction. Pass equity (account balance) for lot size calculation."""
    return _predict_for_symbol("BTCUSD", "btcusd", timeframe, equity)


@app.get("/api/predict/gold", response_model=PredictionResponse)
def predict_gold(
    timeframe: str = "15m",
    equity: Optional[float] = Query(default=None, description="Account balance for lot size (default 10000 if not provided)"),
) -> PredictionResponse:
    """Get XAU/USD (gold) prediction. Pass equity (account balance) for lot size calculation."""
    return _predict_for_symbol("XAUUSD", "xauusd", timeframe, equity)


@app.get("/api/predict/silver", response_model=PredictionResponse)
def predict_silver(
    timeframe: str = "15m",
    equity: Optional[float] = Query(default=None, description="Account balance for lot size (default 10000 if not provided)"),
) -> PredictionResponse:
    """Get XAG/USD (silver) prediction. Pass equity (account balance) for lot size calculation."""
    return _predict_for_symbol("XAGUSD", "xagusd", timeframe, equity)


@app.get("/api/predict/eurusd", response_model=PredictionResponse)
def predict_eurusd(
    timeframe: str = "15m",
    equity: Optional[float] = Query(default=None, description="Account balance for lot size (default 10000 if not provided)"),
) -> PredictionResponse:
    """Get EUR/USD prediction. Pass equity (account balance) for lot size calculation."""
    return _predict_for_symbol("EURUSD", "eurusd", timeframe, equity)


@app.get("/api/predict/usdjpy", response_model=PredictionResponse)
def predict_usdjpy(
    timeframe: str = "15m",
    equity: Optional[float] = Query(default=None, description="Account balance for lot size (default 10000 if not provided)"),
) -> PredictionResponse:
    """Get USD/JPY prediction. Pass equity (account balance) for lot size calculation."""
    return _predict_for_symbol("USDJPY", "usdjpy", timeframe, equity)


@app.get("/api/predict/gbpusd", response_model=PredictionResponse)
def predict_gbpusd(
    timeframe: str = "15m",
    equity: Optional[float] = Query(default=None, description="Account balance for lot size (default 10000 if not provided)"),
) -> PredictionResponse:
    """Get GBP/USD prediction. Pass equity (account balance) for lot size calculation."""
    return _predict_for_symbol("GBPUSD", "gbpusd", timeframe, equity)


@app.get("/api/predict/audusd", response_model=PredictionResponse)
def predict_audusd(
    timeframe: str = "15m",
    equity: Optional[float] = Query(default=None, description="Account balance for lot size (default 10000 if not provided)"),
) -> PredictionResponse:
    """Get AUD/USD prediction. Pass equity (account balance) for lot size calculation."""
    return _predict_for_symbol("AUDUSD", "audusd", timeframe, equity)


@app.get("/api/predict/usdchf", response_model=PredictionResponse)
def predict_usdchf(
    timeframe: str = "15m",
    equity: Optional[float] = Query(default=None, description="Account balance for lot size (default 10000 if not provided)"),
) -> PredictionResponse:
    """Get USD/CHF prediction. Pass equity (account balance) for lot size calculation."""
    return _predict_for_symbol("USDCHF", "usdchf", timeframe, equity)
