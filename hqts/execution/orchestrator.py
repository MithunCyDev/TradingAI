"""
Signal orchestration for HQTS.

Ties together: data buffer, ML inference, SMC filter, risk manager, executor.
Runs in a loop (or on bar-close events) to evaluate and execute trades.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

import numpy as np

from hqts.execution.config import ExecutionConfig
from hqts.execution.executor import OrderExecutor, OrderType
from hqts.execution.market_hours import MarketHoursFilter
from hqts.execution.risk import RiskManager
from hqts.execution.smc import SMCFilter
from hqts.models.inference import InferenceEngine

logger = logging.getLogger(__name__)


class TradingOrchestrator:
    """
    Main orchestration loop: fetch data -> ML signal -> SMC filter -> risk check -> execute.
    """

    def __init__(self, config: Optional[ExecutionConfig] = None) -> None:
        self.config = config or ExecutionConfig()
        r = self.config.risk
        self.risk = RiskManager(
            risk_per_trade_pct=r.risk_per_trade_pct,
            daily_drawdown_limit_pct=r.daily_drawdown_limit_pct,
            max_open_per_symbol=r.max_open_trades_per_symbol,
            max_total_open=r.max_total_open_trades,
        )
        self.executor = OrderExecutor(
            symbol=self.config.order.symbol,
            max_slippage_points=self.config.order.max_slippage_points,
            paper_trade=self.config.order.paper_trade,
        )
        self.smc = SMCFilter(
            require_order_block=self.config.smc.require_order_block,
            require_fvg=self.config.smc.require_fvg,
            require_liquidity_sweep=self.config.smc.require_liquidity_sweep,
            ob_lookback=self.config.smc.ob_lookback_bars,
            fvg_min_size_atr=self.config.smc.fvg_min_size_atr,
            min_ob_strength=self.config.smc.min_ob_strength,
            zone_width_atr=self.config.smc.zone_width_atr,
            require_price_in_zone=self.config.smc.require_price_in_zone,
        )
        self.market_hours = MarketHoursFilter(self.config.market_hours)
        self.inference = InferenceEngine(model_dir=self.config.model_dir)
        self._buffer: pd.DataFrame = pd.DataFrame()

    def update_buffer(self, df: pd.DataFrame) -> None:
        """Update in-memory candle buffer (call from data adapter)."""
        self._buffer = df.tail(self.config.data_buffer_bars).copy()

    def evaluate_signal(self, equity: float = 0.0) -> Optional[OrderType]:
        """
        Evaluate whether to open a trade. Returns OrderType or None.

        Phase 1: ML predicts direction.
        Phase 2: SMC filter validates.
        Phase 3: Risk manager allows.
        """
        if self._buffer.empty or len(self._buffer) < 100:
            return None

        if not self.risk.is_trading_allowed(equity):
            return None

        if not self.market_hours.is_trading_allowed():
            logger.debug("Market closed: skipping trade")
            return None

        result = self.inference.run(
            self._buffer,
            zone_width_atr=self.config.smc.zone_width_atr,
        )
        prob_up = result["prob_up"]
        prob_down = result["prob_down"]
        threshold = float(os.getenv("TRADE_PROB_THRESHOLD", "0.6"))

        if prob_up >= threshold and prob_up > prob_down:
            if self.smc.validate_buy(self._buffer):
                return OrderType.BUY
        elif prob_down >= threshold and prob_down > prob_up:
            if self.smc.validate_sell(self._buffer):
                return OrderType.SELL

        return None

    def _last_atr(self, period: int = 14) -> float:
        """Compute ATR from buffer if not already present."""
        if self._buffer.empty:
            return 1.0
        if "atr" in self._buffer.columns:
            val = self._buffer["atr"].iloc[-1]
            return float(val) if not pd.isna(val) else 1.0
        high = self._buffer["high"].values
        low = self._buffer["low"].values
        close = self._buffer["close"].values
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        atr = pd.Series(tr).rolling(period).mean().iloc[-1]
        return float(atr) if not np.isnan(atr) else 1.0

    def execute_signal(
        self,
        signal: OrderType,
        equity: float,
        atr: Optional[float] = None,
        pip_value_per_lot: Optional[float] = None,
        sl_pips_multiplier: Optional[float] = None,
    ) -> bool:
        """
        Execute a trade: compute lot size, TP/SL, place order.
        """
        if atr is None:
            atr = self._last_atr()
        pip_val = pip_value_per_lot if pip_value_per_lot is not None else self.config.order.pip_value_per_lot
        mult = sl_pips_multiplier if sl_pips_multiplier is not None else self.config.order.sl_pips_multiplier
        rr = self.config.risk.rr_ratio
        sl_dist = atr * 1.0
        tp_dist = atr * rr
        sl_pips = sl_dist * mult
        lot = self.risk.calculate_lot_size(equity, sl_pips, pip_val)

        if self.config.order.lot_size is not None:
            lot = self.config.order.lot_size

        close = float(self._buffer["close"].iloc[-1])
        if signal == OrderType.BUY:
            sl_price = close - sl_dist
            tp_price = close + tp_dist
        else:
            sl_price = close + sl_dist
            tp_price = close - tp_dist

        res = self.executor.place_market_order(signal, lot, sl_price, tp_price)
        return res.success
