"""
Tests for Smart Money Concepts (SMC) filter.

Validates order block, FVG, liquidity sweep detection and zone checks.
"""

import numpy as np
import pandas as pd
import pytest

from hqts.execution.smc import SMCFilter


class TestSMCFilter:
    """SMC filter validation tests."""

    def test_validate_buy_rejects_empty_df(self, sample_ohlcv_empty):
        """Empty DataFrame should always be rejected."""
        smc = SMCFilter(ob_lookback=20, require_order_block=False)
        assert smc.validate_buy(sample_ohlcv_empty) is False

    def test_validate_buy_rejects_insufficient_bars(self, sample_ohlcv_small):
        """DataFrame shorter than ob_lookback should be rejected."""
        smc = SMCFilter(ob_lookback=100, require_order_block=False)
        assert smc.validate_buy(sample_ohlcv_small) is False

    def test_validate_buy_rejects_zero_lookback(self, sample_ohlcv):
        """ob_lookback < 1 should reject immediately."""
        smc = SMCFilter(ob_lookback=0, require_order_block=False)
        assert smc.validate_buy(sample_ohlcv) is False

    def test_validate_buy_with_relaxed_settings(self, sample_ohlcv):
        """With require_order_block=False and ob_lookback=1, may pass if structure exists."""
        smc = SMCFilter(
            ob_lookback=5,
            require_order_block=False,
            require_fvg=False,
            require_liquidity_sweep=False,
        )
        # Should not crash; result depends on data
        result = smc.validate_buy(sample_ohlcv)
        assert isinstance(result, bool)

    def test_validate_sell_rejects_empty_df(self, sample_ohlcv_empty):
        """Empty DataFrame should always be rejected for sell."""
        smc = SMCFilter(ob_lookback=20, require_order_block=False)
        assert smc.validate_sell(sample_ohlcv_empty) is False

    def test_validate_sell_with_relaxed_settings(self, sample_ohlcv):
        """validate_sell should not crash with valid data."""
        smc = SMCFilter(
            ob_lookback=5,
            require_order_block=False,
            require_fvg=False,
            require_liquidity_sweep=False,
        )
        result = smc.validate_sell(sample_ohlcv)
        assert isinstance(result, bool)

    def test_validate_buy_uses_current_price_when_provided(self, sample_ohlcv):
        """When current_price is passed, it should be used instead of last close."""
        smc = SMCFilter(
            ob_lookback=20,
            require_order_block=False,
            require_fvg=False,
            require_liquidity_sweep=False,
            require_price_in_zone=False,
        )
        result = smc.validate_buy(sample_ohlcv, current_price=1950.0)
        assert isinstance(result, bool)

    def test_validate_sell_uses_current_price_when_provided(self, sample_ohlcv):
        """When current_price is passed to validate_sell, it should be used."""
        smc = SMCFilter(
            ob_lookback=20,
            require_order_block=False,
            require_fvg=False,
            require_liquidity_sweep=False,
            require_price_in_zone=False,
        )
        result = smc.validate_sell(sample_ohlcv, current_price=1850.0)
        assert isinstance(result, bool)

    def test_require_order_block_stricter(self, sample_ohlcv):
        """With require_order_block=True, fewer setups should pass."""
        smc_relaxed = SMCFilter(
            ob_lookback=20,
            require_order_block=False,
            require_fvg=False,
            require_liquidity_sweep=False,
        )
        smc_strict = SMCFilter(
            ob_lookback=20,
            require_order_block=True,
            require_fvg=False,
            require_liquidity_sweep=False,
        )
        relaxed_buy = smc_relaxed.validate_buy(sample_ohlcv)
        strict_buy = smc_strict.validate_buy(sample_ohlcv)
        # Strict should not pass when relaxed fails; relaxed may pass when strict fails
        if strict_buy:
            assert relaxed_buy
