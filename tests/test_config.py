"""
Tests for execution configuration.

Validates config loading and env var parsing.
"""

import os

import pytest

from hqts.execution.config import (
    ExecutionConfig,
    RiskConfig,
    SMCConfig,
    _env_bool,
    _env_float,
    _env_int,
)


class TestConfigHelpers:
    """Test env parsing helpers."""

    def test_env_bool_true_values(self):
        """'true', '1', 'yes' should parse as True."""
        try:
            for val in ("true", "1", "yes", "TRUE", "Yes"):
                os.environ["_TEST_BOOL"] = val
                assert _env_bool("_TEST_BOOL", False) is True
        finally:
            os.environ.pop("_TEST_BOOL", None)

    def test_env_bool_false_values(self):
        """'false', '0', 'no' and others should parse as False."""
        try:
            for val in ("false", "0", "no", "", "x"):
                os.environ["_TEST_BOOL"] = val
                assert _env_bool("_TEST_BOOL", True) is False
        finally:
            os.environ.pop("_TEST_BOOL", None)

    def test_env_bool_missing_uses_default(self):
        """Missing env var should use default."""
        assert _env_bool("_NONEXISTENT_VAR_123", True) is True
        assert _env_bool("_NONEXISTENT_VAR_456", False) is False

    def test_env_int_valid(self):
        """Valid int string should parse."""
        os.environ["_TEST_INT"] = "42"
        assert _env_int("_TEST_INT", 0) == 42
        del os.environ["_TEST_INT"]

    def test_env_int_invalid_uses_default(self):
        """Invalid int should use default."""
        os.environ["_TEST_INT"] = "not_a_number"
        assert _env_int("_TEST_INT", 99) == 99
        del os.environ["_TEST_INT"]

    def test_env_float_valid(self):
        """Valid float string should parse."""
        os.environ["_TEST_FLOAT"] = "0.55"
        assert _env_float("_TEST_FLOAT", 0.0) == 0.55
        del os.environ["_TEST_FLOAT"]


class TestExecutionConfig:
    """Test ExecutionConfig instantiation."""

    def test_config_loads_with_defaults(self):
        """ExecutionConfig should load with sensible defaults."""
        config = ExecutionConfig()
        assert config.risk.risk_per_trade_pct == 0.01
        assert config.smc.ob_lookback_bars >= 1  # Clamped from env

    def test_smc_config_ob_lookback_minimum(self):
        """SMC ob_lookback_bars should be at least 1."""
        config = ExecutionConfig()
        assert config.smc.ob_lookback_bars >= 1
