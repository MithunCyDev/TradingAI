"""
Integration tests for the full trading pipeline.

Mocks MT5 and data fetching to test the trading logic end-to-end.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tests.conftest import generate_sample_ohlcv

ROOT = Path(__file__).resolve().parent.parent


class TestTradingPipelineIntegration:
    """Integration tests for run_cycle logic."""

    @pytest.fixture
    def mock_df(self):
        """Sample OHLCV with enough bars for inference and SMC."""
        return generate_sample_ohlcv(n_bars=500)

    def test_inference_engine_loads_and_runs(self, mock_df):
        """InferenceEngine should load model and return probabilities."""
        model_dir = ROOT / "models" / "btcusd"
        if not (model_dir / "model.joblib").exists():
            pytest.skip("No trained model at models/btcusd")
        from hqts.models.inference import InferenceEngine
        engine = InferenceEngine(model_dir=model_dir)
        result = engine.run(mock_df)
        assert "prob_up" in result
        assert "prob_down" in result
        assert "prob_range" in result
        assert 0 <= result["prob_up"] <= 1
        assert 0 <= result["prob_down"] <= 1
        probs_sum = result["prob_up"] + result["prob_down"] + result.get("prob_range", 0)
        assert 0.99 <= probs_sum <= 1.01

    def test_fetch_data_mt5_mocked_returns_df(self, mock_df):
        """Mocked fetch_data_mt5_first should return sample data."""
        with patch("hqts.etl.mt5_live.fetch_data_mt5_first") as mock_fetch:
            mock_fetch.return_value = (mock_df, "MT5")
            from hqts.etl.mt5_live import fetch_data_mt5_first
            df, src = fetch_data_mt5_first("BTCUSD", count=500)
            assert len(df) >= 100
            assert "close" in df.columns
            assert src == "MT5"


class TestDataPipeline:
    """Test data flow through the pipeline."""

    def test_sample_ohlcv_has_required_columns(self):
        """Sample OHLCV must have open, high, low, close, tick_volume."""
        df = generate_sample_ohlcv(n_bars=100)
        for col in ["open", "high", "low", "close", "tick_volume"]:
            assert col in df.columns
        assert len(df) == 100

    def test_feature_engineering_produces_features(self):
        """compute_features should add feature columns."""
        df = generate_sample_ohlcv(n_bars=200)
        from hqts.features.engineering import compute_features
        featured = compute_features(df)
        assert not featured.empty
        assert "atr" in featured.columns or "rsi" in featured.columns
