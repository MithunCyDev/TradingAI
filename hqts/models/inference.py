"""
Inference engine for HQTS ML model.

Loads persisted model, scaler, and config; computes features from raw candles;
returns directional probabilities (Up, Down, Ranging).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from hqts.features.engineering import compute_features
from hqts.models.config import INV_LABEL_MAP

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Lightweight inference wrapper for HQTS classifier.

    Given latest OHLCV candles, computes features and returns
    class probabilities (Down, Range, Up).
    """

    def __init__(
        self,
        model_dir: str | Path = "models",
        model_path: Optional[str | Path] = None,
        scaler_path: Optional[str | Path] = None,
        config_path: Optional[str | Path] = None,
    ) -> None:
        """
        Load model, scaler, and config from directory or explicit paths.

        Args:
            model_dir: Base directory containing model.joblib, scaler.joblib, config.json.
            model_path: Override path to model file.
            scaler_path: Override path to scaler file.
            config_path: Override path to config file.
        """
        model_dir = Path(model_dir)
        self._model_path = Path(model_path) if model_path else model_dir / "model.joblib"
        self._scaler_path = Path(scaler_path) if scaler_path else model_dir / "scaler.joblib"
        self._config_path = Path(config_path) if config_path else model_dir / "config.json"

        import joblib

        self._model = joblib.load(self._model_path)
        self._scaler = joblib.load(self._scaler_path) if self._scaler_path.exists() else None
        with open(self._config_path) as f:
            self._config = json.load(f)
        self._feature_cols = self._config["feature_cols"]
        self._scale_features = self._config.get("scale_features", True)
        inv = self._config.get("inv_label_map", INV_LABEL_MAP)
        self._inv_label_map = {int(k): v for k, v in inv.items()}

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute class probabilities for the last row(s) of a featured DataFrame.

        Args:
            df: DataFrame with feature columns (from compute_features).

        Returns:
            Array of shape (n_samples, 3) with probs for [Down, Range, Up].
        """
        n = len(df)
        X = np.zeros((n, len(self._feature_cols)))
        for i, c in enumerate(self._feature_cols):
            if c in df.columns:
                X[:, i] = df[c].fillna(0).values
            else:
                logger.debug("Missing feature %s; using 0", c)
        if self._feature_cols is None or len(self._feature_cols) == 0:
            X = np.zeros((n, 1))

        if self._scale_features and self._scaler is not None:
            X = self._scaler.transform(X)

        proba = self._model.predict_proba(X)
        return proba

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted class labels (-1, 0, 1)."""
        proba = self.predict_proba(df)
        pred_idx = np.argmax(proba, axis=1)
        return np.array([self._inv_label_map.get(i, 0) for i in pred_idx])

    def run(self, ohlcv_df: pd.DataFrame) -> dict:
        """
        Full pipeline: compute features from OHLCV, then predict.

        Args:
            ohlcv_df: Raw OHLCV DataFrame with time, open, high, low, close, tick_volume.

        Returns:
            Dict with keys: label, prob_up, prob_down, prob_range, probabilities.
        """
        featured = compute_features(ohlcv_df)
        if featured.empty:
            return {"label": 0, "prob_up": 0.0, "prob_down": 0.0, "prob_range": 1.0, "probabilities": None}

        proba = self.predict_proba(featured)
        last = proba[-1]
        pred_idx = int(np.argmax(last))
        label = self._inv_label_map.get(pred_idx, 0)

        # Map by label: -1=Down, 0=Range, 1=Up; proba columns follow model's classes_
        classes = getattr(self._model, "classes_", [0, 1, 2])
        idx_down = list(classes).index(0) if 0 in classes else 0
        idx_range = list(classes).index(1) if 1 in classes else 1
        idx_up = list(classes).index(2) if 2 in classes else 2

        return {
            "label": label,
            "prob_up": float(last[idx_up]) if len(last) > idx_up else 0.0,
            "prob_down": float(last[idx_down]) if len(last) > idx_down else 0.0,
            "prob_range": float(last[idx_range]) if len(last) > idx_range else 0.0,
            "probabilities": last.tolist(),
        }
