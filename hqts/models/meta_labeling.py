"""
Meta-labeling for HQTS (Lopez de Prado).

Second model predicts "is the primary model's directional bet correct?"
Used for sizing and confidence: final_prob = primary_prob * meta_prob.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from hqts.models.config import INV_LABEL_MAP, LABEL_MAP

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    HAS_XGB = True
except (ImportError, Exception):
    HAS_XGB = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight


def build_meta_labels(
    primary_preds: np.ndarray,
    actual_outcomes: np.ndarray,
    label_map: Optional[dict[int, int]] = None,
    inv_label_map: Optional[dict[int, int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build binary meta labels from primary predictions vs actual outcomes.

    Meta model trains only on samples where primary predicted Up or Down (not Range).
    Label = 1 if primary was correct, 0 if wrong.

    Args:
        primary_preds: Primary model predictions (0=Down, 1=Range, 2=Up from sklearn).
        actual_outcomes: Actual labels (0=Down, 1=Range, 2=Up).
        label_map: Optional; default from config.
        inv_label_map: Optional; default from config.

    Returns:
        (indices, y_meta): Indices into original arrays where primary predicted Up/Down,
        and binary labels (1=correct, 0=wrong).
    """
    lm = label_map or LABEL_MAP
    inv = inv_label_map or INV_LABEL_MAP

    # Filter: primary predicted Up (2) or Down (0), not Range (1)
    directional_mask = (primary_preds == 0) | (primary_preds == 2)
    indices = np.where(directional_mask)[0]

    if len(indices) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    pred_directional = primary_preds[indices]
    actual_directional = actual_outcomes[indices]

    y_meta = (pred_directional == actual_directional).astype(np.int32)
    return indices, y_meta


def train_meta_model(
    X: np.ndarray,
    y_meta: np.ndarray,
    indices: np.ndarray,
    model_type: str = "xgboost",
    sample_weight: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Any:
    """
    Train meta classifier on directional samples.

    Args:
        X: Full feature matrix (will be subset by indices).
        y_meta: Binary labels (1=primary correct, 0=wrong).
        indices: Row indices for meta training (primary predicted Up/Down).
        model_type: "xgboost" or "random_forest".
        sample_weight: Optional sample weights.
        **kwargs: Classifier hyperparameters.

    Returns:
        Fitted classifier.
    """
    X_meta = X[indices]
    if sample_weight is not None:
        sw = sample_weight[indices]
    else:
        classes = np.unique(y_meta)
        weights = compute_class_weight("balanced", classes=classes, y=y_meta)
        class_to_weight = dict(zip(classes, weights))
        sw = np.array([class_to_weight[y] for y in y_meta])

    if model_type == "xgboost" and HAS_XGB:
        clf = xgb.XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 3),
            learning_rate=kwargs.get("learning_rate", 0.05),
            random_state=42,
            eval_metric="logloss",
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 5),
            random_state=42,
        )

    clf.fit(X_meta, y_meta, sample_weight=sw)
    return clf


def predict_meta_prob(
    meta_model: Any,
    X: np.ndarray,
    primary_preds: np.ndarray,
) -> np.ndarray:
    """
    Get meta model probability of "primary correct" for directional predictions.

    For Range predictions, returns 1.0 (no meta adjustment).

    Args:
        meta_model: Fitted meta classifier.
        X: Feature matrix.
        primary_preds: Primary predictions (0=Down, 1=Range, 2=Up).

    Returns:
        Array of probabilities (0-1). Range predictions get 1.0.
    """
    probs = np.ones(len(primary_preds), dtype=np.float64)
    directional_mask = (primary_preds == 0) | (primary_preds == 2)
    if not directional_mask.any():
        return probs

    X_dir = X[directional_mask]
    pred_proba = meta_model.predict_proba(X_dir)
    if pred_proba.shape[1] == 2:
        prob_correct = pred_proba[:, 1]
    else:
        prob_correct = pred_proba[:, 0]
    probs[directional_mask] = prob_correct
    return probs


def save_meta_model(meta_model: Any, output_dir: str | Path) -> Path:
    """Save meta model to output_dir/meta_model.joblib."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "meta_model.joblib"
    joblib.dump(meta_model, path)
    logger.info("Saved meta model: %s", path)
    return path


def load_meta_model(model_dir: str | Path) -> Optional[Any]:
    """Load meta model if it exists."""
    path = Path(model_dir) / "meta_model.joblib"
    if not path.exists():
        return None
    return joblib.load(path)
