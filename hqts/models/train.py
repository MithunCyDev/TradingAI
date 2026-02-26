"""
Model training pipeline for HQTS.

Trains a tree-based classifier (XGBoost or RandomForest), persists model,
scaler, and feature config for inference.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from hqts.models.config import FEATURE_COLUMNS, INV_LABEL_MAP, LABEL_MAP, TARGET_COLUMN

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb  # noqa: F401
    HAS_XGB = True
except (ImportError, Exception):
    HAS_XGB = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


def _get_available_features(df: pd.DataFrame) -> list[str]:
    """Return feature columns that exist in the DataFrame."""
    return [c for c in FEATURE_COLUMNS if c in df.columns]


def _prepare_xy(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare X and y arrays, mapping labels to 0,1,2."""
    if feature_cols is None:
        feature_cols = _get_available_features(df)
    X = df[feature_cols].fillna(0).values
    y_raw = df[TARGET_COLUMN].values
    y = np.array([LABEL_MAP.get(int(v), 1) for v in y_raw])  # default to Range
    return X, y


def train_model(
    df: pd.DataFrame,
    model_type: str = "xgboost",
    test_size: float = 0.2,
    scale_features: bool = True,
    output_dir: str | Path = "models",
    cv_splits: Optional[int] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Train a classification model with chronological split.

    Args:
        df: Featured DataFrame with label column.
        model_type: "xgboost" or "random_forest".
        test_size: Fraction of data for holdout (from end).
        scale_features: Whether to StandardScaler X.
        output_dir: Directory to save model, scaler, config.
        cv_splits: If set, run TimeSeriesSplit cross-validation and log mean scores.
        **kwargs: Extra args for the classifier.

    Returns:
        Dict with metrics, feature_cols, and paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = _get_available_features(df)
    if not feature_cols:
        raise ValueError("No feature columns found. Ensure featured DataFrame has expected columns.")

    X, y = _prepare_xy(df, feature_cols)
    n = len(X)
    split_idx = int(n * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if cv_splits is not None and cv_splits > 1:
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        cv_train_scores, cv_test_scores = [], []
        for train_idx, test_idx in tscv.split(X):
            X_cv_train, X_cv_test = X[train_idx], X[test_idx]
            y_cv_train, y_cv_test = y[train_idx], y[test_idx]
            if scale_features:
                scaler_cv = StandardScaler()
                X_cv_train = scaler_cv.fit_transform(X_cv_train)
                X_cv_test = scaler_cv.transform(X_cv_test)
            if model_type == "xgboost" and HAS_XGB:
                clf_cv = xgb.XGBClassifier(
                    n_estimators=kwargs.get("n_estimators", 200),
                    max_depth=kwargs.get("max_depth", 6),
                    learning_rate=kwargs.get("learning_rate", 0.05),
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric="logloss",
                )
            else:
                clf_cv = RandomForestClassifier(
                    n_estimators=kwargs.get("n_estimators", 200),
                    max_depth=kwargs.get("max_depth", 12),
                    random_state=42,
                )
            clf_cv.fit(X_cv_train, y_cv_train)
            cv_train_scores.append(clf_cv.score(X_cv_train, y_cv_train))
            cv_test_scores.append(clf_cv.score(X_cv_test, y_cv_test))
        logger.info(
            "TimeSeriesSplit(n=%d): mean train=%.4f, mean test=%.4f",
            cv_splits,
            sum(cv_train_scores) / len(cv_train_scores),
            sum(cv_test_scores) / len(cv_test_scores),
        )

    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler = None

    if model_type == "xgboost" and HAS_XGB:
        clf = xgb.XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.05),
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 12),
            random_state=42,
        )

    clf.fit(X_train, y_train)
    score_train = clf.score(X_train, y_train)
    score_test = clf.score(X_test, y_test)

    # Persist
    import joblib

    model_path = output_dir / "model.joblib"
    joblib.dump(clf, model_path)
    logger.info("Saved model: %s", model_path)

    if scaler is not None:
        scaler_path = output_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        logger.info("Saved scaler: %s", scaler_path)

    config = {
        "feature_cols": feature_cols,
        "model_type": model_type,
        "scale_features": scale_features,
        "label_map": LABEL_MAP,
        "inv_label_map": INV_LABEL_MAP,
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Saved config: %s", config_path)

    result = {
        "train_accuracy": score_train,
        "test_accuracy": score_test,
        "feature_cols": feature_cols,
        "model_path": str(model_path),
        "config_path": str(config_path),
    }
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="HQTS model training")
    parser.add_argument("input", help="Path to featured CSV")
    parser.add_argument("-o", "--output-dir", default="models", help="Model output directory")
    parser.add_argument("--model", choices=["xgboost", "random_forest"], default="xgboost")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--no-scale", action="store_true", help="Disable feature scaling")
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=None,
        help="Run TimeSeriesSplit cross-validation with N splits (e.g., 5)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    result = train_model(
        df,
        model_type=args.model,
        test_size=args.test_size,
        scale_features=not args.no_scale,
        output_dir=args.output_dir,
        cv_splits=args.cv_splits,
    )
    logger.info("Train accuracy: %.4f, Test accuracy: %.4f", result["train_accuracy"], result["test_accuracy"])


if __name__ == "__main__":
    main()
