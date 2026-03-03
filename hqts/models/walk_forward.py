"""
Walk-forward time-series training for HQTS.

Rolling or expanding windows with strict out-of-sample validation.
Supports triple-barrier labels, meta-labeling, regime awareness, and hyperparameter optimization.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from hqts.features.engineering import compute_features
from hqts.features.labeling import compute_labels_triple_barrier
from hqts.models.config import FEATURE_COLUMNS, INV_LABEL_MAP, LABEL_MAP, TARGET_COLUMN
from hqts.models.meta_labeling import (
    build_meta_labels,
    load_meta_model,
    predict_meta_prob,
    save_meta_model,
    train_meta_model,
)
from hqts.models.train import _get_available_features, _prepare_xy, train_model

logger = logging.getLogger(__name__)

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import xgboost as xgb
    HAS_XGB = True
except (ImportError, Exception):
    HAS_XGB = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward training."""

    train_years: int = 10
    test_years: int = 1
    step_years: int = 1
    mode: str = "rolling"  # rolling | expanding
    atr_mult_sl: float = 1.0
    rr_ratio: float = 2.0
    horizon_bars: int = 16
    vertical_barrier_bars: int = 16
    meta_labeling: bool = True
    regime_aware: bool = True
    hyperopt: bool = True
    hyperopt_trials: int = 50
    model_type: str = "xgboost"


def generate_walk_forward_folds(
    start_year: int,
    end_year: int,
    train_years: int = 10,
    test_years: int = 1,
    step_years: int = 1,
    mode: str = "rolling",
) -> list[tuple[datetime, datetime, datetime, datetime]]:
    """
    Generate walk-forward fold boundaries (train_start, train_end, test_start, test_end).

    Rolling: fixed train window, step forward each fold.
    Expanding: train window grows from fixed start, test advances.

    Returns:
        List of (train_start, train_end, test_start, test_end) as datetime objects (UTC).
    """
    from datetime import timezone

    tz = timezone.utc
    folds = []
    test_start_year = start_year + train_years

    if mode == "rolling":
        train_start_year = start_year
        while test_start_year + test_years <= end_year + 1:
            train_end_year = test_start_year - 1
            test_end_year = test_start_year + test_years - 1

            train_start = datetime(train_start_year, 1, 1, tzinfo=tz)
            train_end = datetime(train_end_year, 12, 31, 23, 59, 59, tzinfo=tz)
            test_start = datetime(test_start_year, 1, 1, tzinfo=tz)
            test_end = datetime(test_end_year, 12, 31, 23, 59, 59, tzinfo=tz)

            folds.append((train_start, train_end, test_start, test_end))
            train_start_year += step_years
            test_start_year += step_years

    else:
        train_start_year = start_year
        while test_start_year + test_years <= end_year + 1:
            train_end_year = test_start_year - 1
            test_end_year = test_start_year + test_years - 1

            train_start = datetime(train_start_year, 1, 1, tzinfo=tz)
            train_end = datetime(train_end_year, 12, 31, 23, 59, 59, tzinfo=tz)
            test_start = datetime(test_start_year, 1, 1, tzinfo=tz)
            test_end = datetime(test_end_year, 12, 31, 23, 59, 59, tzinfo=tz)

            folds.append((train_start, train_end, test_start, test_end))
            test_start_year += step_years

    return folds


def _compute_regime_weights(df: pd.DataFrame) -> np.ndarray:
    """Inverse frequency weighting for volatility_regime to balance regimes."""
    if "volatility_regime" not in df.columns:
        return np.ones(len(df))
    regimes = df["volatility_regime"].fillna(0).values
    unique, counts = np.unique(regimes, return_counts=True)
    regime_to_weight = {r: 1.0 / max(c, 1) for r, c in zip(unique, counts)}
    total = sum(regime_to_weight.values())
    regime_to_weight = {r: w / total * len(unique) for r, w in regime_to_weight.items()}
    return np.array([regime_to_weight.get(r, 1.0) for r in regimes])


def _run_hyperopt(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: WalkForwardConfig,
    sample_weight: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Run Optuna hyperparameter optimization on train set."""
    if not HAS_OPTUNA:
        return {}

    def objective(trial: Any) -> float:
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)

        tscv = TimeSeriesSplit(n_splits=4)
        scaler = StandardScaler()
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_t, X_v = X_train[train_idx], X_train[val_idx]
            y_t, y_v = y_train[train_idx], y_train[val_idx]
            X_t = scaler.fit_transform(X_t)
            X_v = scaler.transform(X_v)

            if config.model_type == "xgboost" and HAS_XGB:
                clf = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    random_state=42,
                    eval_metric="logloss",
                )
            else:
                clf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                )

            sw = sample_weight[train_idx] if sample_weight is not None else None
            clf.fit(X_t, y_t, sample_weight=sw)
            pred = clf.predict(X_v)
            acc = (pred == y_v).mean()
            scores.append(acc)

        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config.hyperopt_trials, show_progress_bar=False)
    best = study.best_params
    logger.info("Hyperopt best params: %s", best)
    return best


def run_walk_forward_training(
    df: pd.DataFrame,
    folds: list[tuple[datetime, datetime, datetime, datetime]],
    config: WalkForwardConfig,
    output_dir: str | Path,
    save_last_fold_only: bool = True,
) -> dict[str, Any]:
    """
    Run walk-forward training across folds.

    For each fold: slice by date, compute features+labels, optionally hyperopt,
    train primary and meta, evaluate on test (strict OOS).

    Args:
        df: Raw OHLCV DataFrame with time, open, high, low, close, tick_volume, symbol, timeframe.
        folds: List of (train_start, train_end, test_start, test_end).
        config: WalkForwardConfig.
        output_dir: Base directory for models (last fold saved to output_dir/symbol).
        save_last_fold_only: If True, only persist model from final fold.

    Returns:
        Dict with fold metrics and paths.
    """
    output_dir = Path(output_dir)
    results = {"folds": [], "test_accuracies": [], "best_params": None}

    if "time" not in df.columns:
        raise ValueError("DataFrame must have 'time' column for walk-forward slicing")
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC", ambiguous="infer")
    else:
        df["time"] = df["time"].dt.tz_convert("UTC")

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
        logger.info(
            "Fold %d: train %s-%s, test %s-%s",
            fold_idx + 1,
            train_start.date(),
            train_end.date(),
            test_start.date(),
            test_end.date(),
        )

        train_mask = (df["time"] >= train_start) & (df["time"] <= train_end)
        test_mask = (df["time"] >= test_start) & (df["time"] <= test_end)
        df_train_raw = df.loc[train_mask].copy()
        df_test_raw = df.loc[test_mask].copy()

        if df_train_raw.empty or df_test_raw.empty:
            logger.warning("Fold %d: empty train or test, skipping", fold_idx + 1)
            continue

        df_train = compute_features(df_train_raw)
        df_train["label"] = compute_labels_triple_barrier(
            df_train,
            rr_ratio=config.rr_ratio,
            horizon_bars=config.horizon_bars,
            atr_mult_sl=config.atr_mult_sl,
            vertical_barrier_bars=config.vertical_barrier_bars,
        )

        df_test = compute_features(df_test_raw)
        df_test["label"] = compute_labels_triple_barrier(
            df_test,
            rr_ratio=config.rr_ratio,
            horizon_bars=config.horizon_bars,
            atr_mult_sl=config.atr_mult_sl,
            vertical_barrier_bars=config.vertical_barrier_bars,
        )

        df_train = df_train.dropna(subset=["atr", "rsi", "label"]).reset_index(drop=True)
        df_test = df_test.dropna(subset=["atr", "rsi", "label"]).reset_index(drop=True)

        if df_train.empty or df_test.empty:
            logger.warning("Fold %d: no valid rows after dropna, skipping", fold_idx + 1)
            continue

        feature_cols = _get_available_features(df_train)
        if not feature_cols:
            logger.warning("Fold %d: no feature columns, skipping", fold_idx + 1)
            continue

        X_train, y_train = _prepare_xy(df_train, feature_cols)
        X_test, y_test = _prepare_xy(df_test, feature_cols)

        sample_weight = None
        if config.regime_aware:
            sample_weight = _compute_regime_weights(df_train)

        train_kwargs = {}
        if config.hyperopt and HAS_OPTUNA:
            best_params = _run_hyperopt(X_train, y_train, config, sample_weight)
            if best_params:
                train_kwargs.update(best_params)
                results["best_params"] = best_params

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_to_weight = dict(zip(classes, weights))
        sw_train = np.array([class_to_weight[y] for y in y_train])
        if sample_weight is not None:
            sw_train = sw_train * sample_weight

        if config.model_type == "xgboost" and HAS_XGB:
            clf = xgb.XGBClassifier(
                n_estimators=train_kwargs.get("n_estimators", 300),
                max_depth=train_kwargs.get("max_depth", 5),
                learning_rate=train_kwargs.get("learning_rate", 0.03),
                subsample=train_kwargs.get("subsample", 0.8),
                colsample_bytree=train_kwargs.get("colsample_bytree", 0.8),
                random_state=42,
                eval_metric="logloss",
            )
        else:
            clf = RandomForestClassifier(
                n_estimators=train_kwargs.get("n_estimators", 200),
                max_depth=train_kwargs.get("max_depth", 12),
                random_state=42,
            )

        clf.fit(X_train_scaled, y_train, sample_weight=sw_train)
        primary_preds = clf.predict(X_train_scaled)

        meta_model = None
        if config.meta_labeling:
            indices, y_meta = build_meta_labels(primary_preds, y_train)
            if len(indices) > 10:
                meta_model = train_meta_model(
                    X_train_scaled, y_meta, indices, model_type=config.model_type
                )

        score_test = clf.score(X_test_scaled, y_test)
        results["test_accuracies"].append(score_test)
        results["folds"].append(
            {
                "fold": fold_idx + 1,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "test_accuracy": score_test,
            }
        )
        logger.info("Fold %d test accuracy: %.4f", fold_idx + 1, score_test)

        if not save_last_fold_only or fold_idx == len(folds) - 1:
            fold_dir = output_dir
            fold_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(clf, fold_dir / "model.joblib")
            joblib.dump(scaler, fold_dir / "scaler.joblib")
            if meta_model is not None:
                save_meta_model(meta_model, fold_dir)
            config_dict = {
                "feature_cols": feature_cols,
                "model_type": config.model_type,
                "scale_features": True,
                "label_map": LABEL_MAP,
                "inv_label_map": INV_LABEL_MAP,
            }
            with open(fold_dir / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)
            logger.info("Saved model to %s", fold_dir)

    if results["test_accuracies"]:
        results["mean_test_accuracy"] = float(np.mean(results["test_accuracies"]))
        logger.info("Mean test accuracy across folds: %.4f", results["mean_test_accuracy"])

    return results
