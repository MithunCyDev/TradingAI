#!/usr/bin/env python3
"""
Fetch 2 months of BTCUSD and XAUUSD across 15m, 1h, 4h timeframes, run feature
pipeline, and fine-tune the model.

Uses yfinance for both (MT5 optional for XAUUSD). 4h data is resampled from 1h.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from hqts.etl.yfinance_fetch import fetch_multi_symbol_multi_timeframe
from hqts.features.pipeline import run_feature_pipeline
from hqts.models.train import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "clean"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Fetch 2 months of 15m, 1h, 4h data for BTCUSD and XAUUSD
    logger.info("Fetching 2 months of BTCUSD and XAUUSD (15m, 1h, 4h)...")
    df = fetch_multi_symbol_multi_timeframe(
        symbols=["BTCUSD", "XAUUSD"],
        intervals=["15m", "1h", "4h"],
        period="60d",
        output_dir=str(data_dir),
        use_mt5=False,
    )

    # Run feature pipeline on combined data
    featured_path = data_dir / "multi_symbol_multi_tf_2mo_featured.csv"
    logger.info("Running feature pipeline...")
    featured = run_feature_pipeline(
        input_path=data_dir / "multi_symbol_multi_tf_2mo.csv",
        output_path=featured_path,
        rr_ratio=2.0,
        horizon_bars=16,
    )

    # Train (fine-tune) model on combined dataset
    logger.info("Training model on combined BTCUSD + XAUUSD (15m, 1h, 4h) data...")
    featured_df = pd.read_csv(featured_path)
    featured_df["time"] = pd.to_datetime(featured_df["time"])
    featured_df = featured_df.sort_values("time").reset_index(drop=True)

    result = train_model(
        featured_df,
        model_type="random_forest",
        test_size=0.2,
        scale_features=True,
        output_dir=str(project_root / "models"),
    )

    logger.info(
        "Fine-tuned model: train_acc=%.4f, test_acc=%.4f",
        result["train_accuracy"],
        result["test_accuracy"],
    )
    logger.info("Model saved to %s", result["model_path"])


if __name__ == "__main__":
    main()
