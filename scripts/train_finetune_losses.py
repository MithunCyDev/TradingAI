#!/usr/bin/env python3
"""
Fine-tune models on loss samples.

Loads base model, extracts loss samples from data/loss_trades.jsonl,
retrains on base + loss data with higher weight for loss rows, and saves.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import pandas as pd

from hqts.etl.loss_samples import extract_loss_samples
from hqts.models.train import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYMBOLS = [
    "BTCUSD", "XAUUSD", "XAGUSD", "EURUSD", "USDJPY", "GBPUSD", "AUDUSD",
    "USTECH", "USOIL",
]


def finetune_symbol(
    symbol: str,
    models_base: Path,
    data_dir: Path,
    loss_path: Path,
    loss_weight: float = 2.0,
) -> bool:
    """
    Fine-tune model for symbol on loss samples.

    Returns:
        True if fine-tuned, False if skipped.
    """
    model_dir = models_base / symbol.lower()
    if not (model_dir / "model.joblib").exists():
        logger.debug("Skipping %s: no base model", symbol)
        return False

    loss_df = extract_loss_samples(
        loss_path,
        data_dir=data_dir,
        context_bars=100,
        period="1y",
    )
    if loss_df.empty:
        logger.info("Skipping %s: no loss samples", symbol)
        return False
    loss_symbol = loss_df[loss_df["symbol"].str.upper() == symbol.upper()] if "symbol" in loss_df.columns else loss_df
    if loss_symbol.empty:
        logger.info("Skipping %s: no loss samples for this symbol", symbol)
        return False

    # Load base training data (featured)
    base_path = None
    for period in ("1y", "6mo"):
        for suffix in ("_pullback_featured", "_featured"):
            p = data_dir / f"{symbol.lower()}_{period}{suffix}.csv"
            if p.exists():
                base_path = p
                break
        if base_path:
            break
    if base_path is None:
        logger.warning("Skipping %s: no base featured data", symbol)
        return False

    base_df = pd.read_csv(base_path)
    if "time" in base_df.columns:
        base_df["time"] = pd.to_datetime(base_df["time"])
    if "timeframe" in base_df.columns:
        base_df = base_df[base_df["timeframe"].astype(str).str.lower().isin(["15m", "m15"])].copy()
    base_df = base_df.sort_values("time").reset_index(drop=True)

    # Align columns: use only columns present in both
    common_cols = [c for c in base_df.columns if c in loss_symbol.columns]
    base_df = base_df[common_cols].copy()
    loss_symbol = loss_symbol[common_cols].copy()
    for c in common_cols:
        if c not in loss_symbol.columns:
            loss_symbol[c] = 0
        loss_symbol[c] = pd.to_numeric(loss_symbol[c], errors="coerce").fillna(0)

    combined = pd.concat([base_df, loss_symbol], ignore_index=True)
    combined = combined.dropna(subset=["atr", "rsi", "label"]).reset_index(drop=True)

    n_base = len(base_df)
    n_loss = len(loss_symbol)
    sample_weight = np.ones(len(combined))
    sample_weight[n_base:] = loss_weight

    logger.info("Fine-tuning %s: %d base + %d loss samples (loss weight=%.1f)", symbol, n_base, n_loss, loss_weight)

    result = train_model(
        combined,
        model_type="random_forest",
        test_size=0.2,
        scale_features=True,
        output_dir=str(model_dir),
        sample_weight=sample_weight,
    )
    logger.info("%s: train_acc=%.4f, test_acc=%.4f", symbol, result["train_accuracy"], result["test_accuracy"])
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune models on loss samples")
    parser.add_argument(
        "--symbol",
        choices=SYMBOLS,
        default=None,
        help="Fine-tune only this symbol",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Base directory for models",
    )
    parser.add_argument(
        "--data-dir",
        default="data/clean",
        help="Directory for featured data",
    )
    parser.add_argument(
        "--loss-path",
        default="data/loss_trades.jsonl",
        help="Path to loss_trades.jsonl",
    )
    parser.add_argument(
        "--loss-weight",
        type=float,
        default=2.0,
        help="Sample weight for loss rows (default: 2.0)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    models_base = Path(args.models_dir) if Path(args.models_dir).is_absolute() else project_root / args.models_dir
    data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else project_root / args.data_dir
    loss_path = Path(args.loss_path) if Path(args.loss_path).is_absolute() else project_root / args.loss_path

    if not loss_path.exists():
        logger.warning("No loss trades at %s; run sync_closed_trades.py first", loss_path)
        sys.exit(0)

    symbols = [args.symbol] if args.symbol else SYMBOLS
    finetuned = 0
    for sym in symbols:
        try:
            if finetune_symbol(sym, models_base, data_dir, loss_path, args.loss_weight):
                finetuned += 1
        except Exception as e:
            logger.exception("Failed to fine-tune %s: %s", sym, e)

    logger.info("Done: %d models fine-tuned", finetuned)


if __name__ == "__main__":
    main()
