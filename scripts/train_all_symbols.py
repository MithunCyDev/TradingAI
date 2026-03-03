#!/usr/bin/env python3
"""
Train per-symbol models for HQTS.

Fetches data for each symbol across multiple timeframes (1m, 3m, 5m, 45m, 1h, 2h, 4h, 1d, 1w)
from MT5 over 1 month by default, runs feature pipeline, and trains a model. Skips symbols
that already have a trained model unless --force.
"""

import argparse
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

SYMBOLS = [
    "BTCUSD",
    "XAUUSD",
    "XAGUSD",
    "EURUSD",
    "USDJPY",
    "GBPUSD",
    "AUDUSD",
    "USTECH",
    "USOIL",
]


def _symbol_to_model_dir(symbol: str, models_base: Path) -> Path:
    """Convert symbol to model directory (e.g., BTCUSD -> models/btcusd)."""
    return models_base / symbol.lower()


def _model_exists(model_dir: Path) -> bool:
    """Check if a trained model exists for the given directory."""
    return (model_dir / "model.joblib").exists()


def _period_suffix(period: str) -> str:
    """Return file suffix from period (e.g. 2y -> 2y, 1y -> 1y, 3mo -> 3mo, 1mo -> 1mo, 1w -> 1w)."""
    p = period.lower()
    if "2y" in p or "730" in p:
        return "2y"
    if "1y" in p or "365" in p:
        return "1y"
    if "6mo" in p or "180" in p:
        return "6mo"
    if "3mo" in p or "90d" in p:
        return "3mo"
    if "1mo" in p or "30d" in p:
        return "1mo"
    if "1w" in p or "7d" in p:
        return "1w"
    return "2mo"


def train_symbol(
    symbol: str,
    data_dir: Path,
    models_base: Path,
    period: str = "1mo",
    force: bool = False,
    pullback_mode: bool = True,
    zone_width_atr: float = 0.75,
    model_type: str = "xgboost",
    n_estimators: int | None = None,
    max_depth: int | None = None,
    learning_rate: float | None = None,
    mt5_only: bool = True,
) -> dict | None:
    """
    Train a model for a single symbol.

    Returns:
        Result dict from train_model, or None if skipped.
    """
    model_dir = _symbol_to_model_dir(symbol, models_base)
    if _model_exists(model_dir) and not force:
        logger.info("Skipping %s: model already exists at %s", symbol, model_dir)
        return None

    logger.info("Training model for %s...", symbol)
    data_dir.mkdir(parents=True, exist_ok=True)

    suffix = _period_suffix(period)
    pullback_suffix = "_pullback" if pullback_mode else ""
    raw_path = data_dir / f"{symbol.lower()}_{suffix}.csv"
    featured_path = data_dir / f"{symbol.lower()}_{suffix}{pullback_suffix}_featured.csv"

    df = fetch_multi_symbol_multi_timeframe(
        symbols=[symbol],
        intervals=["15m", "1m", "3m", "5m", "45m", "1h", "2h", "4h", "1d", "1w"],
        period=period,
        output_dir=None,
        use_mt5=True,
        mt5_only=mt5_only,
    )
    df.to_csv(raw_path, index=False)
    logger.info("Saved raw data to %s (%d rows)", raw_path, len(df))

    run_feature_pipeline(
        input_path=raw_path,
        output_path=featured_path,
        rr_ratio=2.0,
        horizon_bars=16,
        pullback_mode=pullback_mode,
        zone_width_atr=zone_width_atr,
    )

    featured_df = pd.read_csv(featured_path)
    featured_df["time"] = pd.to_datetime(featured_df["time"])
    featured_df = featured_df.sort_values("time").reset_index(drop=True)

    train_kwargs = {}
    if n_estimators is not None:
        train_kwargs["n_estimators"] = n_estimators
    if max_depth is not None:
        train_kwargs["max_depth"] = max_depth
    if learning_rate is not None:
        train_kwargs["learning_rate"] = learning_rate

    result = train_model(
        featured_df,
        model_type=model_type,
        test_size=0.2,
        scale_features=True,
        output_dir=str(model_dir),
        **train_kwargs,
    )

    logger.info(
        "%s: train_acc=%.4f, test_acc=%.4f, saved to %s",
        symbol,
        result["train_accuracy"],
        result["test_accuracy"],
        result["model_path"],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train per-symbol HQTS models")
    parser.add_argument(
        "--symbol",
        choices=SYMBOLS,
        default=None,
        help="Train only this symbol (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if model already exists",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Base directory for per-symbol models",
    )
    parser.add_argument(
        "--data-dir",
        default="data/clean",
        help="Directory for raw and featured data",
    )
    parser.add_argument(
        "--period",
        default="1mo",
        help="Data period (default: 1mo; use 1w, 3mo, 1y, 2y, 6mo, 60d). MT5-only.",
    )
    parser.add_argument(
        "--no-mt5-only",
        action="store_true",
        help="Allow yfinance fallback (default: MT5 only for training)",
    )
    parser.add_argument(
        "--no-pullback",
        action="store_true",
        help="Disable pullback-aware labeling (default: pullback enabled)",
    )
    parser.add_argument(
        "--zone-width-atr",
        type=float,
        default=0.75,
        help="ATR multiplier for zone width in pullback mode (default: 0.75)",
    )
    parser.add_argument(
        "--finetune-losses",
        action="store_true",
        help="After training, fine-tune on loss samples from data/loss_trades.jsonl",
    )
    parser.add_argument(
        "--model",
        choices=["xgboost", "random_forest"],
        default="xgboost",
        help="Model type (default: xgboost)",
    )
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    models_base = Path(args.models_dir) if Path(args.models_dir).is_absolute() else project_root / args.models_dir
    data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else project_root / args.data_dir
    symbols = [args.symbol] if args.symbol else SYMBOLS

    trained = 0
    skipped = 0
    for sym in symbols:
        try:
            result = train_symbol(
                symbol=sym,
                data_dir=data_dir,
                models_base=models_base,
                period=args.period,
                force=args.force,
                pullback_mode=not args.no_pullback,
                zone_width_atr=args.zone_width_atr,
                model_type=args.model,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                mt5_only=not args.no_mt5_only,
            )
            if result is not None:
                trained += 1
            else:
                skipped += 1

            if args.finetune_losses:
                try:
                    import importlib.util
                    finetune_script = Path(__file__).parent / "train_finetune_losses.py"
                    spec = importlib.util.spec_from_file_location("finetune", finetune_script)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    loss_path = Path(__file__).resolve().parent.parent / "data" / "loss_trades.jsonl"
                    mod.finetune_symbol(sym, models_base, data_dir, loss_path)
                except Exception as fe:
                    logger.debug("Finetune %s: %s", sym, fe)
        except Exception as e:
            logger.exception("Failed to train %s: %s", sym, e)
            sys.exit(1)

    logger.info("Done: %d trained, %d skipped", trained, skipped)


if __name__ == "__main__":
    main()
