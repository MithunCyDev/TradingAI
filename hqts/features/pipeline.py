"""
End-to-end feature and labeling pipeline for HQTS.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from hqts.features.engineering import compute_features
from hqts.features.labeling import compute_labels

logger = logging.getLogger(__name__)


def run_feature_pipeline(
    input_path: str | Path,
    output_path: str | Path | None = None,
    atr_period: int = 14,
    rsi_period: int = 14,
    rr_ratio: float = 2.0,
    horizon_bars: int = 16,
) -> pd.DataFrame:
    """
    Load cleaned OHLCV, compute features and labels, optionally save.

    Args:
        input_path: Path to cleaned CSV.
        output_path: If set, save featured DataFrame here.
        atr_period: ATR period.
        rsi_period: RSI period.
        rr_ratio: Risk-reward ratio for labeling.
        horizon_bars: Lookahead bars for TP/SL labeling.

    Returns:
        DataFrame with features and label column.
    """
    df = pd.read_csv(input_path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    df = compute_features(df, atr_period=atr_period, rsi_period=rsi_period)
    df["label"] = compute_labels(df, rr_ratio=rr_ratio, horizon_bars=horizon_bars)

    # Drop rows with NaN from rolling computations (first ~14+ bars)
    df = df.dropna(subset=["atr", "rsi", "label"]).reset_index(drop=True)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved featured data to %s", output_path)

    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="HQTS feature and labeling pipeline")
    parser.add_argument("input", help="Path to cleaned OHLCV CSV")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path")
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--rr-ratio", type=float, default=2.0)
    parser.add_argument("--horizon-bars", type=int, default=16)
    args = parser.parse_args()

    output = args.output or str(Path(args.input).parent / "featured" / Path(args.input).name)
    run_feature_pipeline(
        args.input,
        output_path=output,
        atr_period=args.atr_period,
        rsi_period=args.rsi_period,
        rr_ratio=args.rr_ratio,
        horizon_bars=args.horizon_bars,
    )


if __name__ == "__main__":
    main()
