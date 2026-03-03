#!/usr/bin/env python3
"""Quick diagnostic: run inference on sample bars and print prob distribution."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from hqts.etl.clean import clean_and_validate
from hqts.etl.mt5_live import fetch_data_mt5_first
from hqts.models.inference import InferenceEngine
from hqts.execution.config import ExecutionConfig

def main():
    symbol = "BTCUSD"
    model_dir = Path(__file__).resolve().parent.parent / os.getenv("MODELS_BASE_DIR", "models") / "btcusd"
    if not (model_dir / "model.joblib").exists():
        print("Model not found at", model_dir)
        return

    print("Fetching data...")
    df, _ = fetch_data_mt5_first(symbol, timeframe="15m", count=2000)
    df = clean_and_validate(df)
    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    print(f"Fetched {len(df)} bars")

    engine = InferenceEngine(model_dir=model_dir)
    exec_config = ExecutionConfig()
    buf_size = 500

    prob_ups, prob_downs = [], []
    for i in range(buf_size, min(buf_size + 300, len(df) - 16)):
        buffer = df.iloc[i - buf_size : i].copy()
        try:
            r = engine.run(buffer, zone_width_atr=exec_config.smc.zone_width_atr)
            prob_ups.append(r["prob_up"])
            prob_downs.append(r["prob_down"])
        except Exception as e:
            print(f"Bar {i}: {e}")

    prob_ups = np.array(prob_ups)
    prob_downs = np.array(prob_downs)

    print("\n--- Prob distribution (sample of 300 bars) ---")
    print(f"prob_up:  min={prob_ups.min():.3f} max={prob_ups.max():.3f} mean={prob_ups.mean():.3f}")
    print(f"prob_down: min={prob_downs.min():.3f} max={prob_downs.max():.3f} mean={prob_downs.mean():.3f}")

    for th in [0.5, 0.4, 0.35, 0.33]:
        buys = np.sum((prob_ups >= th) & (prob_ups > prob_downs))
        sells = np.sum((prob_downs >= th) & (prob_downs > prob_ups))
        print(f"Threshold {th}: would trigger buy={buys} sell={sells} total={buys+sells}")

    print("\nSample (first 10): prob_up, prob_down")
    for j in range(min(10, len(prob_ups))):
        print(f"  {prob_ups[j]:.3f}, {prob_downs[j]:.3f}")

if __name__ == "__main__":
    main()
