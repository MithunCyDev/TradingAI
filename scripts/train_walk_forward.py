#!/usr/bin/env python3
"""
Walk-forward training for HQTS.

Uses config/settings.yaml for configuration. Fetches data via Dukascopy (or fallback),
runs triple-barrier labeling, meta-labeling, regime awareness, and hyperparameter
optimization with strict out-of-sample validation.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.loader import get_config
from hqts.etl.dukascopy_fetch import DUKASCOPY_UNSUPPORTED, fetch_dukascopy_multi_timeframe
from hqts.models.walk_forward import (
    WalkForwardConfig,
    generate_walk_forward_folds,
    run_walk_forward_training,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _fetch_data_fallback(symbol: str, intervals: list[str], start_year: int, end_year: int) -> "pd.DataFrame":
    """Fallback to yfinance/MT5 when Dukascopy unavailable for symbol."""
    import pandas as pd

    from hqts.etl.yfinance_fetch import fetch_multi_symbol_multi_timeframe

    years = end_year - start_year + 1
    period = f"{years}y" if years >= 1 else "1y"
    try:
        df = fetch_multi_symbol_multi_timeframe(
            symbols=[symbol],
            intervals=intervals,
            period=period,
            output_dir=None,
            use_mt5=True,
            mt5_only=False,
        )
        return df
    except Exception as e:
        logger.warning("Fallback fetch failed for %s: %s", symbol, e)
        return pd.DataFrame()


def main() -> None:
    config = get_config()
    project_root = Path(__file__).resolve().parent.parent

    models_dir = Path(config.paths.models_dir)
    if not models_dir.is_absolute():
        models_dir = project_root / models_dir
    data_dir = Path(config.paths.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    data_dir.mkdir(parents=True, exist_ok=True)

    symbols = config.data.symbols
    timeframes = config.data.timeframes
    start_year = config.data.start_year
    end_year = config.data.end_year
    data_source = config.data.source

    wf = config.training.walk_forward
    tb = config.training.triple_barrier

    wf_config = WalkForwardConfig(
        train_years=wf.train_years,
        test_years=wf.test_years,
        step_years=wf.step_years,
        mode=wf.mode,
        atr_mult_sl=tb.atr_mult_sl,
        rr_ratio=tb.rr_ratio,
        horizon_bars=tb.horizon_bars,
        vertical_barrier_bars=tb.vertical_barrier_bars,
        meta_labeling=config.training.meta_labeling,
        regime_aware=config.training.regime_aware,
        hyperopt=config.training.hyperopt,
        hyperopt_trials=config.training.hyperopt_trials,
    )

    folds = generate_walk_forward_folds(
        start_year=start_year,
        end_year=end_year,
        train_years=wf.train_years,
        test_years=wf.test_years,
        step_years=wf.step_years,
        mode=wf.mode,
    )
    logger.info("Generated %d walk-forward folds", len(folds))

    start_dt = datetime(start_year, 1, 1, tzinfo=timezone.utc)
    end_dt = datetime(end_year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    for symbol in symbols:
        try:
            logger.info("Training %s...", symbol)

            if data_source == "dukascopy" and symbol.upper() not in DUKASCOPY_UNSUPPORTED:
                try:
                    df = fetch_dukascopy_multi_timeframe(
                        symbol=symbol,
                        intervals=timeframes,
                        start=start_dt,
                        end=end_dt,
                    )
                except Exception as e:
                    logger.warning("Dukascopy failed for %s: %s; trying fallback", symbol, e)
                    df = _fetch_data_fallback(symbol, timeframes, start_year, end_year)
                if df.empty:
                    logger.info("Dukascopy returned no data for %s; trying fallback", symbol)
                    df = _fetch_data_fallback(symbol, timeframes, start_year, end_year)
            else:
                df = _fetch_data_fallback(symbol, timeframes, start_year, end_year)

            if df.empty:
                logger.warning("No data for %s, skipping", symbol)
                continue

            raw_path = data_dir / f"{symbol.lower()}_wf_raw.csv"
            df.to_csv(raw_path, index=False)
            logger.info("Saved raw data to %s (%d rows)", raw_path, len(df))

            model_dir = models_dir / symbol.lower()
            results = run_walk_forward_training(
                df=df,
                folds=folds,
                config=wf_config,
                output_dir=model_dir,
                save_last_fold_only=True,
            )

            logger.info(
                "%s: mean test accuracy=%.4f, folds=%d",
                symbol,
                results.get("mean_test_accuracy", 0.0),
                len(results.get("folds", [])),
            )

        except Exception as e:
            logger.exception("Failed to train %s: %s", symbol, e)
            sys.exit(1)

    logger.info("Walk-forward training complete")


if __name__ == "__main__":
    main()
