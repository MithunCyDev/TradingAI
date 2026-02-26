# Hybrid Quant Trading System (HQTS)

A data-driven algorithmic trading system combining custom ML forecasting with Smart Money Concepts (SMC) execution for MetaTrader 5.

**Platform: Windows** (MetaTrader5 requires Windows)

## Target Assets

- Spot Metals: XAUUSD, XAGUSD
- Major Forex pairs
- Crypto: BTCUSD

## Architecture

- **Data Layer**: MT5 Terminal (tick + OHLCV)
- **Intelligence Engine**: Custom ML models (XGBoost/RandomForest)
- **Execution**: SMC-based rule engine with risk controls
- **Market Hours**: Blocks trading when Forex/metals are closed (weekend)

## Setup (Windows)

1. Python 3.9, 3.10, or 3.11 (not 3.12+)
2. Create venv: `python -m venv .venv` then `.venv\Scripts\activate`
3. Install: `pip install -r requirements.txt`
4. Install MetaTrader 5 terminal and log in

## Quick Start

```bash
# Generate sample data
python scripts/generate_sample_data.py

# Run feature pipeline
python -m hqts.features.pipeline data/clean/XAUUSD_M15_sample.csv -o data/clean/XAUUSD_M15_featured.csv

# Train model
python -m hqts.models.train data/clean/XAUUSD_M15_featured.csv -o models --model random_forest

# Run bot (paper mode)
python scripts/run_bot.py
```

## Fine-tune with Live Data (BTCUSD + XAUUSD, 15m/1h/4h)

```bash
# Fetch 2 months of BTCUSD and XAUUSD across 15m, 1h, 4h timeframes, then train
python scripts/fetch_and_finetune.py
```

## Live Data (MT5)

```bash
# Ensure MT5 terminal is running, then:
python -m hqts.etl.extract --symbol XAUUSD --timeframe M15 --count 100000
```

## Live BTC Trading

```bash


# Custom symbol or timeframe
python scripts/run_live_btc.py --symbol BTCUSDm --timeframe H1 --paper
```

## Project Structure

```
TradingAI/
├── data/           # Raw and cleaned datasets
├── hqts/           # Main package
│   ├── etl/        # Data extraction and cleaning
│   ├── features/   # Feature engineering and labeling
│   ├── models/     # ML training and inference
│   ├── execution/  # Order execution, risk, SMC filters
│   └── logging/    # Structured logging and reporting
├── logs/           # Prediction and trade logs
├── models/         # Persisted models and config
└── scripts/        # CLI entry points
```
