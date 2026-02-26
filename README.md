# Hybrid Quant Trading System (HQTS)

A data-driven algorithmic trading system combining custom ML forecasting with Smart Money Concepts (SMC) execution for MetaTrader 5.

**Platform: Windows** (MetaTrader5 requires Windows)

## Documentation

| Document | Description |
|----------|-------------|
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/API.md](docs/API.md) | REST API for predictions |
| [docs/MODEL.md](docs/MODEL.md) | ML model training and inference |
| [docs/AUTO_TRADER.md](docs/AUTO_TRADER.md) | Automated trading script |
| [docs/SMC.md](docs/SMC.md) | Smart Money Concepts filter |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Environment variables |
| [docs/MT5_SETUP.md](docs/MT5_SETUP.md) | MetaTrader 5 setup |
| [docs/MARKET_HOURS.md](docs/MARKET_HOURS.md) | Market hours filter |

## Target Assets

- Spot Metals: XAUUSD, XAGUSD
- Major Forex pairs
- Crypto: BTCUSD
- Indices/Commodities: USTECH, USOIL

## Architecture

- **Data Layer**: MT5 Terminal (yfinance fallback)
- **Intelligence Engine**: Custom ML models (XGBoost/RandomForest)
- **Execution**: SMC-based rule engine with risk controls
- **Market Hours**: Blocks trading when Forex/metals are closed (weekend)

## Setup (Windows)

1. Python 3.9, 3.10, or 3.11 (not 3.12+)
2. Create venv: `python -m venv .venv` then `.venv\Scripts\activate`
3. Install: `pip install -r requirements.txt`
4. Install MetaTrader 5 terminal and log in (see [docs/MT5_SETUP.md](docs/MT5_SETUP.md))

## Quick Start

```bash
# Train models for all symbols
python scripts/train_all_symbols.py

# Run auto trader (paper mode)
python scripts/run_auto_trader.py --paper

# Start prediction API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Key Commands

| Command | Description |
|---------|-------------|
| `python scripts/train_all_symbols.py` | Train per-symbol models |
| `python scripts/run_auto_trader.py --paper` | Paper trade (no real orders) |
| `python scripts/run_auto_trader.py` | Live trading via MT5 |
| `uvicorn api.main:app` | Start prediction API |
| `python -m hqts.etl.extract --symbol XAUUSD --count 100000` | Extract MT5 data |

## Project Structure

```
TradingAI/
├── api/            # FastAPI prediction endpoints
├── data/           # Raw and cleaned datasets
├── docs/           # Documentation
├── hqts/           # Main package
│   ├── etl/        # Data extraction and cleaning
│   ├── features/   # Feature engineering and labeling
│   ├── models/     # ML training and inference
│   └── execution/  # Order execution, risk, SMC filters
├── logs/           # Prediction and trade logs
├── models/         # Persisted models per symbol
└── scripts/        # CLI entry points
```
