# HQTS Documentation

Documentation for the Hybrid Quant Trading System.

## Documentation Index

| Document | Description |
|----------|-------------|
| [API](API.md) | REST API for predictions, endpoints, and data source |
| [MODEL](MODEL.md) | ML model training, features, and inference |
| [AUTO_TRADER](AUTO_TRADER.md) | Automated trading script and flow |
| [SMC](SMC.md) | Smart Money Concepts filter (OB, FVG, liquidity sweep) |
| [CONFIGURATION](CONFIGURATION.md) | Environment variables and `.env` |
| [MT5_SETUP](MT5_SETUP.md) | MetaTrader 5 setup and symbol names |
| [MARKET_HOURS](MARKET_HOURS.md) | Market hours filter and session config |

## Quick Links

- **Start API**: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
- **Train models**: `python scripts/train_all_symbols.py`
- **Run auto trader**: `python scripts/run_auto_trader.py --paper`
