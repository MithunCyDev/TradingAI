# Configuration Guide

HQTS uses environment variables (`.env`) for configuration. Copy `.env.example` to `.env` and adjust.

## API

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | 0.0.0.0 | Bind address for API server |
| `API_PORT` | 8000 | Port |

## Models & Data

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_BASE_DIR` | models | Directory for per-symbol models |
| `DATA_DIR` | data/clean | Cleaned datasets |
| `YFINANCE_PERIOD` | 60d | Period for yfinance fallback (60d, 6mo, 1y) |

## Trading

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADE_PROB_THRESHOLD` | 0.6 | Min probability (0–1) to trigger trade. 0.4 = 40% |

## SMC (Smart Money Concepts)

| Variable | Default | Description |
|----------|---------|-------------|
| `SMC_REQUIRE_ORDER_BLOCK` | true | Require order block for entry |
| `SMC_REQUIRE_FVG` | false | Require Fair Value Gap |
| `SMC_REQUIRE_LIQUIDITY_SWEEP` | false | Require liquidity sweep |
| `SMC_OB_LOOKBACK_BARS` | 20 | Bars scanned for OB/FVG/sweep |
| `SMC_ZONE_WIDTH_ATR` | 0.5 | ATR multiplier for demand/supply zone width |
| `SMC_FVG_MIN_SIZE_ATR` | 0.3 | Min FVG size in ATR (filter noise) |
| `SMC_MIN_OB_STRENGTH` | 0.0 | Min order block strength (0 = off) |
| `SMC_REQUIRE_PRICE_IN_ZONE` | false | Require price inside zone for entry |

## MT5

| Variable | Default | Description |
|----------|---------|-------------|
| `MT5_ENABLED` | true | Use MT5 for data when available |

## Other

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_SPREAD_POINTS` | 0 | Override per-symbol spread limit (0 = use defaults) |

## Example .env

```env
# API
API_HOST=0.0.0.0
API_PORT=8000

# Models
MODELS_BASE_DIR=models

# Trading
TRADE_PROB_THRESHOLD=0.55

# SMC
SMC_REQUIRE_ORDER_BLOCK=false
SMC_OB_LOOKBACK_BARS=20

# MT5
MT5_ENABLED=true
```
