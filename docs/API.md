# HQTS Prediction API

REST API for ML-based directional predictions. Fetches real-time candle data from MT5 (yfinance fallback) and returns probabilities for Up, Down, and Range.

## Quick Start

```bash
# Start API server (ensure MT5 terminal is running for broker data)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Or with reload for development
uvicorn api.main:app --reload
```

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

| Endpoint | Symbol | Description |
|----------|--------|-------------|
| `GET /api/health` | — | Health check |
| `GET /api/status` | — | MT5 connection status and data source |
| `GET /api/symbols` | — | List supported symbols |
| `GET /api/predict/btc` | BTCUSD | Bitcoin prediction |
| `GET /api/predict/gold` | XAUUSD | Gold (XAU/USD) prediction |
| `GET /api/predict/silver` | XAGUSD | Silver (XAG/USD) prediction |
| `GET /api/predict/eurusd` | EURUSD | EUR/USD prediction |
| `GET /api/predict/usdjpy` | USDJPY | USD/JPY prediction |
| `GET /api/predict/gbpusd` | GBPUSD | GBP/USD prediction |
| `GET /api/predict/audusd` | AUDUSD | AUD/USD prediction |
| `GET /api/predict/usdchf` | USDCHF | USD/CHF prediction |

## Prediction Response

```json
{
  "symbol": "XAUUSD",
  "label": 1,
  "direction": "up",
  "prob_up": 0.62,
  "prob_down": 0.22,
  "prob_range": 0.16,
  "timeframe": "15m",
  "last_close": 2650.5,
  "position": "buy",
  "take_profit": 2665.2,
  "stop_loss": 2635.8,
  "lot_size": 0.01,
  "data_fetched_at": "2026-02-26T20:00:00.000000+00:00",
  "data_source": "MT5"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `label` | int | -1 (down), 0 (range), 1 (up) |
| `direction` | str | `"up"`, `"down"`, or `"range"` |
| `prob_up` / `prob_down` / `prob_range` | float | Class probabilities |
| `position` | str | `"buy"`, `"sell"`, or `"hold"` |
| `take_profit` / `stop_loss` | float | Set when position is buy/sell and prob meets threshold |
| `lot_size` | float | Calculated from equity and risk |
| `data_source` | str | `"MT5"` or `"yfinance"` |

## Query Parameters

**Prediction endpoints** accept:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeframe` | str | `"15m"` | Bar size (15m, 1h) |
| `equity` | float | 1000 | Account balance for lot size calculation |

Example:

```
GET /api/predict/gold?timeframe=15m&equity=10000
```

## Data Source

- **MT5**: Real-time broker data. Use when API runs on same machine as MT5 terminal.
- **yfinance**: Fallback when MT5 is unavailable. Predictions may differ from `run_auto_trader`.

Check `GET /api/status` to verify MT5 connection. For consistent results with the auto trader, ensure `mt5_initialized: true`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | 0.0.0.0 | Bind address |
| `API_PORT` | 8000 | Port |
| `MODELS_BASE_DIR` | models | Path to trained models |
| `TRADE_PROB_THRESHOLD` | 0.6 | Min probability for buy/sell signal |
| `MT5_ENABLED` | true | Use MT5 for data when available |
| `SMC_ZONE_WIDTH_ATR` | 0.5 | Zone width for feature computation |

See [CONFIGURATION.md](CONFIGURATION.md) for full list.
