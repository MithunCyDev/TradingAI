# Auto Trader

Automated trading script that runs every 3 minutes: fetches data, runs ML inference, applies SMC filter, and places orders via MT5 when signals meet threshold.

## Quick Start

```bash
# Paper trade (no real orders)
python scripts/run_auto_trader.py --paper

# Live trading (real orders)
python scripts/run_auto_trader.py
```

**Requirements**: MT5 terminal running and logged in (see [MT5_SETUP.md](MT5_SETUP.md)).

## Flow

1. **Fetch data** – MT5 first, yfinance fallback (500 bars, 15m)
2. **Inference** – Per-symbol model predicts Up/Down/Range
3. **SMC filter** – Validates order block, FVG, or liquidity sweep
4. **Market hours** – Skips when Forex/metals closed (crypto trades 24/7)
5. **Spread check** – Skips if broker spread exceeds symbol limit
6. **Position check** – Skips if already have open position in symbol
7. **Execute** – Places market order with ATR-based SL/TP

## Symbols

| Symbol | Model Dir | Notes |
|--------|-----------|-------|
| BTCUSD | btcusd | 24/7, no SMC order block required |
| XAUUSD | xauusd | Gold |
| XAGUSD | xagusd | Silver |
| EURUSD | eurusd | |
| USDJPY | usdjpy | |
| GBPUSD | gbpusd | |
| AUDUSD | audusd | |
| USDCHF | usdchf | |
| USTECH | ustech | Nasdaq 100 |
| USOIL | usoil | WTI Crude |

## Configuration

| Env Variable | Default | Description |
|--------------|---------|-------------|
| `TRADE_PROB_THRESHOLD` | 0.6 | Min probability for trade (e.g. 0.4 = 40%) |
| `SMC_REQUIRE_ORDER_BLOCK` | false | Require order block (crypto: always false) |
| `SMC_REQUIRE_FVG` | false | Require Fair Value Gap |
| `SMC_REQUIRE_LIQUIDITY_SWEEP` | false | Require liquidity sweep |
| `MAX_SPREAD_POINTS` | 0 | Override per-symbol spread limit |

Per-symbol spread limits in code: XAUUSD 50, EURUSD 25, USDCHF 30, etc. Only rejects when spread is **too high**; smaller spreads are accepted.

## Logs

- **Console**: Colored predictions and trade actions
- **logs/auto_trader.log**: Full log output
- **logs/predictions_live.jsonl**: Prediction records
- **logs/trades_live.jsonl**: Trade execution records

## Market Hours

- **Forex/Metals**: No trading Fri 21:00 UTC – Sun 21:00 UTC
- **Crypto (BTCUSD)**: 24/7

See [MARKET_HOURS.md](MARKET_HOURS.md) for details.
