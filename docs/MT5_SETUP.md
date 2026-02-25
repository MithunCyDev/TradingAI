# MetaTrader 5 Setup (Windows)

## Requirements

- **Windows** (MetaTrader5 is Windows-only)
- **Python**: 3.9, 3.10, or 3.11 (not 3.12+)
- **MetaTrader 5**: Terminal installed and running
- **Account**: Logged in (demo or live)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Live BTC Trading

1. Open MetaTrader 5 and log in.
2. Ensure BTC symbol is in Market Watch (right-click → Show All, or add BTCUSD).
3. Run:

```bash
python scripts/run_live_btc.py --paper   # Paper trade first
python scripts/run_live_btc.py           # Real trading
```

## Symbol Names

Brokers use different names for Bitcoin:
- `BTCUSD`
- `BTCUSDm`
- `BTCUSD.a`

The script auto-detects the first available BTC symbol. Override with `--symbol BTCUSDm`.
