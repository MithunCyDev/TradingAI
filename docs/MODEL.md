# ML Model Guide

HQTS uses per-symbol classifiers (XGBoost/RandomForest) trained on OHLCV + SMC features. Each model predicts directional probability: Up, Down, or Range.

## Training

### Train All Symbols

```bash
# Train models for all supported symbols (6 months data, yfinance + MT5 fallback)
python scripts/train_all_symbols.py

# Force retrain even if model exists
python scripts/train_all_symbols.py --force

# Use 1 year of data
python scripts/train_all_symbols.py --period 1y
```

### Supported Symbols

- **Crypto**: BTCUSD
- **Metals**: XAUUSD, XAGUSD
- **Forex**: EURUSD, USDJPY, GBPUSD, AUDUSD, USDCHF
- **Indices/Commodities**: USTECH, USOIL

### Output

Each symbol gets a directory under `models/`:

```
models/
├── btcusd/
│   ├── model.joblib    # Trained classifier
│   ├── scaler.joblib   # Feature scaler
│   └── config.json     # Feature columns, label mapping
├── xauusd/
├── eurusd/
└── ...
```

## Features

Models use 30+ features from `hqts.features.engineering`:

| Category | Features |
|----------|----------|
| Volatility | `atr`, `atr_pct`, `bb_width`, `volatility_regime` |
| Momentum | `rsi`, `vwap_dist` |
| Time | `hour`, `dow`, `session` |
| Structure | `swing_high`, `swing_low`, `is_swing_high`, `is_swing_low` |
| SMC | `fvg_bullish`, `fvg_bearish`, `ob_zone_strength`, `dist_to_demand`, `dist_to_supply`, `in_demand_zone`, `in_supply_zone`, `is_liquidity_sweep_bull`, `is_liquidity_sweep_bear` |
| News | `is_news_window` |

See `hqts.models.config.FEATURE_COLUMNS` for the full list.

## Inference

### Programmatic

```python
from hqts.models.inference import InferenceEngine
import pandas as pd

engine = InferenceEngine(model_dir="models/xauusd")
df = ...  # OHLCV DataFrame with time, open, high, low, close, tick_volume

result = engine.run(df, zone_width_atr=0.5)
# result: {"label": 1, "prob_up": 0.62, "prob_down": 0.22, "prob_range": 0.16}
```

### Labels

| Label | Class | Meaning |
|-------|-------|---------|
| -1 | Down | Bearish |
| 0 | Range | Sideways |
| 1 | Up | Bullish |

## Data Pipeline

1. **Raw data**: OHLCV from MT5 or yfinance (15m, 1h, 4h)
2. **Feature pipeline**: `run_feature_pipeline()` → `compute_features()` + `compute_labels()`
3. **Training**: `train_model()` with RR-based labels
4. **Inference**: `InferenceEngine.run()` → features → predict → probabilities

## Configuration

- **RR ratio**: 2.0 (risk-reward for label computation)
- **Horizon bars**: 16 (forward-looking window for labels)
- **Zone width**: Configurable via `SMC_ZONE_WIDTH_ATR` for supply/demand zones

See [CONFIGURATION.md](CONFIGURATION.md) for environment variables.
