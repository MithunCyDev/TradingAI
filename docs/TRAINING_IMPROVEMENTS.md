# Training Improvements Guide

This guide explains the improvements made to improve model win rate and SMC filter flexibility.

## ⚠️ Settings That Can Cause 0 Trades

Avoid these combinations—they often result in no trades:
- `TRADE_PROB_THRESHOLD` ≥ 0.65 (model rarely outputs that high)
- `TRADE_PROB_MARGIN` > 0.1 (requires large prob gap)
- `SMC_REQUIRE_ORDER_BLOCK=true` (blocks when no OB/FVG/sweep)
- `zone_width_atr` ≤ 0.5 (stricter labels → model predicts RANGE more)

**Working defaults:** threshold=0.5, margin=0, SMC_REQUIRE_ORDER_BLOCK=false, zone_width_atr=0.75

## Model Training Improvements

### Pullback Mode (Default)

Training now uses pullback-aware labeling by default. Labels are assigned only when price is in a demand/supply zone, which improves signal quality.

```bash
# Default: pullback + 1y data
python scripts/train_all_symbols.py --force

# Disable pullback
python scripts/train_all_symbols.py --no-pullback --force
```

### Class Weights

Balanced class weights are applied automatically when training. This helps the model learn from minority classes (UP/DOWN) when RANGE dominates.

### XGBoost Default

Models are trained with XGBoost by default. Use `--model random_forest` to switch.

### Hyperparameters

Default XGBoost settings: `n_estimators=300`, `max_depth=5`, `learning_rate=0.03` (tuned to reduce overfitting).

Override via CLI:

```bash
python scripts/train_all_symbols.py --n-estimators 400 --max-depth 6 --force
```

### Recommended Training Command

```bash
# MT5-only, 1y data (default)
python scripts/train_all_symbols.py --force

# 2 years of MT5 data
python scripts/train_all_symbols.py --period 2y --force

# Allow yfinance fallback if MT5 fails
python scripts/train_all_symbols.py --no-mt5-only --force
```

(Pullback, 1y period, and MT5-only are now defaults.)

## SMC Filter Relaxation

### Relaxed Defaults

| Variable                      | Old  | New   |
| ----------------------------- | ---- | ----- |
| `SMC_REQUIRE_FVG`             | true | false |
| `SMC_REQUIRE_LIQUIDITY_SWEEP` | true | false |
| `SMC_REQUIRE_PRICE_IN_ZONE`   | true | false |
| `SMC_OB_LOOKBACK_BARS`        | 20   | 30    |
| `SMC_ZONE_WIDTH_ATR`          | 0.5  | 0.6   |

Only order block is required by default. More trades pass the filter.

### OR Logic (Optional)

Set `SMC_REQUIRE_ANY=true` in `.env` to pass when **any** of OB/FVG/sweep is present (OR logic). When false, all required conditions must pass (AND logic).

## Validation

After retraining:

1. **Model-only** (default): `python scripts/run_backtest.py`
2. **Model + SMC**: `python scripts/run_backtest.py --use-smc` or `BACKTEST_USE_SMC=true`
3. Results in `logs/backtest_results/{SYMBOL}_{YYYY-MM}.txt`
