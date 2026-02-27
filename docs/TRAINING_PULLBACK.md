# Training Pullback-Aware Models

This guide explains how to train HQTS models to recognize and trade pullback setups—entries where price retraces into a key zone (demand or supply) before continuing in the trend direction.

## What Is a Pullback?

A **pullback** is a temporary retracement against the prevailing trend:

- **Bullish pullback**: Price moves up, then retraces down into a demand zone (near swing low) before resuming higher
- **Bearish pullback**: Price moves down, then retraces up into a supply zone (near swing high) before resuming lower

Pullbacks offer higher-probability entries because:
- Price is at a tested support/resistance level
- Risk is often lower (tighter stop below swing low or above swing high)
- The trend has already been established

## How Pullback Training Works

### Standard vs Pullback Labeling

**Standard labeling** (`compute_labels`):
- Labels every bar based on whether TP or SL is hit first in the next N bars
- No distinction between breakout entries and pullback entries

**Pullback labeling** (`compute_labels_pullback`):
- **LABEL_UP (Buy)**: Only when price is **in** a demand zone AND TP hits before SL
- **LABEL_DOWN (Sell)**: Only when price is **in** a supply zone AND SL hits before TP (long fails = short wins)
- **LABEL_RANGE**: All other bars (including zones where the trade failed)

This trains the model to recognize: *"When I'm in a pullback zone, what happens next?"*

### New Features

- **`pullback_bull`**: Price is in demand zone and has pulled back (close < close 2 bars ago)
- **`pullback_bear`**: Price is in supply zone and has pulled back (close > close 2 bars ago)
- **`in_demand_zone`**, **`in_supply_zone`**: Existing zone features (price within zone_width_atr of swing low/high)

## Training Commands

### Single Symbol (USTECH)

```bash
cd scripts
python train_all_symbols.py --symbol USTECH --pullback --force --period 1y
```

### All Symbols

```bash
python train_all_symbols.py --pullback --force --period 1y
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--pullback` | off | Enable pullback-aware labeling |
| `--zone-width-atr` | 0.75 | ATR multiplier for zone width (larger = wider zone) |
| `--period` | 6mo | Data period (1y, 6mo, 60d) |
| `--force` | off | Retrain even if model exists |

### Pipeline (CLI)

```bash
python -m hqts.features.pipeline data/clean/ustech_1y.csv -o data/clean/ustech_1y_pullback_featured.csv --pullback --zone-width-atr 0.75
```

## Expected Behavior

- **Class distribution**: With pullback mode, labels are heavily skewed toward RANGE (0) because only bars in zones get UP/DOWN labels. This is intentional.
- **Model focus**: The model learns to predict UP when it sees pullback features (in_demand_zone, pullback_bull, etc.) and vice versa for DOWN.
- **Auto trader**: Use the trained model as usual—it will output higher probabilities for UP when the current context resembles a pullback into demand.

## Tuning

- **zone_width_atr**: 0.5 = stricter (narrower zones), 1.0 = looser (wider zones)
- **horizon_bars**: 16 bars (default) lookahead for TP/SL
- **rr_ratio**: 2.0 (1:2 risk-reward) for labeling

## Files Modified

- `hqts/features/engineering.py`: Added `pullback_bull`, `pullback_bear`
- `hqts/features/labeling.py`: Added `compute_labels_pullback`
- `hqts/features/pipeline.py`: Added `pullback_mode`, `zone_width_atr`
- `hqts/models/config.py`: Added pullback features to `FEATURE_COLUMNS`
- `scripts/train_all_symbols.py`: Added `--pullback`, `--zone-width-atr`
