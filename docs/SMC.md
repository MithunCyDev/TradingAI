# Smart Money Concepts (SMC) Filter

SMC filter validates trade setups against order blocks, Fair Value Gaps, and liquidity sweeps. Used as Phase 2 after ML signal.

## Overview

- **Order Block (OB)**: Last down candle before strong up move (bullish) or last up candle before strong down move (bearish)
- **Fair Value Gap (FVG)**: Price gap between three consecutive bars
- **Liquidity Sweep**: Price sweeps swing high/low then reverses

## Configuration

See [CONFIGURATION.md](CONFIGURATION.md) for env variables. Key settings:

| Variable | Description |
|----------|-------------|
| `SMC_REQUIRE_ORDER_BLOCK` | Require OB for entry (crypto: always false) |
| `SMC_REQUIRE_FVG` | Require FVG |
| `SMC_REQUIRE_LIQUIDITY_SWEEP` | Require liquidity sweep |
| `SMC_OB_LOOKBACK_BARS` | Bars scanned (default 20) |
| `SMC_ZONE_WIDTH_ATR` | Demand/supply zone width in ATR |
| `SMC_FVG_MIN_SIZE_ATR` | Min FVG size to filter noise |
| `SMC_MIN_OB_STRENGTH` | Min OB strength (0 = off) |
| `SMC_REQUIRE_PRICE_IN_ZONE` | Require price inside zone |

## Logic

- **BUY**: Requires bullish OB, FVG, or liquidity sweep in lookback; optionally price in demand zone
- **SELL**: Requires bearish OB, FVG, or liquidity sweep in lookback; optionally price in supply zone

When `require_order_block=true`, at least one OB must be present. When `require_fvg` or `require_liquidity_sweep` is true, those must also be present. Otherwise, any one of OB/FVG/sweep is sufficient.
