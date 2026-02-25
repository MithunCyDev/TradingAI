# Market Hours Guide

HQTS includes a **Market Hours Filter** that blocks new trades when markets are closed.

## Default Behavior (Forex / Spot Metals)

For XAUUSD, XAGUSD, and major Forex pairs:

- **Closed**: Friday 21:00 UTC until Sunday 21:00 UTC
- **Open**: Sunday 21:00 UTC until Friday 21:00 UTC

Times are in **UTC** to align with MT5 and data feeds.

## Configuration

In `ExecutionConfig.market_hours`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `True` | Enable/disable the filter |
| `weekend_closed` | `True` | Apply weekend closure rules |
| `friday_close_utc_hour` | `21` | Hour (0–23) when trading stops on Friday |
| `sunday_open_utc_hour` | `21` | Hour (0–23) when trading resumes on Sunday |
| `trading_start_utc_hour` | `None` | Optional daily start hour; `None` = no limit |
| `trading_end_utc_hour` | `None` | Optional daily end hour; `None` = no limit |

## Crypto (24/7)

For BTCUSD and other crypto, markets never close. Disable weekend closure:

```python
config = ExecutionConfig()
config.market_hours.weekend_closed = False
```

Or disable the filter entirely:

```python
config.market_hours.enabled = False
```

## Restrict to Session Hours

To trade only during London + New York overlap (e.g. 13:00–21:00 UTC):

```python
config.market_hours.trading_start_utc_hour = 13
config.market_hours.trading_end_utc_hour = 21
```

## Broker-Specific Times

Brokers may use different weekend close/open times. Adjust `friday_close_utc_hour` and `sunday_open_utc_hour` to match your broker’s schedule.
