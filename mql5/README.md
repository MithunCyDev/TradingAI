# MT5 Economic Calendar Export

The economic calendar is now sourced from the MT5 desktop terminal. Python reads events from a CSV file exported by this EA.

## Setup

1. Copy `ExportCalendarEA.mq5` to your MT5 `Experts` folder:
   - Typically: `C:\Users\<you>\AppData\Roaming\MetaQuotes\Terminal\<id>\MQL5\Experts\`

2. Open MetaEditor, compile the EA (F7).

3. In MT5, attach the EA to any chart (e.g. EURUSD M15).

4. The EA exports `economic_calendar.csv` to `MQL5/Files/` every 5 minutes.

5. Ensure MT5 terminal is running and logged in when using the auto-trader.

## EA Inputs

- **ExportIntervalMinutes** (default: 5) – How often to export.
- **HighImpactOnly** (default: true) – Export only high-impact events.

## Notes

- The calendar uses MT5's built-in economic calendar data (same as the terminal's Economic Calendar tool).
- If the file is missing or stale (>10 min), the auto-trader will use no news filter for that cycle.
