@echo off
REM Run live BTC trading (Windows)
REM Ensure MT5 is running and logged in

call .venv\Scripts\activate.bat
python scripts/run_live_btc.py %*
