"""
Economic calendar integration for HQTS.

Fetches upcoming economic events from MT5 desktop terminal.
Requires ExportCalendarEA.mq5 attached to a chart in MT5.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Currency to country mapping for event filtering
CURRENCY_TO_COUNTRY = {
    "USD": "US",
    "EUR": "EU",
    "GBP": "GB",
    "JPY": "JP",
    "AUD": "AU",
    "CHF": "CH",
    "XAU": "US",  # Gold often tied to USD
}

# Max age of MT5 calendar file (seconds) before considered stale
MT5_CALENDAR_MAX_AGE_SEC = 600  # 10 minutes


def _get_mt5_calendar_path() -> Optional[Path]:
    """Get path to economic_calendar.csv in MT5 Files folder. Returns None if MT5 not available."""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return None
        ti = mt5.terminal_info()
        if ti is None:
            return None
        data_path = getattr(ti, "data_path", None)
        if not data_path:
            return None
        return Path(data_path) / "MQL5" / "Files" / "economic_calendar.csv"
    except Exception:
        return None


def _read_mt5_calendar(path: Path) -> list[dict]:
    """Parse MT5-exported economic_calendar.csv into event dicts."""
    events = []
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_str = row.get("time_utc", "").strip()
                if not ts_str:
                    continue
                try:
                    ts = int(ts_str)
                    event_time = datetime.fromtimestamp(ts, tz=timezone.utc)
                except (ValueError, TypeError):
                    continue
                impact = (row.get("importance", "") or "medium").lower()
                currency = (row.get("currency", "") or "").strip()
                country = (row.get("country", "") or "").strip()
                title = (row.get("title", "") or "").strip()
                events.append({
                    "time": event_time,
                    "impact": impact,
                    "currency": currency,
                    "country": country,
                    "title": title,
                })
    except Exception as ex:
        logger.warning("Failed to read MT5 calendar file %s: %s", path, ex)
    return events


def fetch_upcoming_events(
    from_dt: Optional[datetime] = None,
    to_dt: Optional[datetime] = None,
    currencies: Optional[list[str]] = None,
    high_impact_only: bool = True,
) -> list[dict]:
    """
    Fetch economic calendar events from MT5 terminal.

    Requires ExportCalendarEA.mq5 attached to a chart in MT5 (exports to MQL5/Files/).

    Args:
        from_dt: Start of range (UTC).
        to_dt: End of range (UTC).
        currencies: Filter by currency (e.g. ["USD", "EUR"]). None = all.
        high_impact_only: If True, only return high-impact events.

    Returns:
        List of dicts with keys: time (datetime), impact (str), currency, country, title.
    """
    from_dt = from_dt or datetime.now(timezone.utc)
    to_dt = to_dt or from_dt + timedelta(days=1)
    currencies = currencies or ["USD", "EUR", "GBP", "JPY", "AUD", "CHF"]

    path = _get_mt5_calendar_path()
    if path is None or not path.exists():
        logger.debug("MT5 calendar file not found; ensure ExportCalendarEA is attached to a chart")
        return []

    try:
        mtime = path.stat().st_mtime
        if (datetime.now(timezone.utc).timestamp() - mtime) > MT5_CALENDAR_MAX_AGE_SEC:
            logger.debug("MT5 calendar file stale (>%d s); re-run ExportCalendarEA", MT5_CALENDAR_MAX_AGE_SEC)
    except OSError:
        return []

    events = _read_mt5_calendar(path)
    filtered = []
    for e in events:
        if high_impact_only and e.get("impact") != "high":
            continue
        if currencies:
            ccy = e.get("currency", "")
            if ccy and ccy not in currencies:
                continue
        t = e.get("time")
        if t and from_dt <= t <= to_dt:
            filtered.append(e)
    return filtered


def is_in_news_window(
    bar_time: datetime,
    events: list[dict],
    minutes_before: int = 30,
    minutes_after: int = 30,
) -> bool:
    """
    Check if a bar's time falls within a news event window.

    Args:
        bar_time: Bar timestamp (timezone-aware).
        events: List from fetch_upcoming_events.
        minutes_before: Window start before event.
        minutes_after: Window end after event.

    Returns:
        True if bar_time is within any event's window.
    """
    if not events:
        return False
    for e in events:
        t = e.get("time")
        if not t or not hasattr(t, "timestamp"):
            continue
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        start = t - timedelta(minutes=minutes_before)
        end = t + timedelta(minutes=minutes_after)
        if start <= bar_time <= end:
            return True
    return False
