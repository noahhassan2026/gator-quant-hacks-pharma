import os
import time
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ET = ZoneInfo("America/New_York")
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)


def fetch_minute_bars(ticker, start, end, api_key, adjusted=False):
    """1-minute bars from Massive between two tz-aware datetimes. Index is US/Eastern."""
    url = (
        f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/minute/"
        f"{int(start.timestamp() * 1000)}/{int(end.timestamp() * 1000)}"
    )
    # adjusted=False keeps prices on the same raw basis as the Databento daily closes;
    # pass adjusted=True when comparing against split-adjusted daily bars
    params = {"adjusted": "true" if adjusted else "false", "sort": "asc", "limit": 50000, "apiKey": api_key}
    for attempt in range(5):
        res = requests.get(url, params=params, timeout=30)
        if res.status_code != 429:
            break
        time.sleep(5 * (attempt + 1))  # rate-limited: wait and retry instead of giving up
    res.raise_for_status()
    rows = res.json().get("results") or []
    if not rows:
        return pd.DataFrame()

    bars = pd.DataFrame(rows)
    bars.index = pd.to_datetime(bars["t"], unit="ms", utc=True).dt.tz_convert(ET)
    return bars.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})


def get_entry_fill(ticker, accepted_et, delay_minutes=1, extended_hours=False, adjusted=False):
    """
    First price we could actually trade at after a filing goes public.

    accepted_et:    the filing's SEC acceptance time (tz-aware, US/Eastern)
    delay_minutes:  time to receive, score and route the order after acceptance
    extended_hours: if False, filings outside 9:30-16:00 ET fill at the next regular open

    Returns (fill_time, fill_price) or (None, None) if no bars are available.
    """
    api_key = os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        raise ValueError("MASSIVE_API_KEY environment variable is missing.")

    tradable_at = accepted_et + timedelta(minutes=delay_minutes)
    # Look 5 days ahead so weekends and holidays roll to the next session automatically
    bars = fetch_minute_bars(ticker, tradable_at, tradable_at + timedelta(days=5), api_key, adjusted)
    if bars.empty:
        return None, None

    # Bars are stamped at their start, so a bar that starts at/after tradable_at is fully after the news
    bars = bars[bars.index >= tradable_at]
    if not extended_hours:
        t = bars.index.time
        bars = bars[(t >= MARKET_OPEN) & (t < MARKET_CLOSE)]
    if bars.empty:
        return None, None

    first = bars.iloc[0]
    return bars.index[0], float(first["open"])


if __name__ == "__main__":
    # A 16:05 filing should fill at the next morning's 9:30 open
    accepted = datetime(2024, 1, 2, 16, 5, 12, tzinfo=ET)
    fill_time, fill_price = get_entry_fill("LLY", accepted)
    print(f"Accepted {accepted} -> fill at {fill_time} @ {fill_price}")
