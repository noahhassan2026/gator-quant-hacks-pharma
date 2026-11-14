import os
import threading
import time
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ET = ZoneInfo("America/New_York")
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)

# Where prices come from: MARKET_DATA=massive (default, MASSIVE_API_KEY) or MARKET_DATA=alpaca
# (APCA_API_KEY_ID and APCA_API_SECRET_KEY; the free plan gives full SIP history except the last 15 minutes).
DATA_SOURCE = (os.environ.get("MARKET_DATA") or "massive").strip().lower()
ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
ALPACA_FEED = os.environ.get("ALPACA_FEED") or "sip"

# Every request (all threads together) is spaced to stay under the plan's limit. Massive's free plan
# allows 5 a minute (MASSIVE_REQUESTS_PER_MINUTE=5); Alpaca's free plan 200, so it defaults to 190.
REQUESTS_PER_MINUTE = float(os.environ.get("MARKET_DATA_REQUESTS_PER_MINUTE") or os.environ.get("MASSIVE_REQUESTS_PER_MINUTE")
                            or (190 if DATA_SOURCE == "alpaca" else 0))
_rate_lock = threading.Lock()
_next_slot = [0.0]


def missing_data_key():
    """Name of the first market-data key that isn't set, or None."""
    keys = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY") if DATA_SOURCE == "alpaca" else ("MASSIVE_API_KEY",)
    return next((k for k in keys if not os.environ.get(k)), None)


def data_get(url, params, headers=None):
    """GET within the rate limit, waiting out 429s; raises if it never gets through."""
    for attempt in range(8):
        if REQUESTS_PER_MINUTE:
            with _rate_lock:
                now = time.monotonic()
                slot = max(now, _next_slot[0])
                _next_slot[0] = slot + 60.0 / REQUESTS_PER_MINUTE
            time.sleep(max(0.0, slot - now))
        res = requests.get(url, params=params, headers=headers, timeout=30)
        if res.status_code != 429:
            res.raise_for_status()
            return res
        time.sleep(min(60, 10 * (attempt + 1)))  # rate-limited: wait and retry instead of giving up
    raise RuntimeError(f"{DATA_SOURCE} kept rate-limiting {url.split('?')[0]}; set a lower MARKET_DATA_REQUESTS_PER_MINUTE")


def alpaca_bars(ticker, timeframe, start, end, adjusted):
    """Bars from Alpaca as Massive-style rows (t in ms, o/h/l/c/v), following every page."""
    headers = {"APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"]}
    end = min(pd.Timestamp(end), pd.Timestamp.now(tz=ET) - pd.Timedelta(minutes=16))  # free plan: no SIP data from the last 15 min
    if end <= pd.Timestamp(start):
        return []
    params = {"symbols": ticker, "timeframe": timeframe, "start": pd.Timestamp(start).isoformat(), "end": end.isoformat(),
              "limit": 10000, "adjustment": "split" if adjusted else "raw", "feed": ALPACA_FEED, "sort": "asc"}
    rows = []
    while True:
        data = data_get(ALPACA_BARS_URL, params, headers).json()
        bars = data.get("bars") or {}
        rows += bars.get(ticker, []) if isinstance(bars, dict) else bars
        if not data.get("next_page_token"):
            break
        params["page_token"] = data["next_page_token"]
    for r in rows:
        r["t"] = int(pd.Timestamp(r["t"]).timestamp() * 1000)
    return rows


def fetch_daily_bars(ticker, start, end):
    """Split-adjusted daily bars between two YYYY-MM-DD dates, as Massive-style rows."""
    if DATA_SOURCE == "alpaca":
        return alpaca_bars(ticker, "1Day", pd.Timestamp(start, tz=ET), pd.Timestamp(end, tz=ET) + pd.Timedelta(days=1), True)
    url = f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": os.environ["MASSIVE_API_KEY"]}
    return data_get(url, params).json().get("results") or []


def fetch_minute_bars(ticker, start, end, api_key=None, adjusted=False):
    """1-minute bars between two tz-aware datetimes. Index is US/Eastern."""
    if DATA_SOURCE == "alpaca":
        rows = alpaca_bars(ticker, "1Min", start, end, adjusted)
    else:
        url = (
            f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/minute/"
            f"{int(start.timestamp() * 1000)}/{int(end.timestamp() * 1000)}"
        )
        # adjusted=False keeps prices on the same raw basis as the Databento daily closes;
        # pass adjusted=True when comparing against split-adjusted daily bars
        params = {"adjusted": "true" if adjusted else "false", "sort": "asc", "limit": 50000,
                  "apiKey": api_key or os.environ["MASSIVE_API_KEY"]}
        rows = data_get(url, params).json().get("results") or []
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
    if missing_data_key():
        raise ValueError(f"{missing_data_key()} environment variable is missing.")

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
