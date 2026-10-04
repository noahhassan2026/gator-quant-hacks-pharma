import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

# Step 2 of training: for every filing, what did the stock actually do next?
# Input:  historical_sec_8k_2016_2024.csv (from sec_edgar_8k.py)
# Output: labels.csv, one row per tradable filing with market features + outcome labels.

FILINGS_CSV = "historical_sec_8k_2016_2024.csv"
OUTPUT_CSV = "labels.csv"
PRICE_DIR = "prices"            # daily bars are cached here, one CSV per ticker
HEDGE = "XPH"                   # returns are measured against the pharma ETF
START, END = "2015-10-01", "2025-03-31"  # room for 20-day lookback and 30-day follow-through
COST = 0.0010                   # 10 bps per side
MIN_PRICE, MIN_DOLLAR_VOLUME = 5.0, 1_000_000  # skip stocks you couldn't really trade
USE_MINUTE_ENTRY = True         # fill at the first minute bar after the filing (needs intraday_entry.py)
WORKERS = 16                    # parallel Massive requests; lower this if you see rate-limit pauses

minute_errors = []

# Exit menu the exit model chooses from: (take-profit, stop, max days). None = no target.
EXIT_RULES = {
    "quick":      (0.05, 0.03, 3),
    "short_drift": (0.08, 0.05, 10),
    "default":    (0.15, 0.08, 30),
    "wide":       (0.25, 0.12, 30),
    "time_only":  (None, 0.15, 20),
    "vol_scaled": ("vol", "vol", 15),
}


def daily_bars(ticker):
    """Split-adjusted daily bars from Massive, cached to disk."""
    os.makedirs(PRICE_DIR, exist_ok=True)
    path = os.path.join(PRICE_DIR, f"{ticker}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)

    url = f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/day/{START}/{END}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": os.environ["MASSIVE_API_KEY"]}
    for _ in range(5):
        res = requests.get(url, params=params, timeout=30)
        if res.status_code == 429:
            time.sleep(15)
            continue
        res.raise_for_status()
        break
    rows = res.json().get("results") or []
    df = pd.DataFrame(rows)
    if not df.empty:
        df.index = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})[["open", "high", "low", "close", "volume"]]
    df.to_csv(path)
    time.sleep(0.25)
    return df


def simulate_exit(path, rule, vol):
    """Net return of one exit rule on a daily path of cumulative hedged returns (index 0 = entry day)."""
    tp, sl, max_days = rule
    if tp == "vol":
        tp, sl = 2 * vol * np.sqrt(max_days), vol * np.sqrt(max_days)
    for day in range(1, min(max_days, len(path) - 1) + 1):
        r = path[day]
        if sl is not None and r <= -sl:
            return r - 2 * COST
        if tp is not None and r >= tp:
            return r - 2 * COST
    return path[min(max_days, len(path) - 1)] - 2 * COST


def label_filing(f, px, hedge):
    accepted = pd.Timestamp(f["AcceptedET"])
    if accepted.tzinfo is not None:
        accepted = accepted.tz_convert("America/New_York").tz_localize(None)

    # Entry day: the filing's own day if it came out before the 4pm close, else the next trading day
    after_close = accepted.time() >= pd.Timestamp("16:00").time()
    days = px.index[px.index > accepted.normalize()] if after_close else px.index[px.index >= accepted.normalize()]
    if len(days) < 21:
        return None
    entry_day = days[0]
    i0 = px.index.get_loc(entry_day)
    if i0 < 20 or entry_day not in hedge.index:
        return None

    # Features: only data from before the filing
    hist = px.iloc[i0 - 20:i0]
    h_hist = hedge.loc[hist.index.intersection(hedge.index)]
    price = hist["close"].iloc[-1]
    dollar_vol = (hist["close"] * hist["volume"]).mean()
    if price < MIN_PRICE or dollar_vol < MIN_DOLLAR_VOLUME:
        return None
    rets = hist["close"].pct_change().dropna()
    vol = rets.std()

    # Entry price: first minute bar after the filing if available, else the entry day's close.
    # The XPH hedge leg uses the open for next-morning fills, and the day's close as an
    # approximation for fills during market hours.
    entry_price, hedge_entry = None, None
    if USE_MINUTE_ENTRY:
        try:
            from intraday_entry import get_entry_fill
            _, entry_price = get_entry_fill(f["Ticker"], pd.Timestamp(f["AcceptedET"]).to_pydatetime(), adjusted=True)
        except Exception as e:
            entry_price = None
            if not minute_errors:  # show the reason once instead of failing silently
                minute_errors.append(e)
                print(f"  [minute entry failed, using daily close instead: {e!r}]", flush=True)
        if entry_price is not None:
            outside_hours = entry_day.normalize() > accepted.normalize() or accepted.time() < pd.Timestamp("09:30").time()
            hedge_entry = hedge["open"].loc[entry_day] if outside_hours else hedge["close"].loc[entry_day]
    entry_source = "minute"
    if entry_price is None:
        entry_price = px["close"].iloc[i0]
        hedge_entry = hedge["close"].loc[entry_day]
        entry_source = "close"

    # Forward path of the hedged trade: stock return minus XPH return, day 0..30
    fwd = px["close"].iloc[i0:i0 + 31]
    h_fwd = hedge["close"].reindex(fwd.index).ffill()
    path = (fwd / entry_price - 1) - (h_fwd / hedge_entry - 1)
    path = path.to_numpy(copy=True)
    path[0] = 0.0 if np.isnan(path[0]) else path[0]

    def ex(n):
        return path[min(n, len(path) - 1)]

    row = {
        "Accession": f["Accession"],
        "Ticker": f["Ticker"],
        "AcceptedET": f["AcceptedET"],
        "EntryDay": entry_day.date().isoformat(),
        "EntrySource": entry_source,
        # market features
        "pre_ret_5": hist["close"].iloc[-1] / hist["close"].iloc[-6] - 1 - (h_hist["close"].iloc[-1] / h_hist["close"].iloc[-6] - 1),
        "pre_ret_20": hist["close"].iloc[-1] / hist["close"].iloc[0] - 1 - (h_hist["close"].iloc[-1] / h_hist["close"].iloc[0] - 1),
        "vol_20": vol,
        "log_dollar_volume": np.log10(dollar_vol),
        "log_price": np.log10(price),
        "after_hours": int(after_close or accepted.time() < pd.Timestamp("09:30").time()),
        # outcome labels (future data, used only as training targets)
        "excess_1d": ex(1), "excess_5d": ex(5), "excess_10d": ex(10), "excess_20d": ex(20),
        "mfe_30d": np.nanmax(path[1:]), "mae_30d": np.nanmin(path[1:]),
    }
    for name, rule in EXIT_RULES.items():
        row[f"long_{name}"] = simulate_exit(path, rule, vol)
        row[f"short_{name}"] = simulate_exit(-path, rule, vol)
    return row


def main():
    filings = pd.read_csv(FILINGS_CSV, usecols=["Accession", "Ticker", "AcceptedET", "Date", "CIK"])
    filings = filings.dropna(subset=["Ticker", "AcceptedET"])
    print(f"{len(filings)} filings with a ticker and exact time")

    # Days since the same company's previous filing
    filings["AcceptedTS"] = pd.to_datetime(filings["AcceptedET"], utc=True)
    filings = filings.sort_values("AcceptedTS")
    filings["days_since_last_filing"] = filings.groupby("CIK")["AcceptedTS"].diff().dt.days.fillna(365).clip(upper=365)

    hedge = daily_bars(HEDGE)

    # 1. Daily bars for every ticker, several at a time (cached, so re-runs skip this)
    tickers = sorted(filings["Ticker"].unique())
    print(f"Fetching daily prices for {len(tickers)} tickers...", flush=True)

    def safe_bars(t):
        try:
            return t, daily_bars(t)
        except Exception as e:
            print(f"  [{t}: {e}]")
            return t, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        prices = dict(pool.map(safe_bars, tickers))

    # 2. Label every filing, several at a time (each may make one minute-bar request)
    todo = [f for _, f in filings.iterrows() if not prices.get(f["Ticker"], pd.DataFrame()).empty]
    print(f"Labeling {len(todo)} filings...", flush=True)

    def safe_label(f):
        try:
            row = label_filing(f, prices[f["Ticker"]], hedge)
        except Exception as e:
            print(f"  [{f['Accession']}: {e}]")
            return None
        if row:
            row["days_since_last_filing"] = f["days_since_last_filing"]
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for n, row in enumerate(pool.map(safe_label, todo), 1):
            if row:
                rows.append(row)
            if n % 1000 == 0:
                print(f"  {n}/{len(todo)} done, {len(rows)} tradable", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} labeled filings to {OUTPUT_CSV}")
    if not df.empty:
        print(f"Entry price source: {df['EntrySource'].value_counts().to_dict()}")
        print(df[["excess_1d", "excess_10d", "mfe_30d", "mae_30d"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
