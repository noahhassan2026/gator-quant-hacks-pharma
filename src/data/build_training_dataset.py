import os
import time
import requests
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# Massive 8-K Text endpoint: full item text for every 8-K, not just the AI-tagged ones.
# (The 8-K Disclosures endpoint only covers January 2022 onward.)
BASE_URL = "https://api.massive.com/stocks/filings/8-K/vX/text"


def massive_get(url, params, api_key):
    """One GET with rate-limit retries. Returns parsed JSON or None."""
    for _ in range(5):
        try:
            res = requests.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            print(f" [Network error: {e}]", end="")
            return None
        if res.status_code == 429:
            print(" [Rate limited, waiting 15s]", end="", flush=True)
            time.sleep(15)
            continue
        if res.status_code != 200:
            print(f" [HTTP {res.status_code}: {res.text[:200]}]", end="")
            return None
        return res.json()
    return None


def fetch_8k_text(filters, api_key):
    """Pulls every 8-K matching the filters, following next_url pages.
    Uses only the exact-match filters Massive documents for this endpoint
    (ticker, cik, form_type, filing_date)."""
    params = {"sort": "filing_date.asc", "limit": 100, "apiKey": api_key, **filters}
    rows = []
    url, first = BASE_URL, True

    while url:
        data = massive_get(url, params if first else None, api_key)
        first = False
        if data is None:
            break

        for f in data.get("results", []):
            text = f.get("items_text") or ""
            date = f.get("filing_date")
            if date:
                rows.append({
                    "Date": date[:10],
                    "Ticker": f.get("ticker") or filters.get("ticker", ""),
                    "Form": f.get("form_type"),
                    "Accession": f.get("accession_number"),
                    "CIK": f.get("cik"),
                    "URL": f.get("filing_url"),
                    "Text": text,
                })

        url = data.get("next_url")
        if url and "apiKey=" not in url:
            url += ("&" if "?" in url else "?") + f"apiKey={api_key}"
        time.sleep(0.25)

    return rows


def probe_coverage(api_key):
    """Prints the oldest and newest filing Massive has on this endpoint for your key."""
    print("Checking what dates Massive has for 8-K text...")
    for order in ("asc", "desc"):
        data = massive_get(BASE_URL, {"sort": f"filing_date.{order}", "limit": 1, "apiKey": api_key}, api_key)
        results = (data or {}).get("results") or []
        label = "Oldest" if order == "asc" else "Newest"
        if results:
            r = results[0]
            print(f"  {label} filing: {r.get('filing_date')} ({r.get('ticker')})")
        else:
            print(f"  {label} filing: none returned. Raw response: {str(data)[:300]}")
    print()


# ---------------------------------------------------------
# Exact publish time from SEC EDGAR
# ---------------------------------------------------------
# Massive gives the filing date only. EDGAR's submissions API gives the exact
# acceptanceDateTime for every filing, one request per company.
SEC_HEADERS = {
    # SEC requires a User-Agent with your name and email, e.g. "David Portilla you@ufl.edu"
    "User-Agent": os.environ.get("SEC_USER_AGENT", ""),
    "Accept-Encoding": "gzip, deflate",
}


def _norm_accession(acc):
    return str(acc).replace("-", "").strip() if acc else None


def fetch_sec_acceptance_times(cik):
    """Returns {accession_without_dashes: acceptance datetime (US/Eastern)} for one company."""
    if not SEC_HEADERS["User-Agent"]:
        raise ValueError("Set SEC_USER_AGENT, e.g. export SEC_USER_AGENT='Your Name you@email.com'")

    cik10 = str(cik).zfill(10)
    base = "https://data.sec.gov/submissions/"
    res = requests.get(f"{base}CIK{cik10}.json", headers=SEC_HEADERS, timeout=30)
    res.raise_for_status()
    data = res.json()

    # "recent" holds the latest ~1,000 filings; older ones are split into extra files.
    blocks = [data["filings"]["recent"]]
    for extra in data["filings"].get("files", []):
        time.sleep(0.15)  # SEC allows 10 requests/second
        r = requests.get(base + extra["name"], headers=SEC_HEADERS, timeout=30)
        r.raise_for_status()
        blocks.append(r.json())

    times = {}
    for b in blocks:
        for acc, accepted in zip(b["accessionNumber"], b["acceptanceDateTime"]):
            # EDGAR writes this with a trailing "Z", but the clock time is US/Eastern.
            # Check: earnings 8-Ks should cluster around 06:00-08:00 and 16:00-16:30.
            naive = datetime.strptime(accepted[:19], "%Y-%m-%dT%H:%M:%S")
            times[_norm_accession(acc)] = naive.replace(tzinfo=ET)
    return times


def add_acceptance_times(df):
    times_by_cik = {}
    for cik in df["CIK"].dropna().unique():
        print(f"SEC EDGAR acceptance times for CIK {cik}...", end="", flush=True)
        try:
            times_by_cik[cik] = fetch_sec_acceptance_times(cik)
            print(" ok")
        except Exception as e:
            print(f" [Error: {e}]")
        time.sleep(0.15)

    accepted = [
        times_by_cik.get(cik, {}).get(_norm_accession(acc))
        for cik, acc in zip(df["CIK"], df["Accession"])
    ]
    df["AcceptedET"] = pd.to_datetime(pd.Series(accepted, dtype="object"), utc=True).dt.tz_convert(ET)
    return df


def build_dataset(tickers, start_year, end_year):
    """tickers=None pulls every company's 8-Ks, one trading day at a time."""
    api_key = os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        raise ValueError("MASSIVE_API_KEY environment variable is missing.")

    who = "ALL companies" if tickers is None else f"{len(tickers)} tickers"
    print(f"Pulling 8-K text from Massive for {who}, {start_year}-{end_year}")
    print("=" * 60)

    probe_coverage(api_key)

    all_rows = []
    if tickers is None:
        # One request per trading day using the exact filing_date filter
        days = pd.bdate_range(f"{start_year}-01-01", f"{end_year}-12-31")
        current_month = None
        month_count = 0
        for day in days:
            if day.strftime("%Y-%m") != current_month:
                if current_month:
                    print(f" {month_count} filings", flush=True)
                current_month, month_count = day.strftime("%Y-%m"), 0
                print(f"{current_month}...", end="", flush=True)
            rows = fetch_8k_text({"filing_date": day.strftime("%Y-%m-%d")}, api_key)
            month_count += len(rows)
            all_rows.extend(rows)
        print(f" {month_count} filings", flush=True)
    else:
        # A single company has a few hundred 8-Ks at most: pull them all, keep the years we want
        for ticker in tickers:
            print(f"{ticker}...", end="", flush=True)
            rows = [r for r in fetch_8k_text({"ticker": ticker}, api_key)
                    if str(start_year) <= r["Date"][:4] <= str(end_year)]
            print(f" {len(rows)} filings in {start_year}-{end_year}", flush=True)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    # Same filing can show up twice across page boundaries; keep one per accession number.
    df = df.drop_duplicates(subset=["Accession"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    print("\nLooking up exact publish times on SEC EDGAR...")
    df = add_acceptance_times(df)
    df = df.sort_values(["AcceptedET", "Ticker"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    # None = every company's 8-Ks. To pull only a watchlist, use a list instead, e.g.
    # TICKERS = ["LLY", "BMY", "REGN", "BIIB", "AMRN", "VRTX", "JNJ", "TEVA", "PFE"]
    TICKERS = None

    df = build_dataset(TICKERS, start_year=2016, end_year=2024)

    if df.empty:
        print("\nNo filings returned. Check your API key and plan.")
    else:
        output_filename = "historical_sec_8k_2016_2024.csv"
        df.to_csv(output_filename, index=False)

        print("\n" + "=" * 60)
        print(f"Saved {len(df)} filings to {output_filename}")
        print("\nFilings per year (a missing year means Massive has no data for it):")
        print(df["Date"].str[:4].value_counts().sort_index().to_string())

        missing = df["AcceptedET"].isna().sum()
        print(f"\nFilings without an exact SEC time: {missing} (drop these from the backtest)")
        print("\nFilings by hour published (ET) - expect peaks near 7-8 and 16:")
        print(df["AcceptedET"].dropna().dt.hour.value_counts().sort_index().to_string())
        print("=" * 60)
