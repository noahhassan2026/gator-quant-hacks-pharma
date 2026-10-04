import html
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

# Pulls every 8-K straight from SEC EDGAR, with the exact second each one went public.
# EDGAR has every filing back to 1994, so 2016-2024 is complete.

ET = ZoneInfo("America/New_York")
OUTPUT_CSV = "historical_sec_8k_2016_2024.csv"
MAX_TEXT_CHARS = 30000  # 8-K + press-release exhibits; enough for LLM scoring

# Industry filter (SEC SIC codes): drugs, diagnostics, biologics, and biotech R&D firms
PHARMA_SIC = {"2834", "2835", "2836", "8731"}

# 8-K items that tend to move a drug stock. Leaves out routine items such as
# 5.07 (shareholder vote results), 5.03 (bylaw changes) and 9.01-only exhibit filings.
MATERIAL_ITEMS = {
    "1.01",  # material agreement: licensing deals, partnerships, settlements
    "1.02",  # agreement terminated (partner walks away)
    "1.03",  # bankruptcy
    "2.01",  # acquisition or sale of assets
    "2.02",  # earnings
    "2.05",  # restructuring / layoffs
    "2.06",  # impairment (failed program written off)
    "3.01",  # delisting notice
    "4.02",  # past financials can't be relied on
    "7.01",  # Reg FD disclosure: trial data, investor presentations
    "8.01",  # other events: FDA decisions, trial results, court rulings
}
SIC_CACHE_CSV = "sec_company_sic.csv"

# Speed settings. SEC allows 10 requests/second per IP; staying a little under avoids blocks.
WORKERS = 8
REQUESTS_PER_SECOND = 8
MAX_DOWNLOAD_BYTES = 3_000_000
# Exhibit types that never help scoring (images, XBRL data, spreadsheets). The .txt lists these
# after the 8-K and press releases, so the download stops as soon as one shows up.
STOP_AT_TYPES = ("GRAPHIC", "EX-101", "XML", "ZIP", "JSON", "EXCEL", "PDF")

HEADERS = {
    # SEC requires a User-Agent with your name and email
    "User-Agent": os.environ.get("SEC_USER_AGENT", ""),
    "Accept-Encoding": "gzip, deflate",
}


_rate_lock = threading.Lock()
_next_slot = [0.0]


def _wait_for_slot():
    """Shared across threads, so all workers together stay under REQUESTS_PER_SECOND."""
    with _rate_lock:
        now = time.monotonic()
        slot = max(now, _next_slot[0])
        _next_slot[0] = slot + 1.0 / REQUESTS_PER_SECOND
    time.sleep(max(0.0, slot - now))


def sec_get(url, stop_markers=None):
    """GET from sec.gov within the rate limit. With stop_markers, streams the response and
    stops reading once any marker appears (or MAX_DOWNLOAD_BYTES), skipping the rest."""
    for attempt in range(5):
        _wait_for_slot()
        try:
            with requests.get(url, headers=HEADERS, timeout=30, stream=stop_markers is not None) as res:
                if res.status_code in (429, 503):
                    time.sleep(10 * (attempt + 1))
                    continue
                res.raise_for_status()
                if stop_markers is None:
                    return res.text
                chunks, size, tail = [], 0, ""
                for chunk in res.iter_content(65536):
                    piece = chunk.decode("utf-8", errors="ignore")
                    chunks.append(piece)
                    size += len(chunk)
                    window = tail + piece
                    if size >= MAX_DOWNLOAD_BYTES or any(m in window for m in stop_markers):
                        break
                    tail = piece[-200:]
                return "".join(chunks)
        except requests.RequestException as e:
            print(f" [Network error: {e}]", end="")
            time.sleep(5)
    raise RuntimeError(f"SEC kept refusing {url}")


def parallel(fn, items):
    """Runs fn over items with WORKERS threads, yielding (item, result or exception) as each finishes."""
    def safe(item):
        try:
            return item, fn(item)
        except Exception as e:
            return item, e
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        yield from pool.map(safe, items)


def load_ticker_map():
    """CIK -> ticker for companies that are listed today."""
    import json
    data = json.loads(sec_get("https://www.sec.gov/files/company_tickers.json"))
    return {int(row["cik_str"]): row["ticker"] for row in data.values()}


def list_8ks(year, quarter):
    """Every 8-K and 8-K/A filed in one quarter, from EDGAR's master index."""
    text = sec_get(f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.idx")
    rows = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) != 5 or parts[2] not in ("8-K", "8-K/A"):
            continue
        cik, company, form, date, path = parts
        rows.append({
            "CIK": int(cik),
            "Company": company,
            "Form": form,
            "Date": date,
            "Accession": path.rsplit("/", 1)[-1].replace(".txt", ""),
            "URL": "https://www.sec.gov/Archives/" + path,
        })
    return rows


def _strip_html(raw):
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(raw)).strip()


def parse_filing(raw):
    """Pulls the exact acceptance time, item list and readable text from a full .txt submission."""
    accepted = None
    m = re.search(r"<ACCEPTANCE-DATETIME>\s*(\d{14})", raw) or re.search(r"ACCEPTANCE-DATETIME:\s*(\d{14})", raw)
    if m:
        # EDGAR acceptance times are US/Eastern
        accepted = datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=ET)

    items = "; ".join(re.findall(r"ITEM INFORMATION:\s*(.+)", raw))

    # Keep the 8-K itself and press-release exhibits (EX-99.x); skip images, XBRL, etc.
    parts = []
    # (?:...|\Z) also keeps a document cut off by the early stop
    for doc in re.findall(r"(?s)<DOCUMENT>(.*?)(?:</DOCUMENT>|\Z)", raw):
        doc_type = re.search(r"<TYPE>\s*([^\s<]+)", doc)
        doc_type = doc_type.group(1).upper() if doc_type else ""
        if doc_type.startswith("8-K") or doc_type.startswith("EX-99"):
            body = re.search(r"(?s)<TEXT>(.*?)(?:</TEXT>|\Z)", doc)
            if body:
                parts.append(f"[{doc_type}] " + _strip_html(body.group(1)))
    text = "\n\n".join(parts)[:MAX_TEXT_CHARS]
    return accepted, items, text


def get_company_filings(cik, include_history):
    """SIC code plus {accession: item list} for one company, from EDGAR's submissions API."""
    import json
    base = "https://data.sec.gov/submissions/"
    data = json.loads(sec_get(f"{base}CIK{int(cik):010d}.json"))
    blocks = [data["filings"]["recent"]]
    if include_history:
        for extra in data["filings"].get("files", []):
            blocks.append(json.loads(sec_get(base + extra["name"])))
    items = {}
    for b in blocks:
        for acc, it in zip(b["accessionNumber"], b.get("items", [])):
            items[acc] = it
    return str(data.get("sic", "")), items


def load_sic_codes(ciks):
    """SIC code for each company, cached to disk so re-runs skip the lookups."""
    cache = {}
    if os.path.exists(SIC_CACHE_CSV):
        cached = pd.read_csv(SIC_CACHE_CSV, dtype=str)
        cache = dict(zip(cached["CIK"].astype(int), cached["SIC"].fillna("")))
    todo = [c for c in ciks if c not in cache]
    print(f"Looking up industry codes for {len(todo)} companies ({len(cache)} cached)...", flush=True)
    for i, (cik, result) in enumerate(parallel(lambda c: get_company_filings(c, include_history=False)[0], todo), 1):
        if isinstance(result, Exception):
            print(f"  [CIK {cik}: {result}]")
        else:
            cache[cik] = result
        if i % 500 == 0 or i == len(todo):
            pd.DataFrame({"CIK": list(cache), "SIC": list(cache.values())}).to_csv(SIC_CACHE_CSV, index=False)
            print(f"  {i}/{len(todo)}", flush=True)
    return cache


def select_filings(all_filings, tickers, ticker_map):
    """Keeps pharma/biotech companies and filings with market-moving items."""
    if tickers is not None:
        wanted = {cik for cik, t in ticker_map.items() if t in set(tickers)}
        missing = set(tickers) - {ticker_map[c] for c in wanted}
        if missing:
            print(f"Not found on SEC (may file 6-K instead of 8-K): {sorted(missing)}")
        all_filings = [f for f in all_filings if f["CIK"] in wanted]

    # Without a ticker there are no prices to train on, and amendments (8-K/A) rarely move a stock
    before = len(all_filings)
    all_filings = [f for f in all_filings if f["Form"] == "8-K" and f["CIK"] in ticker_map]
    print(f"Listed companies, original 8-Ks only: {len(all_filings)} of {before}")

    sic = load_sic_codes(sorted({f["CIK"] for f in all_filings}))
    pharma_ciks = {c for c, code in sic.items() if code in PHARMA_SIC}
    filings = [f for f in all_filings if f["CIK"] in pharma_ciks]
    print(f"Pharma/biotech: {len(pharma_ciks)} companies, {len(filings)} of {len(all_filings)} 8-Ks")

    print("Reading item codes for those filings...", flush=True)
    items = {}
    ciks = sorted({f["CIK"] for f in filings})
    for cik, result in parallel(lambda c: get_company_filings(c, include_history=True)[1], ciks):
        if isinstance(result, Exception):
            print(f"  [CIK {cik}: {result}]")
        else:
            items.update(result)

    kept = []
    for f in filings:
        codes = {c.strip() for c in items.get(f["Accession"], "").split(",") if c.strip()}
        # Keep filings whose items we couldn't read, rather than silently dropping them
        if not codes or codes & MATERIAL_ITEMS:
            kept.append(f)
    print(f"Market-moving items: {len(kept)} of {len(filings)} pharma 8-Ks\n")
    return kept


def main(tickers, start_year, end_year):
    if not HEADERS["User-Agent"]:
        raise ValueError("Set SEC_USER_AGENT first, e.g. export SEC_USER_AGENT='David Portilla you@email.com'")

    ticker_map = load_ticker_map()

    print("Reading EDGAR's quarterly indexes...", flush=True)
    all_filings = []
    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            try:
                all_filings += list_8ks(year, quarter)
            except requests.HTTPError:
                print(f"  {year} Q{quarter}: no index yet, skipping")  # a quarter that hasn't started
    filings = select_filings(all_filings, tickers, ticker_map)

    # Resume support: skip filings already saved by an earlier run
    done = set()
    if os.path.exists(OUTPUT_CSV):
        done = set(pd.read_csv(OUTPUT_CSV, usecols=["Accession"])["Accession"])
        print(f"Resuming: {len(done)} filings already in {OUTPUT_CSV}")

    by_quarter = {}
    for f in filings:
        if f["Accession"] not in done:
            by_quarter.setdefault(f["Date"][:4] + " Q" + str((int(f["Date"][5:7]) - 1) // 3 + 1), []).append(f)

    for label in sorted(by_quarter):
        filings_q = by_quarter[label]
        print(f"\n{label}: {len(filings_q)} 8-Ks to download", flush=True)
        stop_markers = [f"<TYPE>{t}" for t in STOP_AT_TYPES]
        batch = []
        for i, (f, result) in enumerate(parallel(lambda f: parse_filing(sec_get(f["URL"], stop_markers)), filings_q), 1):
            if isinstance(result, Exception):
                print(f"  [skip {f['Accession']}: {result}]")
            else:
                accepted, items, text = result
                f.update({
                    "Ticker": ticker_map.get(f["CIK"], ""),
                    "AcceptedET": accepted.isoformat() if accepted else "",
                    "Items": items,
                    "Text": text,
                })
                batch.append(f)

            if batch and (len(batch) == 200 or i == len(filings_q)):
                pd.DataFrame(batch).to_csv(OUTPUT_CSV, mode="a", index=False,
                                           header=not os.path.exists(OUTPUT_CSV))
                print(f"  saved {i}/{len(filings_q)}", flush=True)
                batch = []

    if not os.path.exists(OUTPUT_CSV):
        print("\nNo filings matched the filters.")
        return
    df = pd.read_csv(OUTPUT_CSV)
    accepted = pd.to_datetime(df["AcceptedET"], utc=True, errors="coerce").dt.tz_convert(ET)
    print("\n" + "=" * 60)
    print(f"Total 8-Ks saved: {len(df)}  (missing exact time: {accepted.isna().sum()})")
    print("\nFilings per year:")
    print(df["Date"].astype(str).str[:4].value_counts().sort_index().to_string())
    print("\nFilings by hour published (ET):")
    print(accepted.dropna().dt.hour.value_counts().sort_index().to_string())


if __name__ == "__main__":
    # None = every pharma/biotech company on EDGAR (filtered by PHARMA_SIC and MATERIAL_ITEMS above).
    # Safe to stop with Ctrl+C and re-run; it picks up where it left off.
    # To pull only a watchlist, use a list instead, e.g.
    # TICKERS = ["LLY", "BMY", "REGN", "BIIB", "AMRN", "VRTX", "JNJ", "TEVA", "PFE"]
    TICKERS = None
    # Training data:  python sec_edgar_8k.py
    # Test data:      python sec_edgar_8k.py 2025 2026 filings_2025_2026.csv
    import sys
    if len(sys.argv) >= 4:
        OUTPUT_CSV = sys.argv[3]
    start_year, end_year = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) >= 3 else (2016, 2024)
    main(TICKERS, start_year=start_year, end_year=end_year)
