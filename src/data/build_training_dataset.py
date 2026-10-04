import os
import time
import requests
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

def fetch_historical_8ks_paginated(tickers, start_year, end_year):
    """
    Pulls SEC 8-Ks from Massive API using cursor pagination.
    Adapted to Massive's database limits (2020 onwards).
    """
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not massive_key:
        raise ValueError("❌ MASSIVE_API_KEY environment variable is missing.")

    # Reverted to the endpoint we KNOW works and has the AI tags
    base_url = "https://api.massive.com/stocks/filings/8-K/vX/disclosures"
    all_filings = []

    current_date = datetime(start_year, 1, 1)
    end_limit_date = datetime(end_year, 12, 31)

    print(f"📥 Initiating Paginated Sponsor Data Pull: {start_year} to {end_year}")
    print("=" * 60)

    # Loop year-by-year
    while current_date <= end_limit_date:
        next_date = current_date + relativedelta(years=1) - relativedelta(days=1)
        
        start_str = current_date.strftime("%Y-%m-%d")
        end_str = next_date.strftime("%Y-%m-%d")
        
        print(f"📡 Querying SEC 8-Ks for {start_str} to {end_str}...", end="", flush=True)

        params = {
            "tickers.any_of": ",".join(tickers),
            "start_date": start_str, # Using the original date parameters
            "end_date": end_str,
            "limit": 100,  # Safe chunk size to avoid API clipping
            "apikey": massive_key
        }

        year_count = 0
        url = base_url
        
        # --- PAGINATION LOOP ---
        while url:
            try:
                res = requests.get(url, params=params if url == base_url else None, timeout=15)
                
                if res.status_code == 200:
                    data = res.json()
                    results = data.get("results", [])
                    
                    for f in results:
                        ticker = f.get("ticker") or (f.get("tickers")[0] if f.get("tickers") else "UNKNOWN")
                        text = f.get("supporting_text") or f.get("description", "")
                        raw_date = f.get("filing_date") or f.get("acceptance_datetime", "")
                        date_str = raw_date[:10] if raw_date else None
                        
                        if ticker and text and date_str:
                            all_filings.append({
                                "Date": date_str,
                                "Ticker": ticker,
                                "Text": text
                            })
                            year_count += 1

                    # Follow the cursor to the next page
                    next_url = data.get("next_url")
                    if next_url:
                        url = next_url
                        if "apikey=" not in url:
                            url += f"&apikey={massive_key}"
                        time.sleep(0.5) 
                    else:
                        url = None # End of the year
                else:
                    print(f" [Failed: API Status {res.status_code}]", end="")
                    url = None
                    
            except Exception as e:
                print(f" [Error: {e}]", end="")
                url = None

        print(f" [Retrieved {year_count} filings]")
        current_date = next_date + relativedelta(days=1)

    return pd.DataFrame(all_filings)

if __name__ == "__main__":
    pharma_tickers = ["LLY", "BMY", "REGN", "NVS", "BIIB", "AMRN", "VRTX", "JNJ", "TEVA", "PFE"]
    
    # Set to 2020 to align with Massive's actual database availability
    df = fetch_historical_8ks_paginated(
        tickers=pharma_tickers,
        start_year=2020, 
        end_year=2024
    )
    
    if not df.empty:
        df = df.dropna()
        df = df.sort_values(by=["Date", "Ticker"]).reset_index(drop=True)
        
        output_filename = "historical_sec_8k_2020_2024.csv"
        df.to_csv(output_filename, index=False)
        
        print("\n" + "=" * 60)
        print(f"✅ Data Pull Complete!")
        print(f"Total SEC 8-Ks Extracted: {len(df)}")
        print(f"Dataset saved to: {output_filename}")
        print("=" * 60)
    else:
        print("\n❌ Extraction returned zero results. Check your API key.")