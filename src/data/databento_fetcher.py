import os
import databento as db
import pandas as pd
from datetime import datetime, timedelta

def fetch_historical_price_data(ticker: str, event_date_str: str, days_before: int = 5, days_after: int = 10) -> pd.DataFrame:
    """
    Fetches daily OHLCV data from Databento around a specific legal event date.
    """
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        raise ValueError("Missing DATABENTO_API_KEY environment variable.")

    # Initialize the Databento Historical client
    client = db.Historical(api_key)

    # Calculate date range
    event_date = datetime.strptime(event_date_str, "%Y-%m-%d")
    start_date = (event_date - timedelta(days=days_before)).strftime("%Y-%m-%d")
    end_date = (event_date + timedelta(days=days_after)).strftime("%Y-%m-%d")

    print(f"Fetching Databento data for {ticker} from {start_date} to {end_date}...")

    try:
        # Request the historical price dataset
        data = client.timeseries.get_range(
            dataset="XNAS.ITCH", 
            schema="ohlcv-1d",
            symbols=[ticker],
            start=start_date,
            end=end_date,
            stype_in="raw_symbol"
        )
        
        # Convert to Pandas DataFrame (Databento now handles price conversion automatically)
        df = data.to_df()
        return df

    except Exception as e:
        print(f"Error fetching Databento data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Target: The actual Amarin (AMRN) Patent Invalidation by Judge Miranda Du
    test_ticker = "AMRN"
    test_event_date = "2020-03-30"  # The correct date of the crash
    
    try:
        # Execute the fetcher
        price_df = fetch_historical_price_data(test_ticker, test_event_date, days_before=5, days_after=10)
        
        if not price_df.empty:
            print(f"\n--- Databento Price Data for {test_ticker} ---")
            print(price_df[['close', 'volume']])
            
            # Calculate the impact
            max_price = price_df['close'].max()
            min_price = price_df['close'].min()
            crash_pct = ((min_price - max_price) / max_price) * 100
            
            print(f"\n--- Catalyst Impact ---")
            print(f"Pre-Ruling High:  ${max_price:.2f}")
            print(f"Post-Ruling Low:  ${min_price:.2f}")
            print(f"Total Volatility: {crash_pct:.2f}%")
            print("\nHackathon Note: This perfectly illustrates the directional trade opportunity.")
        else:
            print("\nNo data returned. Check your Databento API key and permissions.")
            
    # Here is the missing exception block!
    except Exception as e:
        print(f"Execution Error: {e}")