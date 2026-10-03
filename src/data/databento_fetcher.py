import os
import databento as db
import pandas as pd
from datetime import datetime, timedelta

def fetch_historical_price_data(ticker: str, event_date_str: str, days_before: int = 5, days_after: int = 5) -> pd.DataFrame:
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
        # Databento requires dataset names. 'XNAS.ITCH' is standard for US Equities (Nasdaq)
        data = client.timeseries.get_range(
            dataset="XNAS.ITCH", 
            schema="ohlcv-1d",       # Daily Open, High, Low, Close, Volume
            symbols=[ticker],
            start=start_date,
            end=end_date,
            stype_in="raw_symbol"
        )
        
        # Convert the raw Databento output into a clean Pandas DataFrame
        df = data.to_df()
        
        # Databento returns prices as integers (multiplied by 1e9). Convert back to standard decimals.
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col] / 1e9
                
        return df

    except Exception as e:
        print(f"Error fetching Databento data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Test case: Let's assume our BMS Markman ruling happened on 2023-08-15
    test_ticker = "BMY"  # Bristol-Myers Squibb ticker
    test_event_date = "2023-08-15"
    
    # You need to export this in your terminal first: export DATABENTO_API_KEY="your_key"
    try:
        price_df = fetch_historical_price_data(test_ticker, test_event_date)
        if not price_df.empty:
            print(f"\n--- Databento Price Data for {test_ticker} ---")
            # Print just the closing prices and volume
            print(price_df[['close', 'volume']])
        else:
            print("\nNo data returned. Check your API key and Databento account permissions.")
    except ValueError as e:
        print(e)