import os
import sys
import requests
from google import genai

# Add src to python path to import your other scripts cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from signals.judge_evaluator import analyze_legal_text
from execution.webull_executor import execute_paper_trade, LegalAlphaSignal

def poll_massive_api():
    """
    Polls Massive's AI-ready SEC Filings API for breaking Hatch-Waxman 8-Ks.
    """
    api_key = os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        raise ValueError("Missing MASSIVE_API_KEY environment variable.")
        
    print("📡 Polling Massive API for material 8-K legal filings...")
    
    # Using Massive's endpoint for pre-parsed 8-K disclosures
    url = "https://api.massive.com/stocks/filings/8-K/vX/disclosures"
    
    # We only care about our pharma watchlist and strictly "Regulatory / Legal" events
    params = {
        "tickers.any_of": "AMRN,BMY,JNJ,LLY,NVS,SNY,VRTX,AGN",
        "primary_category": "Regulatory / Legal", 
        "limit": 1,
        "apikey": api_key
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        print(f"Massive API Error {response.status_code}: {response.text}")
        return []

if __name__ == "__main__":
    print("🚀 Starting Gator Quant Hacks Live Trading Pipeline...")
    
    # In a production bot, this runs in a while True: loop. 
    # For the hackathon demo, we run it once.
    recent_filings = poll_massive_api()
    
    if not recent_filings:
        print("No new legal 8-K filings found today. Portfolio remains flat.")
        sys.exit(0)
        
    gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    for filing in recent_filings:
        ticker = filing.get("ticker", "UNKNOWN")
        # Massive automatically extracts the exact paragraph containing the regulatory event
        text_excerpt = filing.get("supporting_text", "") 
        
        if not text_excerpt:
            continue
            
        print(f"\n🚨 ALERT: Breaking Legal 8-K detected for {ticker}!")
        print("🧠 Routing SEC document text to Gemini for NLP analysis...")
        
        # 1. Score the filing using your existing Gemini script
        gemini_signal = analyze_legal_text(text_excerpt, gemini_client)
        
        print(f"\n📊 Gemini Alpha Score: {gemini_signal.alpha_score} (Confidence: {gemini_signal.confidence})")
        print(f"📝 Legal Finding: {gemini_signal.key_legal_finding}")
        
        # 2. Execute the trade using your existing Webull script
        if abs(gemini_signal.alpha_score) > 0.6:
            print(f"\n⚡ High confidence signal generated. Routing order to Webull...")
            
            # Map the Gemini output to the simplified Webull executor format
            trade_signal = LegalAlphaSignal(
                ticker=gemini_signal.ticker, 
                alpha_score=gemini_signal.alpha_score
            )
            execute_paper_trade(trade_signal)
        else:
            print("\n⚖️ Signal is too neutral. No trade executed.")