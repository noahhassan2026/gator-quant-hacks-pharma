import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pydantic import BaseModel
from google import genai

# Add project root to sys.path for clean imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.databento_fetcher import fetch_historical_price_data

# ---------------------------------------------------------
# 1. AGGRESSIVE CATALYST NLP SCHEMA
# ---------------------------------------------------------
class CatalystSignal(BaseModel):
    trade_type: str       # "LONG", "SHORT", "STRADDLE", "NEUTRAL"
    confidence: float     # 0.0 to 1.0
    reasoning: str        # Concise legal/business rationale

def analyze_generalized_event(text: str, gemini_client: genai.Client) -> CatalystSignal:
    """
    Aggressive classification of SEC 8-Ks and federal dockets. 
    Hunts for smaller edges (partnerships, pipeline updates) in addition to major legal rulings.
    """
    sys_prompt = """
    You are an aggressive quantitative analyst for a multi-strategy biopharma hedge fund.
    Analyze the provided SEC filing or judicial docket and classify the actionable trade strategy:
    
    1. "LONG": Any bullish news (won litigation, patent upheld, FDA drug approval, clinical trial beat, positive settlement, new strategic partnerships, cost-cutting measures, positive pipeline updates, or revenue growth).
    2. "SHORT": Any bearish news (lost litigation, generic entry permitted, FDA rejection, clinical trial failure, major government lawsuit, lost partnerships, or negative financial updates).
    3. "STRADDLE": Imminent binary catalyst with unknown direction (trial start date, FDA PDUFA date, scheduled advisory committee vote, or upcoming earnings date announced).
    4. "NEUTRAL": STRICTLY routine administrative noise with absolutely zero market impact (e.g., standard annual meetings, standard equity grants to executives, routine debt refinancing).
    
    BE AGGRESSIVE. If the filing leans even slightly positive or negative, classify it as LONG or SHORT with a lower confidence (e.g., 0.30 - 0.50). Only use NEUTRAL for pure administrative paperwork.
    """
    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=text,
        config={
            "system_instruction": sys_prompt,
            "response_mime_type": "application/json",
            "response_schema": CatalystSignal,
            "temperature": 0.2, # Slightly higher temp for more aggressive interpretation
        }
    )
    return CatalystSignal.model_validate_json(response.text)

# ---------------------------------------------------------
# 2. UNLIMITED EVENT INGESTION PIPELINE
# ---------------------------------------------------------
def fetch_all_historical_events(start_date: datetime, end_date: datetime, max_events: int = 1500):
    """
    Pulls ALL historical data from Massive and CourtListener since the start date.
    """
    events = []
    
    # 1. Fetch SEC 8-Ks via Massive API (No low limit)
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if massive_key:
        print(f"📡 Querying Massive API for ALL SEC 8-Ks ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})...", flush=True)
        url = "https://api.massive.com/stocks/filings/8-K/vX/disclosures"
        params = {
            "tickers.any_of": "LLY,BMY,REGN,NVS,BIIB,AMRN,VRTX,JNJ,TEVA,PFE",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "limit": max_events,
            "apikey": massive_key
        }
        try:
            res = requests.get(url, params=params, timeout=15)
            if res.status_code == 200:
                results = res.json().get("results")
                if isinstance(results, list):
                    for f in results:
                        ticker = f.get("ticker") or (f.get("tickers")[0] if f.get("tickers") else None)
                        text = f.get("supporting_text") or f.get("description", "")
                        raw_date = f.get("filing_date") or f.get("acceptance_datetime", "")
                        date_str = raw_date[:10] if raw_date else None
                        
                        if ticker and text and date_str:
                            events.append({
                                "ticker": ticker,
                                "date": date_str,
                                "text": text,
                                "source": "SEC 8-K"
                            })
                    print(f"   -> Successfully ingested {len(events)} filings from Massive.", flush=True)
        except Exception as e:
            print(f"   [Warning] Massive query encountered error: {e}", flush=True)

    # 2. Fetch Federal Opinions via CourtListener API
    cl_token = os.environ.get("COURTLISTENER_API_KEY")
    if cl_token:
        print("⚖️ Querying CourtListener API for Federal Opinions...", flush=True)
        pharma_mapping = {
            "LLY": "Eli Lilly", "BMY": "Bristol-Myers Squibb", "REGN": "Regeneron",
            "NVS": "Novartis", "AMRN": "Amarin", "VRTX": "Vertex Pharmaceuticals",
            "JNJ": "Johnson & Johnson", "TEVA": "Teva Pharmaceuticals"
        }
        headers = {
            "Accept": "application/json",
            "Authorization": f"Token {cl_token}",
            "User-Agent": "GatorQuantHacksBot/4.0 (quant-research@ufl.edu)"
        }
        url = "https://www.courtlistener.com/api/rest/v4/search/"
        cl_count = 0
        for ticker, name in pharma_mapping.items():
            params = {
                "q": f'"{name}"',
                "type": "o",
                "date_filed_min": start_date.strftime("%Y-%m-%d"),
                "ordering": "-date_filed"
            }
            try:
                res = requests.get(url, params=params, headers=headers, timeout=(3, 5)) # Fast timeout
                if res.status_code == 200:
                    results = res.json().get("results")
                    if isinstance(results, list):
                        for case in results[:15]:  # Pulled up to 15 cases per ticker
                            snippet = case.get("snippet") or case.get("case_name") or ""
                            raw_date = case.get("date_filed")
                            if raw_date and isinstance(raw_date, str):
                                date_str = raw_date[:10]
                                events.append({
                                    "ticker": ticker,
                                    "date": date_str,
                                    "text": f"JUDICIAL RULING in {case.get('case_name', 'Matter')}: {snippet}",
                                    "source": "CourtListener"
                                })
                                cl_count += 1
                time.sleep(1.0)
            except Exception:
                # Silently pass timeouts to keep the engine moving quickly
                pass
        print(f"   -> Successfully ingested {cl_count} judicial dockets from CourtListener.", flush=True)
    else:
        print("[Notice] COURTLISTENER_API_KEY not set. Continuing with Massive filings.", flush=True)

    # Sort chronological
    sorted_events = sorted(events, key=lambda x: x["date"])
    return sorted_events

# ---------------------------------------------------------
# 3. DAY-BY-DAY TIME-SERIES SIMULATION ENGINE
# ---------------------------------------------------------
def run_dynamic_simulation(events, start_date, end_date, gemini_client, max_api_calls=1500):
    print("\n" + "=" * 70, flush=True)
    print(f"🚀 INITIATING DAY-BY-DAY SIMULATION: {start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}", flush=True)
    print(f"Config: Aggressive NLP Filters | Low Execution Threshold", flush=True)
    print("=" * 70, flush=True)

    events_by_date = {}
    for ev in events:
        events_by_date.setdefault(ev["date"], []).append(ev)

    portfolio_value = 100000.00
    daily_equity_curve = []
    trade_history = []
    active_positions = {}
    
    total_traded_volume = 0.0
    tx_fee_rate = 0.0010  # 10 bps
    gemini_calls_made = 0

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")

        if current_date.day == 1:
            print(f"\n📅 [CALENDAR] {current_date.strftime('%B %Y')} | Portfolio Value: ${portfolio_value:,.2f} | Open Positions: {len(active_positions)} | API Quota Used: {gemini_calls_made}", flush=True)

        todays_events = events_by_date.get(date_str, [])

        for event in todays_events:
            ticker = event["ticker"]
            source = event["source"]

            if gemini_calls_made >= max_api_calls:
                print(f"[{date_str}] ⚠️ Hard API quota limit reached. Skipping further analysis.", flush=True)
                break

            print(f"[{date_str}] 🔍 Evaluating {source} for {ticker}...", end="", flush=True)
            
            try:
                signal = analyze_generalized_event(event["text"], gemini_client)
                gemini_calls_made += 1
                time.sleep(4.2)  # Strict 15 RPM compliance
            except Exception as e:
                if "429" in str(e):
                    print(f" [API Quota Exhausted. Sleeping 60s...]", flush=True)
                    time.sleep(60)
                else:
                    print(f" [API Error: {e}]", flush=True)
                continue

            # NEW LOWER THRESHOLD: 0.25 (Catches more speculative trades)
            if signal.trade_type == "NEUTRAL" or signal.confidence < 0.25:
                print(f" -> Neutral/Filtered (Conf: {signal.confidence:.2f})", flush=True)
                continue

            print(f" -> ⚡ SIGNAL: {signal.trade_type} (Conf: {signal.confidence:.2f})", flush=True)
            print(f"   Reasoning: {signal.reasoning[:95]}...", flush=True)

            # Check for existing position -> Counter-Catalyst exit
            if ticker in active_positions:
                pos = active_positions[ticker]
                if pos["type"] != signal.trade_type:
                    print(f"   🔄 COUNTER-CATALYST detected on {ticker}! Triggering immediate position closure.", flush=True)
                    pos["force_close"] = True
                continue

            # Compute Dynamic Allocation: Risk 5% to 25% based on confidence
            alloc_pct = min(0.25, max(0.05, 0.25 * signal.confidence))
            capital = portfolio_value * alloc_pct

            # Request post-event pricing window from Databento
            max_allowed = datetime.now() - timedelta(days=1)
            days_to_fetch = min(40, (max_allowed - current_date).days)
            if days_to_fetch < 3:
                print(f"   [Skipped] Insufficient future trading days available to track position.", flush=True)
                continue

            print(f"   📊 Querying Databento execution data for {ticker}...", end="", flush=True)
            price_df = fetch_historical_price_data(ticker, date_str, days_before=1, days_after=days_to_fetch)
            
            if price_df.empty or len(price_df) < 2:
                print(" [No Databento pricing available]", flush=True)
                continue
            print(" [Success]", flush=True)

            price_df['clean_date'] = price_df.index.tz_localize(None).floor('D')
            price_curve = dict(zip(price_df['clean_date'].dt.strftime('%Y-%m-%d'), price_df['close']))
            
            entry_price = price_df.iloc[0]['close']
            shares = int(capital / entry_price)
            if shares <= 0:
                continue

            # Entry fee & tracking
            portfolio_value -= (shares * entry_price * tx_fee_rate)
            total_traded_volume += (shares * entry_price)

            active_positions[ticker] = {
                "type": signal.trade_type,
                "shares": shares,
                "entry_price": entry_price,
                "entry_date": date_str,
                "days_held": 0,
                "price_curve": price_curve,
                "force_close": False,
                "confidence": signal.confidence
            }
            print(f"   ⚡ EXECUTED: {signal.trade_type} {shares:,} shares of {ticker} @ ${entry_price:.2f} (${capital:,.2f} deployed)", flush=True)

        # 2. DAILY VALUATION & POSITION LIFECYCLE
        closed_tickers = []
        for ticker, pos in active_positions.items():
            pos["days_held"] += 1
            current_price = pos["price_curve"].get(date_str)
            
            if current_price is None:
                continue  # Weekend or market holiday

            # Evaluate strategy returns
            pct_move = (current_price / pos["entry_price"]) - 1.0
            
            if pos["type"] == "LONG":
                current_pnl_pct = pct_move
            elif pos["type"] == "SHORT":
                current_pnl_pct = -pct_move
            elif pos["type"] == "STRADDLE":
                # Options proxy: 6% premium paid upfront, decaying linearly over 30 days
                premium = 0.06
                theta_decay = (pos["days_held"] / 30.0) * premium
                current_pnl_pct = abs(pct_move) - premium - theta_decay

            # Value-Driven Exit Logic
            exit_reason = None
            if pos["force_close"]:
                exit_reason = "Counter-Catalyst"
            elif current_pnl_pct >= 0.15:
                exit_reason = "Take Profit (+15%)"
            elif current_pnl_pct <= -0.08:
                exit_reason = "Stop Loss (-8%)"
            elif pos["days_held"] >= 30:
                exit_reason = "Time Decay (30d Timeout)"

            if exit_reason:
                gross_pnl = current_pnl_pct * (pos["entry_price"] * pos["shares"])
                net_pnl = gross_pnl - (pos["shares"] * current_price * tx_fee_rate)
                portfolio_value += net_pnl
                total_traded_volume += (pos["shares"] * current_price)

                print(f"[{date_str}] 🛑 CLOSED {pos['type']} {ticker} | Exit: ${current_price:.2f} | Reason: {exit_reason} | Return: {current_pnl_pct:+.2%} (Net PnL: ${net_pnl:+,.2f})", flush=True)
                trade_history.append({
                    "ticker": ticker,
                    "type": pos["type"],
                    "entry_date": pos["entry_date"],
                    "exit_date": date_str,
                    "days": pos["days_held"],
                    "return": current_pnl_pct,
                    "pnl": net_pnl,
                    "reason": exit_reason
                })
                closed_tickers.append(ticker)

        for t in closed_tickers:
            del active_positions[t]

        daily_equity_curve.append(portfolio_value)
        current_date += timedelta(days=1)

    return daily_equity_curve, trade_history, total_traded_volume, gemini_calls_made

# ---------------------------------------------------------
# 4. MAIN RUNNER & PERFORMANCE TEAR SHEET
# ---------------------------------------------------------
def main():
    print("=" * 70, flush=True)
    print("AGGRESSIVE MULTI-STRATEGY QUANT SIMULATION (2026 YTD)", flush=True)
    print("Architecture: SEC 8-Ks | Court Dockets | Gemini Flash Lite | Databento", flush=True)
    print("=" * 70, flush=True)

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("Missing GEMINI_API_KEY environment variable.")
    gemini_client = genai.Client(api_key=gemini_key)

    # 1. Start from January 1, 2026
    start_date = datetime(2026, 1, 1)
    end_date = datetime.now() - timedelta(days=2)

    # 2. Pull ALL historical events
    events = fetch_all_historical_events(start_date, end_date, max_events=2000)
    print(f"\nTimeline Assembled: {len(events)} candidate catalyst events.", flush=True)

    equity_curve, trades, volume_traded, total_calls = run_dynamic_simulation(
        events=events,
        start_date=start_date,
        end_date=end_date,
        gemini_client=gemini_client,
        max_api_calls=2000 # Removed the hard cap
    )

    # ---------------------------------------------------------
    # PERFORMANCE METRICS CALCULATION
    # ---------------------------------------------------------
    start_val = 100000.00
    end_val = equity_curve[-1] if equity_curve else start_val
    net_pnl = end_val - start_val
    total_return = net_pnl / start_val

    equity_series = pd.Series(equity_curve)
    daily_returns = equity_series.pct_change().dropna()
    
    # Annualized Sharpe Ratio (Rf = 4.0%)
    rf_daily = 0.04 / 252
    excess = daily_returns - rf_daily
    sharpe_ratio = np.sqrt(252) * (excess.mean() / excess.std()) if excess.std() > 0 else 0.0

    # Max Drawdown
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Portfolio Turnover
    avg_portfolio = equity_series.mean() if equity_series.mean() > 0 else start_val
    turnover = (volume_traded / 2.0) / avg_portfolio

    print("\n" + "=" * 70, flush=True)
    print("🏆 SYSTEMATIC PERFORMANCE TEAR SHEET (GATOR QUANT HACKS)", flush=True)
    print("=" * 70, flush=True)
    print(f"Starting Capital      : ${start_val:,.2f}", flush=True)
    print(f"Ending Capital        : ${end_val:,.2f}", flush=True)
    print(f"Net Cumulative Return : {total_return:+.2%} (${net_pnl:+,.2f})", flush=True)
    print(f"Sharpe Ratio          : {sharpe_ratio:.2f}", flush=True)
    print(f"Maximum Drawdown      : {max_drawdown:+.2%}", flush=True)
    print(f"Portfolio Turnover    : {turnover:.2f}x", flush=True)
    print(f"Gemini Calls Consumed : {total_calls} calls", flush=True)
    print(f"Total Trades Closed   : {len(trades)}", flush=True)

    if trades:
        win_count = sum(1 for t in trades if t["pnl"] > 0)
        print(f"Win Rate              : {win_count / len(trades):.1%}", flush=True)
        print("\n📝 DETAILED TRADE AUDIT LEDGER:", flush=True)
        print(f"{'TICKER':<6} {'TYPE':<9} {'ENTRY':<11} {'EXIT':<11} {'DAYS':<5} {'RETURN':<8} {'PNL':<11} {'EXIT REASON'}", flush=True)
        print("-" * 75, flush=True)
        for t in trades:
            print(f"{t['ticker']:<6} {t['type']:<9} {t['entry_date']:<11} {t['exit_date']:<11} {t['days']:<5} {t['return']:<+8.2%} ${t['pnl']:<+10,.2f} {t['reason']}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()