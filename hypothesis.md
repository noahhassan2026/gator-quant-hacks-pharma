# Hypothesis: Hatch-Waxman Legal Alpha Pipeline

## 1. Core Hypothesis Statement
Market makers efficiently price the initiation of pharmaceutical patent litigation, but inefficiently price the specific judicial assignments and interim milestone filings in Hatch-Waxman ANDA (Abbreviated New Drug Application) lawsuits. By pairing historical judicial propensity metrics with Gemini-driven Natural Language Processing (NLP) on milestone court filings, we can predict patent validity outcomes and generate statistically significant directional alpha.

## 2. Asset Universe & Risk Neutralization
*   **Target Universe:** U.S. publicly traded branded pharmaceutical innovators facing generic patent challenges.
*   **Beta Neutrality:** Pair long/short equity allocations with offsetting positions in the SPDR S&P Pharmaceuticals ETF (XPH) to isolate litigation alpha from macroeconomic healthcare factor risk.

## 3. Data Infrastructure & Stack Integration
*   **Massive API (Live Stream & Catalysts):** Real-time market data feed and AI-tagged SEC 8-K disclosures (Item 8.01) to identify real-time litigation filings during market hours.
*   **Databento (Historical Data Layer):** Nanosecond/millisecond-resolution historical OHLCV and tick data across target tickers to compute out-of-sample Sharpe ratio, turnover, and maximum drawdown across historical trial windows.
*   **CourtListener / PACER:** Docket and transcript feeds for the District of Delaware (D. Del.) and District of New Jersey (D.N.J.).

## 4. Signal Construction (The Gemini Alpha Engine)
Signals are derived from two distinct quantitative feeds:

**A. Historical Judicial Profiling**
Triggered upon docket assignment.
*   **The Metric:** Historical Patent Validity Score ($P_{bias} \in [-1.0, 1.0]$) quantifying a judge's historical rate of upholding innovator patent validity versus ruling in favor of generic invalidation.

**B. Unstructured Legal NLP Engine (Gemini API)**
Triggered when milestone filings hit the docket (Markman claim construction rulings, Motions for Summary Judgment, and Preliminary Injunction orders).
*   **Mechanism:** Feed dense legal orders into Gemini to parse patent claim language. Gemini outputs a structured JSON sentiment classification indicating whether the judge's construction favors the branded claim scope or broadens it for generic entry.

## 5. Execution Strategy (Webull API)
*   **Signal Aggregation:** A composite score combining $P_{bias}$ and Gemini filing sentiment determines position sizing.
*   **Order Routing:** Automatically trigger limit orders via the Webull Paper Trading API:
    *   **Long Branded / Short XPH:** Positive combined score (judge leans innovator + favorable Markman ruling).
    *   **Short Branded / Long XPH:** Negative combined score (judge leans generic challenger).
*   **Exit Conditions:** Close positions upon formal settlement disclosure or within 24 hours post-trial ruling to avoid extended post-verdict binary chop.
