# Hypothesis: Hatch-Waxman Legal Alpha Pipeline

## 1. Core Hypothesis Statement
Market makers efficiently price the initiation of pharmaceutical patent litigation, but inefficiently price the specific judicial assignments and interim milestone filings in Hatch-Waxman ANDA (Abbreviated New Drug Application) lawsuits. By combining historical judicial win-rate analysis with Natural Language Processing (NLP) on milestone court filings, we can accurately predict patent validity outcomes and generate statistically significant directional alpha.

## 2. Asset Universe & Risk Management
*   **Target Assets:** U.S. publicly traded branded pharmaceutical companies (Innovators) facing generic patent challenges.
*   **Risk Neutralization:** Sector-neutral hedging utilizing the SPDR S&P Pharmaceuticals ETF (XPH) to isolate legal alpha from macroeconomic healthcare beta.

## 3. Data Architecture (Massive Track Integration)
*   **PACER / CourtListener API:** High-throughput scraping of daily docket updates in the District of Delaware (D. Del.) and District of New Jersey (D.N.J.).
*   **SEC EDGAR Database:** 8-K Item 8.01 (Other Events) cross-referencing to map legal entities to tickers via CIK.
*   **USPTO PTAB Data:** Patent Trial and Appeal Board inter partes review (IPR) petitions.

## 4. Signal Construction (The Edge)
The model generates continuous alpha signals based on two distinct data pillars:

**A. Judicial Profiling Signal**
Triggered immediately upon case assignment.
*   **Metric:** The "Judge Validity Score" — A normalized historical ratio of the assigned federal judge's propensity to uphold patent validity (favoring branded pharma) vs. invalidating patents (favoring generics).

**B. Interim Filing Sentiment Signal**
Triggered by the filing of specific, high-leverage court documents.
*   **Markman Hearings (Claim Construction):** NLP sentiment analysis on the judge's claim construction order. Because the Markman ruling dictates how patent terms are defined, a favorable construction order is highly predictive of a final trial win. 
*   **Motions for Summary Judgment (MSJ):** Scoring the frequency and language of MSJ denials.
*   **Preliminary Injunctions (PI):** Real-time scraping of PI grants, which block generic manufacturers from entering the market "at-risk" during the trial.

## 5. Execution Strategy
*   **Entry 1 (Judge Assignment):** Initiate scaled long/short equity positions the day a judge with a strong historical bias (>0.7 Alpha Score) is assigned to a high-impact case.
*   **Entry 2 (Filing Catalyst):** Purchase options straddles 5 days prior to a scheduled Markman hearing to capture the implied volatility (IV) expansion.
*   **Exit:** Close directional equity positions either immediately upon a settlement disclosure or 24 hours post-trial ruling to avoid post-market binary drawdown risk.
