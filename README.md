# Pharma 8-K Event Strategy (Gator Quant Hacks 2026, Track 03)

Trades U.S. pharma and biotech stocks right after they file a material 8-K with the SEC. An LLM (gpt-oss-20b) scores each filing's text; a gradient-boosted model combines that score with market features to decide long, short or skip, and a second model picks the exit rule. Every position is hedged with the pharma ETF XPH, so returns are measured against the sector.

- Hypothesis: [hypothesis.md](hypothesis.md)
- Quant note: [quant_note.pdf](quant_note.pdf)

## Setup

Python 3.11+.

```bash
pip install -r requirements.txt
cp .env.example .env      # fill in your own keys; .env is git-ignored
set -a; source .env; set +a
```

| Variable | What it is |
|---|---|
| `MASSIVE_API_KEY` | Massive (Polygon) market data: daily and minute bars |
| `OSS_API_KEY`, `OSS_BASE_URL`, `OSS_MODEL` | Any OpenAI-compatible endpoint serving gpt-oss-20b (we used UF NaviGator) |
| `SEC_USER_AGENT` | Your name and email; SEC EDGAR requires it |

No data or keys are in this repo. All data is downloaded by the scripts below.

## Reproduce the headline numbers

One command, from the repo root:

```bash
python run_all.py
```

It runs these steps in order (and skips any step whose output already exists):

```bash
# 1. Download the 8-K filings from SEC EDGAR (free)
python src/data/sec_edgar_8k.py                                   # 2016-2024 -> historical_sec_8k_2016_2024.csv
python src/data/sec_edgar_8k.py 2025 2026 filings_2025_2026.csv   # out-of-sample filings

# 2. Label 2016-2024 outcomes, score filing text, tune and train (walk-forward 2020-2024)
python src/data/run_training.py

# 3. Out-of-sample backtest, 2025-02-15 to today, minute-level entries
python src/data/backtest.py

# 4. IS vs OOS table, costs doubled, turnover, capacity, factor regression, Deflated Sharpe
python src/data/report_metrics.py
```

Outputs: `tuning_results.csv`, `backtest_trades.csv`, `backtest_equity.csv`, `backtest_report.png`, `report_metrics.json`.

LLM scores can differ slightly between runs, so numbers may move a little from the ones in the quant note.

## How it works

1. **Universe.** 8-Ks from companies with SIC codes 2834, 2835, 2836 and 8731, limited to material items (1.01, 2.02, 5.02, 7.01, 8.01 and others). Price at least $5 and at least $1M daily dollar volume.
2. **Timing.** A signal can only use the filing once the SEC has accepted it. Entry is the first regular-hours minute at least 1 minute after acceptance.
3. **Features.** LLM sentiment, surprise and materiality; 5- and 20-day returns versus XPH; volatility; dollar volume; price; after-hours flag; event type; 8-K item codes.
4. **Models.** `HistGradientBoostingClassifier` for direction (does the stock beat XPH over the next N days?) and for the exit rule (one of six target/stop/time menus).
5. **Validation.** Walk-forward: each year from 2020 to 2024 is predicted by a model trained only on earlier years, with a 45-day gap. 444 setting and cutoff combinations were compared; a setting had to be positive in at least 4 of 5 years.
6. **Costs.** 10 bps per side, plus Webull-style borrow fees on shorts (estimated from dollar volume, 0.5% to 25% a year; XPH 0.5%).
7. **Portfolio.** $100k, 5% per trade, at most 20 open positions and 100% gross.

## Files

| File | Purpose |
|---|---|
| `src/data/sec_edgar_8k.py` | Downloads pharma 8-Ks with acceptance times from EDGAR |
| `src/data/build_labels.py` | Features and outcome labels for 2016-2024 |
| `src/data/intraday_entry.py` | Minute-bar entry fills |
| `src/data/oss_labeler.py` | Scores filing text with gpt-oss-20b |
| `src/data/train_models.py` | Walk-forward tuning and final models |
| `src/data/run_training.py` | Runs labels, scoring and training |
| `src/data/backtest.py` | Out-of-sample backtest |
| `src/data/report_metrics.py` | Report statistics for the quant note |
| `src/data/live_learning.py` | Weekly retraining for live use (not used for the reported results) |

## Limitations

See the quant note. In short: tickers come from today's SEC ticker map, so delisted companies are missing (survivorship bias); borrow rates are estimates; the out-of-sample period was looked at more than once during development (disclosed in the note).
