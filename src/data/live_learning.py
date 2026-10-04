import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

import joblib
import numpy as np
import pandas as pd

import build_labels as bl
import oss_labeler as ol
import train_models as tm
from intraday_entry import ET

# Keeps the models learning from new filings. Run it on a schedule (weekly is plenty):
#   python src/data/live_learning.py              (download new filings first)
#   python src/data/live_learning.py --no-download (skip the SEC download)
#
# 1. Download new 8-Ks since 2025 (resumes; only new ones are fetched).
# 2. Once a filing's outcome is fully known (about 50 calendar days, the longest exit rule),
#    label it the same way build_labels.py labeled 2016-2024.
# 3. Score its text with the LLM (filings already scored by backtest.py are reused).
# 4. Retrain with the same settings and cutoff, and compare against the current model on the
#    most recent months neither has learned from. The new model replaces the old one only if it
#    trades at least as well there. Every replaced model is kept in models/ so you can roll back.
#
# It learns from every filing's outcome, not only the ones it traded: learning only from its own
# trades would teach it nothing about the filings it skipped.

MATURE_DAYS = 50          # calendar days until the 30-trading-day outcome window has closed
EVAL_MONTHS = 3           # how much recent data the new and current models are compared on
MIN_EVAL_TRADES = 30      # fewer trades than this is too little to judge; retrain anyway
PRICE_DIR = "prices_live"
MODEL_DIR = "models"
CHECKED_TXT = "labels_live_checked.txt"   # filings already labeled or found untradable
LOG_CSV = "live_learning_log.csv"
MODEL_FILES = ["direction_model.joblib", "exit_model.joblib", "model_config.json"]
WORKERS = 8


def download():
    import sec_edgar_8k as sec
    sec.OUTPUT_CSV = tm.LIVE_FILINGS_CSV
    sec.main(None, 2025, date.today().year)


def fresh_prices():
    """Daily bars up to today. Cached files from earlier days are refreshed."""
    os.makedirs(PRICE_DIR, exist_ok=True)
    for name in os.listdir(PRICE_DIR):
        path = os.path.join(PRICE_DIR, name)
        if date.fromtimestamp(os.path.getmtime(path)) < date.today():
            os.remove(path)
    bl.PRICE_DIR, bl.START, bl.END = PRICE_DIR, "2024-10-01", date.today().isoformat()


def label_new_filings():
    if not os.path.exists(tm.LIVE_FILINGS_CSV):
        sys.exit(f"{tm.LIVE_FILINGS_CSV} not found; run without --no-download first.")
    filings = pd.read_csv(tm.LIVE_FILINGS_CSV, usecols=["Accession", "Ticker", "AcceptedET", "CIK"])
    filings = filings.dropna(subset=["Ticker", "AcceptedET"])
    filings["TS"] = pd.to_datetime(filings["AcceptedET"], utc=True, errors="coerce")
    filings = filings.dropna(subset=["TS"])
    filings = filings[filings["TS"] > pd.Timestamp(tm.DATA_END, tz=ET)]

    # Days since the company's previous filing, counting the 2016-2024 history too
    hist = pd.read_csv(tm.FILINGS_CSV, usecols=["CIK", "AcceptedET"]).dropna()
    hist["TS"] = pd.to_datetime(hist["AcceptedET"], utc=True, errors="coerce")
    both = pd.concat([hist[["CIK", "TS"]].assign(Accession=None), filings[["Accession", "CIK", "TS"]]])
    both = both.dropna(subset=["TS"]).sort_values("TS")
    both["gap"] = both.groupby("CIK")["TS"].diff().dt.days.fillna(365).clip(upper=365)
    filings["days_since_last_filing"] = filings["Accession"].map(both.dropna(subset=["Accession"]).set_index("Accession")["gap"])

    checked = set(open(CHECKED_TXT).read().split()) if os.path.exists(CHECKED_TXT) else set()
    matured = filings["TS"] <= pd.Timestamp.now(tz=ET) - pd.Timedelta(days=MATURE_DAYS)
    todo = filings[matured & ~filings["Accession"].isin(checked)]
    print(f"Filings since 2025: {len(filings)}, outcome known: {matured.sum()}, new to label: {len(todo)}")
    if todo.empty:
        return 0

    fresh_prices()
    hedge = bl.daily_bars(bl.HEDGE)

    def safe_bars(t):
        try:
            return t, bl.daily_bars(t)
        except Exception as e:
            print(f"  [{t}: {e}]")
            return t, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        prices = dict(pool.map(safe_bars, sorted(todo["Ticker"].unique())))

    def safe_label(f):
        px = prices.get(f["Ticker"])
        if px is None or px.empty:
            return f["Accession"], None
        try:
            row = bl.label_filing(f, px, hedge)
        except Exception as e:
            print(f"  [{f['Accession']}: {e}]")
            return f["Accession"], "error"
        if row:
            row["days_since_last_filing"] = f["days_since_last_filing"]
        return f["Accession"], row

    rows, done = [], []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for acc, row in pool.map(safe_label, [f for _, f in todo.iterrows()]):
            if row == "error":
                continue  # try again next run
            done.append(acc)
            if row:
                rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(tm.LIVE_LABELS_CSV, mode="a", index=False, header=not os.path.exists(tm.LIVE_LABELS_CSV))
    with open(CHECKED_TXT, "a") as fh:
        fh.write("".join(a + "\n" for a in done))
    print(f"Labeled {len(rows)} tradable filings ({len(done) - len(rows)} untradable: price under ${bl.MIN_PRICE:.0f} or too little volume)\n")
    return len(rows)


def score_text():
    if not os.path.exists(tm.LIVE_LABELS_CSV):
        return
    ol.FILINGS_CSV, ol.LABELS_CSV, ol.OUTPUT_CSV, ol.ONLY_TRADABLE = (
        tm.LIVE_FILINGS_CSV, tm.LIVE_LABELS_CSV, tm.LIVE_TEXT_CSV, True)
    ol.main()


def trading_result(direction, exit_model, features, df, cutoff):
    p_up = direction.predict_proba(df[features])[:, 1]
    rule_long, rule_short = tm.exit_choices(exit_model, df, features)
    rets, _ = tm.trade_returns(df, p_up, cutoff, rule_long, rule_short)
    return len(rets), (rets.mean() if len(rets) else np.nan)


def retrain():
    cfg = json.load(open("model_config.json"))
    settings, cutoff = cfg.get("settings", tm.OLD_SETTING), cfg["cutoff"]
    trained_through = pd.Timestamp(cfg.get("trained_through", tm.DATA_END))

    df = tm.load(include_live=True, data_end=None)
    latest = df["EntryDay"].max()
    new_rows = int((df["EntryDay"] > trained_through).sum())
    print(f"Training data: {len(df)} filings through {latest.date()} ({new_rows} newer than the current model)")
    if new_rows == 0:
        print("Nothing new to learn from yet.")
        return

    # Compare on recent months the current model never trained on
    gap = pd.Timedelta(days=tm.GAP_DAYS)
    eval_start = max(trained_through + gap, latest - pd.DateOffset(months=EVAL_MONTHS))
    recent = df[df["EntryDay"] >= eval_start]
    challenger = tm.fit_models(df[df["EntryDay"] <= eval_start - gap], settings)
    current = (joblib.load("direction_model.joblib"), joblib.load("exit_model.joblib"))
    n_cur, avg_cur = trading_result(*current, cfg["features"], recent, cutoff)
    n_new, avg_new = trading_result(*challenger, tm.FEATURES, recent, cutoff)
    print(f"Since {eval_start.date()} ({len(recent)} filings), net avg trade: "
          f"current model {avg_cur:+.2%} over {n_cur} trades, retrained model {avg_new:+.2%} over {n_new} trades")

    if max(n_cur, n_new) < MIN_EVAL_TRADES:
        promote, why = True, f"under {MIN_EVAL_TRADES} recent trades to judge, so more data wins by default"
    elif n_new == 0:
        promote, why = False, "the retrained model made no trades in the recent window"
    elif n_cur == 0 or avg_new >= avg_cur:
        promote, why = True, "the retrained model traded at least as well on recent filings"
    else:
        promote, why = False, "the retrained model traded worse on recent filings"

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    if promote:
        # Keep the outgoing model so it can be restored, then train on everything
        backup = os.path.join(MODEL_DIR, f"replaced_{stamp}")
        os.makedirs(backup, exist_ok=True)
        for f in MODEL_FILES:
            shutil.copy(f, backup)
        direction, exit_model = tm.fit_models(df, settings)
        joblib.dump(direction, "direction_model.joblib")
        joblib.dump(exit_model, "exit_model.joblib")
        cfg.update({"features": tm.FEATURES, "trained_through": latest.date().isoformat(), "settings": settings})
        json.dump(cfg, open("model_config.json", "w"), indent=2)
        print(f"Updated the model ({why}); trained through {latest.date()}. Old model saved in {backup}")
    else:
        print(f"Kept the current model ({why}).")

    log = {"run": stamp, "filings": len(df), "new_filings": new_rows, "eval_from": eval_start.date().isoformat(),
           "current_trades": n_cur, "current_avg": avg_cur, "retrained_trades": n_new, "retrained_avg": avg_new,
           "updated": promote, "reason": why}
    pd.DataFrame([log]).to_csv(LOG_CSV, mode="a", index=False, header=not os.path.exists(LOG_CSV))


def main():
    for key in ("MASSIVE_API_KEY", "OSS_API_KEY"):
        if not os.environ.get(key):
            sys.exit(f"Set {key} in this terminal first.")
    if "--no-download" not in sys.argv:
        if not os.environ.get("SEC_USER_AGENT"):
            sys.exit("Set SEC_USER_AGENT first (your name and email), or run with --no-download.")
        print("Step 1/4: new SEC filings")
        download()
    print("\nStep 2/4: labeling filings whose outcome is now known")
    label_new_filings()
    print("Step 3/4: scoring filing text")
    score_text()
    print("\nStep 4/4: retraining")
    retrain()


if __name__ == "__main__":
    main()
