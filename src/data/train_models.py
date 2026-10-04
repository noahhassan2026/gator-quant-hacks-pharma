import itertools
import json
import os
import re
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from build_labels import EXIT_RULES

# Step 4 of training: learn when to trade and how to exit.
# Input:  labels.csv (build_labels.py) + text_features.csv (oss_labeler.py)
#         + historical_sec_8k_2016_2024.csv (only its Items column)
# Output: direction_model.joblib, exit_model.joblib, model_config.json, tuning_results.csv
#
# 1. Walk-forward tuning on 2016-2024 only: for each validation year 2020-2024, train on every
#    earlier year and predict that year. Every model setting and trade cutoff is scored on
#    these out-of-sample predictions, with label returns net of costs and short borrow fees.
# 2. Retrain the best setting on all of 2016-2024 and save it.
# Filings from 2025 on are never touched here, so backtest.py stays an honest test.

FILINGS_CSV = "historical_sec_8k_2016_2024.csv"
DATA_END = "2024-12-31"
# Newer filings that live_learning.py adds once their outcome is known
LIVE_FILINGS_CSV = "filings_2025_2026.csv"
LIVE_LABELS_CSV = "labels_live.csv"
LIVE_TEXT_CSV = "test_text_features.csv"   # shared with backtest.py, so nothing is scored twice
VALID_YEARS = [2020, 2021, 2022, 2023, 2024]
GAP_DAYS = 45            # skip filings right before each validation year: their labels look into it
MIN_TRADES = 150         # a setting needs this many out-of-sample trades to be considered
MIN_GOOD_YEARS = 4       # ...and a positive average trade in at least 4 of the 5 validation years

# Settings tried. The first entry is the old fixed setting, kept for comparison.
OLD_SETTING = {"horizon": 10, "max_depth": 3, "learning_rate": 0.05, "min_samples_leaf": 20}
GRID = {"horizon": [5, 10, 20], "max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1], "min_samples_leaf": [20, 100]}
CUTOFFS = np.round(np.arange(0.52, 0.76, 0.02), 2)

EVENT_TYPES = [
    "patent_litigation", "settlement", "fda_decision", "clinical_trial_result", "earnings",
    "licensing_or_partnership", "merger_or_acquisition", "financing", "restructuring",
    "management_change", "delisting_or_compliance", "other",
]
# 8-K item codes as yes/no features (e.g. 2.02 earnings, 3.01 delisting notice, 8.01 other events)
ITEM_CODES = ["1.01", "1.02", "2.01", "2.02", "2.05", "2.06", "3.01", "5.02", "7.01", "8.01"]
FEATURES = (
    ["sentiment", "surprise", "materiality", "binary_catalyst_ahead",
     "pre_ret_5", "pre_ret_20", "vol_20", "log_dollar_volume", "log_price",
     "after_hours", "days_since_last_filing"]
    + [f"event_{e}" for e in EVENT_TYPES]
    + [f"item_{c.replace('.', '_')}" for c in ITEM_CODES]
)
RULES = list(EXIT_RULES)

# Short borrow fees. Webull charges daily: closing value x the stock's loan rate / 360, every
# calendar day the short is open. Historical per-stock rates aren't public, so the rate is
# estimated from the stock's average daily dollar volume (thinly traded biotechs cost far more).
# backtest.py uses the same tiers, or your real Webull rates if you put them in borrow_rates.csv.
BORROW_TIERS = [            # (min average daily $ volume, annual rate)
    (100_000_000, 0.005),   # large, liquid: easy to borrow
    (20_000_000, 0.02),
    (5_000_000, 0.08),
    (0, 0.25),              # small, thin: often hard to borrow
]
HEDGE_BORROW_RATE = 0.005   # XPH, shorted as the hedge on long trades


def borrow_rate(log_dollar_volume):
    """Estimated annual borrow rate from log10(average daily dollar volume)."""
    dv = 10 ** np.asarray(log_dollar_volume, dtype=float)
    return np.select([dv >= floor for floor, _ in BORROW_TIERS[:-1]],
                     [rate for _, rate in BORROW_TIERS[:-1]], default=BORROW_TIERS[-1][1])


def add_item_features(df, items):
    """items: the filing's 8-K item list, like "2.02,9.01", aligned with df."""
    items = items.fillna("").astype(str)
    for c in ITEM_CODES:
        df[f"item_{c.replace('.', '_')}"] = items.str.contains(rf"(?:^|[,\s]){re.escape(c)}(?:$|[,\s])").astype(int)
    return df


def add_text_features(df):
    df["binary_catalyst_ahead"] = df["binary_catalyst_ahead"].astype(str).str.lower().eq("true").astype(int)
    for e in EVENT_TYPES:
        df[f"event_{e}"] = (df["event_type"] == e).astype(int)
    return df


def load(include_live=False, data_end=DATA_END):
    """Labels + text scores + item codes. include_live adds the filings live_learning.py collected."""
    label_files, text_files, filing_files = ["labels.csv"], ["text_features.csv"], [FILINGS_CSV]
    if include_live and os.path.exists(LIVE_LABELS_CSV) and os.path.exists(LIVE_TEXT_CSV):
        label_files.append(LIVE_LABELS_CSV)
        text_files.append(LIVE_TEXT_CSV)
        filing_files.append(LIVE_FILINGS_CSV)
    labels = pd.concat([pd.read_csv(f) for f in label_files]).drop_duplicates("Accession", keep="last")
    text = pd.concat([pd.read_csv(f) for f in text_files]).drop_duplicates("Accession", keep="last")
    df = add_text_features(labels.merge(text, on="Accession", how="inner"))
    items = pd.concat([pd.read_csv(f, usecols=["Accession", "Items"]) for f in filing_files if os.path.exists(f)]
                      or [pd.DataFrame(columns=["Accession", "Items"])])
    items = items.drop_duplicates("Accession").set_index("Accession")["Items"]
    df = add_item_features(df, df["Accession"].map(items))
    df["EntryDay"] = pd.to_datetime(df["EntryDay"])
    if data_end is not None:
        df = df[df["EntryDay"] <= pd.Timestamp(data_end)]

    # Net every rule's return of borrow fees. Labels don't record how long each trade lasted, so
    # this charges the rule's full time limit (calendar days = trading days x 7/5): slightly
    # pessimistic for trades that hit their target or stop early.
    rate = borrow_rate(df["log_dollar_volume"])
    for name, (_, _, max_days) in EXIT_RULES.items():
        years = max_days * 7 / 5 / 360
        df[f"short_{name}"] -= rate * years
        df[f"long_{name}"] -= HEDGE_BORROW_RATE * years
    return df.sort_values("EntryDay").reset_index(drop=True)


def new_model(s):
    return HistGradientBoostingClassifier(max_depth=s["max_depth"], learning_rate=s["learning_rate"],
                                          min_samples_leaf=s["min_samples_leaf"], max_iter=200,
                                          l2_regularization=1.0, random_state=0)


def exit_training_rows(df):
    """Each filing appears twice, once per side; the target is the exit rule that paid best."""
    parts = []
    for side, prefix in ((1, "long"), (-1, "short")):
        part = df[FEATURES].copy()
        part["side"] = side
        part["best_rule"] = df[[f"{prefix}_{r}" for r in RULES]].to_numpy().argmax(axis=1)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    return out[FEATURES + ["side"]], out["best_rule"]


def fit_models(df, s):
    """Direction model: will the stock beat XPH over the next `horizon` trading days?
    Exit model: which rule on the menu works best for this kind of filing and side?"""
    direction = new_model(s).fit(df[FEATURES], (df[f"excess_{s['horizon']}d"] > 0).astype(int))
    Xe, ye = exit_training_rows(df)
    return direction, new_model(s).fit(Xe, ye)


def exit_choices(exit_model, df, features=None):
    """Rule index the exit model picks for each filing, if long and if short."""
    picks = []
    for side in (1, -1):
        X = df[features or FEATURES].copy()
        X["side"] = side
        picks.append(exit_model.predict(X).astype(int))
    return picks


def trade_returns(df, p_up, cutoff, rule_long, rule_short):
    """Net return of each trade taken at this cutoff, with the exit model's rule."""
    side = np.where(p_up >= cutoff, 1, np.where(p_up <= 1 - cutoff, -1, 0))
    rows = np.arange(len(df))
    longs = df[[f"long_{r}" for r in RULES]].to_numpy()[rows, rule_long]
    shorts = df[[f"short_{r}" for r in RULES]].to_numpy()[rows, rule_short]
    rets = np.where(side == 1, longs, shorts)
    return rets[side != 0], side[side != 0]


def out_of_fold(df, s):
    """Walk-forward predictions: each validation year comes from a model trained only on earlier years."""
    parts = []
    for year in VALID_YEARS:
        train = df[df["EntryDay"] <= pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=GAP_DAYS)]
        valid = df[df["EntryDay"].dt.year == year]
        direction, exit_model = fit_models(train, s)
        rule_long, rule_short = exit_choices(exit_model, valid)
        parts.append((valid, direction.predict_proba(valid[FEATURES])[:, 1], rule_long, rule_short))
    return parts


def evaluate(parts, cutoff):
    by_year, all_rets, all_sides = {}, [], []
    for (valid, p, rl, rs), year in zip(parts, VALID_YEARS):
        rets, sides = trade_returns(valid, p, cutoff, rl, rs)
        by_year[year] = rets.mean() if len(rets) else np.nan
        all_rets.append(rets)
        all_sides.append(sides)
    rets, sides = np.concatenate(all_rets), np.concatenate(all_sides)
    n = len(rets)
    score = rets.mean() / rets.std() * np.sqrt(n) if n >= 2 and rets.std() > 0 else -np.inf
    good_years = sum(1 for v in by_year.values() if v > 0)
    return {"cutoff": cutoff, "trades": n, "longs": int((sides == 1).sum()), "avg": rets.mean() if n else np.nan,
            "win": (rets > 0).mean() if n else np.nan, "score": score, "good_years": good_years,
            **{str(y): v for y, v in by_year.items()}}


def main():
    df = load()
    print(f"{len(df)} filings 2016-2024; walk-forward validation on {VALID_YEARS[0]}-{VALID_YEARS[-1]}")

    settings = [OLD_SETTING] + [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    settings = [s for i, s in enumerate(settings) if s not in settings[:i]]
    results, start = [], time.time()
    for i, s in enumerate(settings, 1):
        parts = out_of_fold(df, s)
        for cutoff in CUTOFFS:
            results.append({**s, **evaluate(parts, cutoff), "old": s == OLD_SETTING})
        print(f"  {i}/{len(settings)} settings tried ({(time.time() - start) / 60:.1f} min)", flush=True)

    res = pd.DataFrame(results)
    eligible = res[(res["trades"] >= MIN_TRADES) & (res["good_years"] >= MIN_GOOD_YEARS)]
    if eligible.empty:
        print("No setting was positive in enough years; falling back to the best overall score.")
        eligible = res[res["trades"] >= MIN_TRADES] if (res["trades"] >= MIN_TRADES).any() else res
    best = eligible.sort_values("score", ascending=False).iloc[0]
    old = res[res["old"]].sort_values("score", ascending=False).iloc[0]

    cols = ["horizon", "max_depth", "learning_rate", "min_samples_leaf", "cutoff", "trades", "longs",
            "avg", "win", "score"] + [str(y) for y in VALID_YEARS]
    fmt = lambda t: t[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}")
    print("\nTop settings (out-of-sample, net of 10 bps per side and borrow fees; years = avg trade):")
    print(fmt(eligible.sort_values("score", ascending=False).head(8)))
    print("\nOld setting at its best cutoff, for comparison:")
    print(fmt(old.to_frame().T))
    print("\nNote: picking the best of many settings flatters these numbers a bit. "
          "The 2025+ backtest is the honest test.")

    s = {k: (int(best[k]) if k != "learning_rate" else float(best[k])) for k in GRID}
    print(f"\nTraining final models on all of 2016-2024 with {s}, cutoff {best['cutoff']:.2f}...")
    direction, exit_model = fit_models(df, s)
    joblib.dump(direction, "direction_model.joblib")
    joblib.dump(exit_model, "exit_model.joblib")
    with open("model_config.json", "w") as fh:
        json.dump({"cutoff": round(float(best["cutoff"]), 2), "settings": s, "trained_through": DATA_END,
                   "features": FEATURES, "rules": RULES,
                   "exit_rules": {k: list(v) for k, v in EXIT_RULES.items()},
                   "borrow_tiers": BORROW_TIERS, "hedge_borrow_rate": HEDGE_BORROW_RATE}, fh, indent=2)
    res.to_csv("tuning_results.csv", index=False)
    print("Saved direction_model.joblib, exit_model.joblib, model_config.json and tuning_results.csv")


if __name__ == "__main__":
    main()
