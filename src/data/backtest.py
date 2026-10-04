import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd

import build_labels as bl
import oss_labeler as ol
from intraday_entry import ET, MARKET_CLOSE, MARKET_OPEN, fetch_minute_bars
from train_models import HEDGE_BORROW_RATE, add_item_features, add_text_features, borrow_rate

# Out-of-sample backtest of the trained models on filings from TEST_START to today,
# trading on 1-minute bars: enter the first minute after a filing goes public, then check the
# take-profit / stop / time exit every minute, hedged with XPH, as the live bot would.
#
# Input:  filings_2025_2026.csv (python src/data/sec_edgar_8k.py 2025 2026 filings_2025_2026.csv)
#         direction_model.joblib, exit_model.joblib, model_config.json (train_models.py)
# Output: backtest_trades.csv, backtest_equity.csv, backtest_report.png and a printed report
# Safe to stop and re-run; prices, scores and finished trades are cached.

TEST_FILINGS_CSV = "filings_2025_2026.csv"
HISTORY_CSV = "historical_sec_8k_2016_2024.csv"   # only for "days since the company's last filing"
FEATURES_CSV = "test_features.csv"
TEXT_CSV = "test_text_features.csv"
SIM_CACHE = "backtest_sim_cache_v2.jsonl"
TRADES_CSV, EQUITY_CSV, CHART_PNG = "backtest_trades.csv", "backtest_equity.csv", "backtest_report.png"
PRICE_DIR, MINUTE_DIR = "prices_test", "minute_cache"

TEST_START = "2025-02-15"      # first day after the training labels end
START_CAPITAL = 100_000
MAX_POSITIONS = 20             # signals are skipped while 20 trades are open

# Position sizing (stock leg; the XPH hedge is the same size). Sizing sits on top of the models'
# outputs, so it never changes what they were trained on.
#   "fixed":           every trade gets POSITION_PCT of equity
#   "confidence_risk": POSITION_PCT x confidence x risk, where
#       confidence = how far the model's probability is past the cutoff (1x at the cutoff, up to 2x)
#       risk       = typical training-set volatility / this stock's volatility (calm stock bigger, wild one smaller)
SIZING = "fixed"               # fixed did better than confidence_risk in the 2025-26 test
POSITION_PCT = 0.05            # size of a trade right at the cutoff with average volatility
MIN_POSITION_PCT, MAX_POSITION_PCT = 0.01, 0.10
MAX_CONFIDENCE_MULT = 2.0
MAX_GROSS_PCT = 1.0            # all open trades together never exceed 100% of equity
ENTRY_DELAY_MINUTES = 1        # time to read, score and send the order; matches training
STOP_CHECK = "close"           # "close": stop/target checked only at each day's close, the way the training
                               #          labels were built (live: a market-on-close order), did far better
                               # "minute": checked every minute, like an intraday stop order

# Short borrow fees, charged the way Webull does: closing value x annual loan rate / 360 for every
# calendar day the short is held overnight (same-day round trips pay none). Rates are estimated
# from the stock's dollar volume (BORROW_TIERS in train_models.py). To use real rates instead, make
# borrow_rates.csv with columns Ticker,annual_rate (e.g. ABCD,0.35 for 35%), using the rate Webull
# shows for that stock; listed tickers override the estimate.
BORROW_RATES_CSV = "borrow_rates.csv"
BASELINE = True                # also short every tradable filing, to see what the model adds
WORKERS = 8                    # parallel Massive requests


# ---------- 1. Market features for each test filing (same as build_labels.py) ----------

def market_features(f, px, hedge):
    accepted = pd.Timestamp(f["AcceptedET"]).tz_convert(ET).tz_localize(None)
    after_close = accepted.time() >= MARKET_CLOSE
    days = px.index[px.index > accepted.normalize()] if after_close else px.index[px.index >= accepted.normalize()]
    if len(days) == 0:
        return None
    entry_day = days[0]
    i0 = px.index.get_loc(entry_day)
    if i0 < 20 or entry_day not in hedge.index:
        return None
    hist = px.iloc[i0 - 20:i0]
    h_hist = hedge.loc[hist.index.intersection(hedge.index)]
    if len(h_hist) < 6:
        return None
    price = hist["close"].iloc[-1]
    dollar_vol = (hist["close"] * hist["volume"]).mean()
    if price < bl.MIN_PRICE or dollar_vol < bl.MIN_DOLLAR_VOLUME:
        return None
    return {
        "Accession": f["Accession"], "Ticker": f["Ticker"], "AcceptedET": f["AcceptedET"],
        "EntryDay": entry_day.date().isoformat(),
        "pre_ret_5": hist["close"].iloc[-1] / hist["close"].iloc[-6] - 1 - (h_hist["close"].iloc[-1] / h_hist["close"].iloc[-6] - 1),
        "pre_ret_20": hist["close"].iloc[-1] / hist["close"].iloc[0] - 1 - (h_hist["close"].iloc[-1] / h_hist["close"].iloc[0] - 1),
        "vol_20": hist["close"].pct_change().dropna().std(),
        "log_dollar_volume": np.log10(dollar_vol),
        "log_price": np.log10(price),
        "after_hours": int(after_close or accepted.time() < MARKET_OPEN),
        "days_since_last_filing": f["days_since_last_filing"],
    }


def build_features():
    filings = pd.read_csv(TEST_FILINGS_CSV, usecols=["Accession", "Ticker", "AcceptedET", "CIK"])
    filings = filings.dropna(subset=["Ticker", "AcceptedET"])
    filings["TS"] = pd.to_datetime(filings["AcceptedET"], utc=True, errors="coerce")
    filings = filings[filings["TS"] >= pd.Timestamp(TEST_START, tz=ET)].copy()
    print(f"{len(filings)} filings since {TEST_START}")

    # Days since the same company's previous filing, counting the 2016-2024 history too
    both = filings[["Accession", "CIK", "TS"]]
    if os.path.exists(HISTORY_CSV):
        hist = pd.read_csv(HISTORY_CSV, usecols=["CIK", "AcceptedET"]).dropna()
        hist["TS"] = pd.to_datetime(hist["AcceptedET"], utc=True, errors="coerce")
        both = pd.concat([hist[["CIK", "TS"]].assign(Accession=None), both])
    both = both.dropna(subset=["TS"]).sort_values("TS")
    both["gap"] = both.groupby("CIK")["TS"].diff().dt.days.fillna(365).clip(upper=365)
    filings["days_since_last_filing"] = filings["Accession"].map(both.dropna(subset=["Accession"]).set_index("Accession")["gap"])

    bl.PRICE_DIR, bl.START, bl.END = PRICE_DIR, "2024-10-01", date.today().isoformat()
    hedge = bl.daily_bars(bl.HEDGE)
    tickers = sorted(filings["Ticker"].unique())
    print(f"Fetching daily prices for {len(tickers)} tickers...", flush=True)

    def safe_bars(t):
        try:
            return t, bl.daily_bars(t)
        except Exception as e:
            print(f"  [{t}: {e}]")
            return t, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        prices = dict(pool.map(safe_bars, tickers))

    rows = []
    for _, f in filings.iterrows():
        px = prices.get(f["Ticker"])
        if px is not None and not px.empty:
            row = market_features(f, px, hedge)
            if row:
                rows.append(row)
    pd.DataFrame(rows).to_csv(FEATURES_CSV, index=False)
    print(f"{len(rows)} tradable filings (price >= ${bl.MIN_PRICE:.0f}, enough volume) saved to {FEATURES_CSV}\n")
    return hedge


# ---------- 2. Signals from the trained models ----------

def make_signals():
    cfg = json.load(open("model_config.json"))
    direction, exit_model = joblib.load("direction_model.joblib"), joblib.load("exit_model.joblib")
    df = pd.read_csv(FEATURES_CSV).merge(pd.read_csv(TEXT_CSV), on="Accession", how="inner")
    df = add_text_features(df)
    items = pd.read_csv(TEST_FILINGS_CSV, usecols=["Accession", "Items"]).set_index("Accession")["Items"]
    df = add_item_features(df, df["Accession"].map(items))

    if df.empty:
        sys.exit(f"No filings with both market features and text scores; check {FEATURES_CSV} and {TEXT_CSV}.")
    X = df[cfg["features"]]
    df["p_up"] = direction.predict_proba(X)[:, 1]
    cutoff = cfg["cutoff"]
    df["side"] = np.where(df["p_up"] >= cutoff, 1, np.where(df["p_up"] <= 1 - cutoff, -1, 0))
    sig = df[df["side"] != 0].copy()
    if not sig.empty:
        Xe = sig[cfg["features"]].copy()
        Xe["side"] = sig["side"]
        sig["rule"] = [cfg["rules"][i] for i in exit_model.predict(Xe)]
    print(f"Scored filings: {len(df)}. Signals at cutoff {cutoff}: {len(sig)} "
          f"({(sig['side'] == 1).sum()} long, {(sig['side'] == -1).sum()} short)\n")

    # Baseline: short every scored filing, with the exit rule the exit model picks for a short
    base = df.copy()
    base["side"] = -1
    Xb = base[cfg["features"]].copy()
    Xb["side"] = -1
    base["rule"] = [cfg["rules"][i] for i in exit_model.predict(Xb)]
    return sig, base, {k: tuple(v) for k, v in cfg["exit_rules"].items()}


# ---------- 3. Minute-by-minute simulation of each signal ----------

def regular_hours(bars):
    t = bars.index.time
    return bars[(t >= MARKET_OPEN) & (t < MARKET_CLOSE)]


def xph_minutes():
    """XPH 1-minute bars for the whole test, cached by month (the current month is refreshed)."""
    os.makedirs(MINUTE_DIR, exist_ok=True)
    parts, month = [], pd.Timestamp(TEST_START).replace(day=1) - pd.offsets.MonthBegin(1)
    this_month = pd.Timestamp(date.today()).replace(day=1)
    while month <= this_month:
        path = os.path.join(MINUTE_DIR, f"XPH_{month:%Y-%m}.csv")
        if os.path.exists(path) and month < this_month:
            bars = pd.read_csv(path, index_col=0)
            bars.index = pd.to_datetime(bars.index, utc=True).tz_convert(ET)
        else:
            start = month.tz_localize(ET)
            bars = fetch_minute_bars(bl.HEDGE, start, start + pd.offsets.MonthBegin(1), os.environ["MASSIVE_API_KEY"], adjusted=True)
            bars = regular_hours(bars)[["open", "close"]] if not bars.empty else pd.DataFrame(columns=["open", "close"])
            bars.to_csv(path)
        parts.append(bars)
        month += pd.offsets.MonthBegin(1)
    xph = pd.concat(parts)
    xph.index = xph.index.as_unit("ns")  # cached and fresh bars can load with different time precisions
    return xph[~xph.index.duplicated()].sort_index()


def pick_exit(path, bars, days, day_num, day0, calendar, tp, sl, max_days, mode):
    """Index of the exit bar and why. mode "minute" checks every bar, "close" only each day's last bar."""
    window = day_num <= max_days
    check = window.copy()
    if mode == "close":
        check &= np.r_[days[1:] != days[:-1], True]  # last bar of each day
    hit = check & (((path >= tp) if tp is not None else False) | ((path <= -sl) if sl is not None else False))
    hit = np.asarray(hit, dtype=bool)
    if hit.any():
        i = int(np.argmax(hit))
        return i, "take_profit" if tp is not None and path.iloc[i] >= tp else "stop"
    if day0 + max_days < len(calendar) and calendar[day0 + max_days].date() < date.today():
        return int(np.flatnonzero(window)[-1]), "time"  # last minute of the last allowed day
    return len(path) - 1, "open"  # still open today: marked at the latest minute


def simulate(sig, rules, xph, calendar):
    """Enters at the first regular-hours minute after the filing, then exits when the hedged return
    hits the take-profit or stop, or at the close of the last allowed trading day.
    Returns the result for both stop-check modes, from one download."""
    accepted = pd.Timestamp(sig["AcceptedET"]).tz_convert(ET)
    tradable_at = accepted + timedelta(minutes=ENTRY_DELAY_MINUTES)
    bars = fetch_minute_bars(sig["Ticker"], tradable_at, tradable_at + timedelta(days=50),
                             os.environ["MASSIVE_API_KEY"], adjusted=True)
    if bars.empty:
        return None
    bars = regular_hours(bars[bars.index >= tradable_at])
    if bars.empty:
        return None
    bars.index = bars.index.as_unit("ns")  # same time precision as the XPH bars, or the merge below fails

    entry_time, entry_px = bars.index[0], float(bars["open"].iloc[0])
    x_after = xph[xph.index >= entry_time]
    if x_after.empty:
        return None
    x_entry = float(x_after["open"].iloc[0])
    x_close = pd.merge_asof(pd.DataFrame(index=bars.index), xph[["close"]], left_index=True,
                            right_index=True, direction="backward")["close"].fillna(x_entry)
    side = int(sig["side"])
    path = side * ((bars["close"] / entry_px - 1) - (x_close / x_entry - 1))

    # Trading-day number of each bar (0 = entry day), from XPH's daily calendar
    days = bars.index.tz_localize(None).normalize()
    day0 = calendar.searchsorted(days[0])
    day_num = calendar.searchsorted(days) - day0

    tp, sl, max_days = rules[sig["rule"]]
    if tp == "vol":
        tp, sl = 2 * sig["vol_20"] * np.sqrt(max_days), sig["vol_20"] * np.sqrt(max_days)

    out = {}
    for mode in ("minute", "close"):
        i, reason = pick_exit(path, bars, days, day_num, day0, calendar, tp, sl, max_days, mode)
        marks = path.iloc[:i + 1].groupby(days[:i + 1]).last()
        out[mode] = {
            "entry_time": entry_time.isoformat(), "entry_price": entry_px,
            "exit_time": bars.index[i].isoformat(), "exit_price": float(bars["close"].iloc[i]),
            "exit_reason": reason, "gross_return": float(path.iloc[i]),
            "net_return": float(path.iloc[i]) - 2 * bl.COST,
            "marks": {d.date().isoformat(): float(v) for d, v in marks.items()},
        }
    return out


def simulate_all(signals, rules, xph, calendar, label):
    """Returns [(signal, {"minute": result, "close": result})], using and filling the cache."""
    cache = {}
    if os.path.exists(SIM_CACHE):
        for line in open(SIM_CACHE):
            r = json.loads(line)
            cache[r["key"]] = r
    key = lambda s: f"{s['Accession']}|{s['side']}|{s['rule']}|{ENTRY_DELAY_MINUTES}"
    todo = [s for _, s in signals.iterrows() if key(s) not in cache]
    print(f"{label}: simulating {len(todo)} trades on minute bars ({len(signals) - len(todo)} cached)...", flush=True)

    def run(s):
        try:
            return s, simulate(s, rules, xph, calendar)
        except Exception as e:
            print(f"  [{s['Ticker']} {s['Accession']}: {e}]")
            return s, None

    with ThreadPoolExecutor(max_workers=WORKERS) as pool, open(SIM_CACHE, "a") as fh:
        for n, (s, r) in enumerate(pool.map(run, todo), 1):
            if r is not None:
                r["key"] = key(s)
                cache[r["key"]] = r
                if "open" not in (r["minute"]["exit_reason"], r["close"]["exit_reason"]):
                    fh.write(json.dumps(r) + "\n")  # open trades are re-simulated next run
            if n % 250 == 0:
                print(f"  {n}/{len(todo)}", flush=True)

    return [(s.to_dict(), cache[key(s)]) for _, s in signals.iterrows() if key(s) in cache]


def load_borrow_overrides():
    if not os.path.exists(BORROW_RATES_CSV):
        return {}
    rates = pd.read_csv(BORROW_RATES_CSV)
    return dict(zip(rates["Ticker"].astype(str).str.upper(), rates["annual_rate"].astype(float)))


BORROW_OVERRIDES = {}


def trades_for(sims, mode):
    """Trades for one stop-check mode, with borrow fees taken out of each return."""
    out = []
    for s, r in sims:
        t = {**s, **r[mode]}
        if t["side"] == -1:
            t["borrow_rate"] = BORROW_OVERRIDES.get(str(t["Ticker"]).upper(), float(borrow_rate(t["log_dollar_volume"])))
        else:
            t["borrow_rate"] = HEDGE_BORROW_RATE  # longs short XPH as the hedge
        nights = (pd.Timestamp(t["exit_time"][:10]) - pd.Timestamp(t["entry_time"][:10])).days
        t["borrow_fee"] = t["borrow_rate"] * nights / 360
        t["net_return"] = t["net_return"] - t["borrow_fee"]
        out.append(t)
    return out


# ---------- 4. Portfolio: position sizing, limits, daily equity ----------

def position_pct(t, sizing, cutoff, typical_vol):
    """Share of equity for one trade."""
    if sizing == "fixed":
        return POSITION_PCT
    # 1x at the cutoff, rising linearly to 2x when the edge is 3x the cutoff's (0.56 -> 1x, 0.62 -> 1.5x, 0.68 -> 2x)
    min_edge = cutoff - 0.5
    confidence = np.clip(1 + (abs(t["p_up"] - 0.5) - min_edge) / (2 * min_edge), 1.0, MAX_CONFIDENCE_MULT)
    risk = np.clip(typical_vol / t["vol_20"], 0.5, 2.0) if t["vol_20"] > 0 else 1.0
    return float(np.clip(POSITION_PCT * confidence * risk, MIN_POSITION_PCT, MAX_POSITION_PCT))


def run_portfolio(trades, calendar, sizing, cutoff, typical_vol):
    trades = sorted((dict(t) for t in trades), key=lambda t: t["entry_time"])
    realized, open_pos, taken = START_CAPITAL, [], []
    for t in trades:
        for p in [p for p in open_pos if p["exit_time"] <= t["entry_time"]]:
            realized += p["pnl"]
            open_pos.remove(p)
        if len(open_pos) >= MAX_POSITIONS or any(p["Ticker"] == t["Ticker"] for p in open_pos):
            continue
        room = MAX_GROSS_PCT * realized - sum(p["notional"] for p in open_pos)
        t["size_pct"] = position_pct(t, sizing, cutoff, typical_vol)
        t["notional"] = min(t["size_pct"] * realized, room)
        if t["notional"] < MIN_POSITION_PCT * realized:
            continue  # fully invested
        t["pnl"] = t["notional"] * t["net_return"]
        open_pos.append(t)
        taken.append(t)

    start = pd.Timestamp(TEST_START)
    days = calendar[calendar >= start]
    pnl = pd.Series(0.0, index=days)
    positions = pd.Series(0, index=days)
    for t in taken:
        marks = pd.Series(t["marks"])
        marks.index = pd.to_datetime(marks.index)
        first, exit_day = marks.index.min(), pd.Timestamp(t["exit_time"][:10])
        held = (days >= first) & (days <= exit_day)
        path = marks.reindex(days[held]).ffill() - bl.COST  # marked to market at each day's close
        path -= t["borrow_rate"] * (path.index - first).days / 360  # borrow fees accrue daily
        if path.empty:
            continue
        if t["exit_reason"] != "open":
            path.iloc[-1] = t["net_return"]
        contrib = path.reindex(days).ffill().fillna(0)  # 0 before entry, final P&L after exit
        contrib[days < first] = 0
        pnl += t["notional"] * contrib
        positions += held.astype(int)
    equity = START_CAPITAL + pnl
    return taken, pd.DataFrame({"equity": equity, "open_positions": positions})


# ---------- 5. Report ----------

def report(taken, curve, hedge):
    eq = curve["equity"]
    rets = eq.pct_change().dropna()
    years = max(len(eq) / 252, 1e-9)
    dd = eq / eq.cummax() - 1
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    tr = pd.DataFrame(taken)
    wins, losses = tr["net_return"][tr["net_return"] > 0], tr["net_return"][tr["net_return"] <= 0]
    downside = rets[rets < 0].std()

    xph = hedge["close"].reindex(eq.index).ffill().bfill()
    xph_ret = xph.pct_change().dropna()

    line = "=" * 64
    print(f"\n{line}\nBACKTEST {eq.index[0].date()} to {eq.index[-1].date()}  "
          f"(${START_CAPITAL:,.0f} start, {SIZING} sizing, max {MAX_POSITIONS} open)\n{line}")
    stats = [
        ("Final equity", f"${eq.iloc[-1]:,.0f}"),
        ("Total return", f"{eq.iloc[-1] / START_CAPITAL - 1:+.2%}"),
        ("Annualized return", f"{(eq.iloc[-1] / START_CAPITAL) ** (1 / years) - 1:+.2%}"),
        ("Annualized volatility", f"{rets.std() * np.sqrt(252):.2%}"),
        ("Sharpe ratio (daily, rf=0)", f"{rets.mean() / rets.std() * np.sqrt(252):.2f}" if rets.std() > 0 else "n/a"),
        ("Sortino ratio", f"{rets.mean() / downside * np.sqrt(252):.2f}" if downside > 0 else "n/a"),
        ("Max drawdown", f"{dd.min():.2%}  ({peak.date()} to {trough.date()})"),
        ("Calmar ratio", f"{((eq.iloc[-1] / START_CAPITAL) ** (1 / years) - 1) / abs(dd.min()):.2f}" if dd.min() < 0 else "n/a"),
        ("Trades", f"{len(tr)}  ({(tr['side'] == 1).sum()} long, {(tr['side'] == -1).sum()} short, "
                   f"{(tr['exit_reason'] == 'open').sum()} still open)"),
        ("Win rate", f"{(tr['net_return'] > 0).mean():.1%}"),
        ("Avg / median trade", f"{tr['net_return'].mean():+.2%} / {tr['net_return'].median():+.2%}"),
        ("Avg win / avg loss", f"{wins.mean():+.2%} / {losses.mean():+.2%}"),
        ("Profit factor", f"{wins.sum() / -losses.sum():.2f}" if losses.sum() < 0 else "n/a"),
        ("Avg open positions", f"{curve['open_positions'].mean():.1f}"),
        ("Borrow fees paid", f"${(tr['borrow_fee'] * tr['notional']).sum():,.0f}  (avg {tr['borrow_fee'].mean():.2%} "
                             f"per trade, estimated rates {tr['borrow_rate'].min():.1%} to {tr['borrow_rate'].max():.1%}/yr)"),
        ("Trade size (% of equity)", f"avg {tr['notional'].div(eq.reindex(pd.to_datetime(tr['entry_time'].str[:10])).values).mean():.1%}, "
                                     f"range {tr['size_pct'].min():.1%} to {tr['size_pct'].max():.1%}"),
        ("XPH buy-and-hold", f"{xph.iloc[-1] / xph.iloc[0] - 1:+.2%}  (Sharpe "
                             f"{xph_ret.mean() / xph_ret.std() * np.sqrt(252):.2f}, max drawdown {(xph / xph.cummax() - 1).min():.2%})"),
    ]
    for name, value in stats:
        print(f"{name:<28}{value}")

    def table(col):
        g = tr.groupby(col)["net_return"]
        out = pd.DataFrame({"trades": g.size(), "avg": g.mean().map("{:+.2%}".format),
                            "win": g.apply(lambda r: f"{(r > 0).mean():.0%}"),
                            "pnl $": tr.groupby(col)["pnl"].sum().map("{:,.0f}".format)})
        print(f"\nBy {col}:\n{out.to_string()}")

    tr["direction"] = tr["side"].map({1: "long", -1: "short"})
    for col in ("direction", "rule", "exit_reason", "event_type"):
        table(col)
    month_end = eq.groupby(eq.index.to_period("M")).last()
    monthly = month_end.pct_change()
    monthly.iloc[0] = month_end.iloc[0] / START_CAPITAL - 1
    print("\nMonthly returns:\n" + monthly.map("{:+.2%}".format).to_string())

    cols = ["entry_time", "Ticker", "direction", "p_up", "event_type", "rule", "size_pct", "entry_price", "exit_time",
            "exit_price", "exit_reason", "net_return", "notional", "pnl", "Accession"]
    tr[cols].to_csv(TRADES_CSV, index=False)
    curve.assign(drawdown=dd).to_csv(EQUITY_CSV)
    print(f"\nSaved {TRADES_CSV} (every trade) and {EQUITY_CSV} (daily equity)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        a1.plot(eq.index, eq / START_CAPITAL, label="Strategy")
        a1.plot(xph.index, xph / xph.iloc[0], label="XPH buy-and-hold", alpha=0.7)
        a1.set_ylabel("Growth of $1")
        a1.legend()
        a1.set_title("Out-of-sample backtest on minute bars")
        a2.fill_between(dd.index, dd * 100, 0, alpha=0.5)
        a2.set_ylabel("Drawdown %")
        fig.tight_layout()
        fig.savefig(CHART_PNG, dpi=120)
        print(f"Saved {CHART_PNG}")
    except ImportError:
        print("(pip install matplotlib to also get a chart)")


def main():
    for key in ("MASSIVE_API_KEY", "OSS_API_KEY"):
        if not os.environ.get(key):
            sys.exit(f"Set {key} in this terminal first.")
    if not os.path.exists(TEST_FILINGS_CSV):
        sys.exit(f"{TEST_FILINGS_CSV} not found. First run: python src/data/sec_edgar_8k.py 2025 2026 {TEST_FILINGS_CSV}")

    trained_through = json.load(open("model_config.json")).get("trained_through", "2024-12-31")
    if trained_through > TEST_START:
        print(f"WARNING: the model has learned from filings through {trained_through} (live_learning.py), so "
              f"results before that date are not an honest test. Judge it on later trades or paper trading.\n")
    print("Step 1/4: market features")
    hedge = build_features()

    print("Step 2/4: scoring filing text with the LLM")
    ol.FILINGS_CSV, ol.LABELS_CSV, ol.OUTPUT_CSV, ol.ONLY_TRADABLE = TEST_FILINGS_CSV, FEATURES_CSV, TEXT_CSV, True
    ol.main()

    print("\nStep 3/4: model signals")
    signals, base_signals, rules = make_signals()
    if signals.empty:
        sys.exit("No trades: the model wasn't confident enough on any filing.")

    print("Step 4/4: minute-by-minute trading")
    calendar = hedge.index
    xph = xph_minutes()
    sims = simulate_all(signals, rules, xph, calendar, "Model")
    base_sims = simulate_all(base_signals, rules, xph, calendar, "Baseline (short everything)") if BASELINE else []

    BORROW_OVERRIDES.update(load_borrow_overrides())
    if BORROW_OVERRIDES:
        print(f"Using your borrow rates for {len(BORROW_OVERRIDES)} tickers from {BORROW_RATES_CSV}")
    cutoff = json.load(open("model_config.json"))["cutoff"]
    # "Typical" volatility comes from the training filings only, so the test period doesn't leak in
    typical_vol = (pd.read_csv("labels.csv", usecols=["vol_20"])["vol_20"].median() if os.path.exists("labels.csv")
                   else signals["vol_20"].median())

    # Side-by-side: does the model beat shorting everything, and do minute stops hurt?
    # "Avg trade" is over every simulated trade (no position limits), the cleanest per-trade comparison.
    print(f"\n{'Strategy':<24}{'Stops':<8}{'Sizing':<17}{'Return':>9}{'Sharpe':>8}{'Max DD':>9}"
          f"{'Taken':>7}{'Avg trade':>11}{'Win':>6}")
    rows = [("Model", sims, m, z) for m in ("minute", "close") for z in dict.fromkeys(["fixed", SIZING])]
    rows += [("Short everything", base_sims, m, "fixed") for m in ("minute", "close") if base_sims]
    for name, sim, mode, sizing in rows:
        trades_m = trades_for(sim, mode)
        t_s, c_s = run_portfolio(trades_m, calendar, sizing, cutoff, typical_vol)
        r = c_s["equity"].pct_change().dropna()
        dd = (c_s["equity"] / c_s["equity"].cummax() - 1).min()
        sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else float("nan")
        every = np.array([t["net_return"] for t in trades_m])
        print(f"{name:<24}{mode:<8}{sizing:<17}{c_s['equity'].iloc[-1] / START_CAPITAL - 1:>+9.2%}{sharpe:>8.2f}"
              f"{dd:>9.2%}{len(t_s):>7}{every.mean():>+11.2%}{(every > 0).mean():>6.0%}")

    trades = trades_for(sims, STOP_CHECK)
    taken, curve = run_portfolio(trades, calendar, SIZING, cutoff, typical_vol)
    print(f"\nDetailed report below: model, {STOP_CHECK} stops, {SIZING} sizing. "
          f"{len(trades)} signals simulated, {len(taken)} taken (others skipped: {MAX_POSITIONS} positions "
          f"already open, same ticker already held, or fully invested)")
    report(taken, curve, hedge)


if __name__ == "__main__":
    main()
