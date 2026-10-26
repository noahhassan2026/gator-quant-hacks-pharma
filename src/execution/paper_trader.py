import argparse
import html
import json
import os
import re
import sys
import time
import uuid
from datetime import date, datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()  # before the imports below: sec_edgar_8k reads SEC_USER_AGENT when it loads
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
import build_labels as bl  # noqa: E402
import oss_labeler as ol  # noqa: E402
import sec_edgar_8k as sec  # noqa: E402
from backtest import MAX_GROSS_PCT, MAX_POSITIONS, MIN_POSITION_PCT, POSITION_PCT, ENTRY_DELAY_MINUTES  # noqa: E402
from intraday_entry import ET, MARKET_CLOSE, MARKET_OPEN, fetch_minute_bars  # noqa: E402
from train_models import add_item_features, add_text_features  # noqa: E402

# Paper trades the strategy live on Webull's paper trading (sandbox) account, the same way
# backtest.py simulates it:
#   1. Watch SEC EDGAR's live 8-K feed. Keep pharma/biotech companies (PHARMA_SIC) and
#      market-moving items (MATERIAL_ITEMS), exactly as sec_edgar_8k.py does for training.
#   2. Score the filing text with the LLM (oss_labeler.py), build the market features from the
#      20 trading days before entry (Massive daily bars), and run the trained direction and exit models.
#   3. Enter at the first regular-hours minute at least 1 minute after the SEC accepted the filing
#      (filings outside 9:30-16:00 ET enter at the next open), with a market order for the stock and
#      the opposite position in XPH for the same dollar amount.
#   4. A few minutes before each close, check every open trade's hedged return against its exit
#      rule (take-profit, stop, time limit), the "close" stop check the backtest uses, and close it.
#
# Commands (run from the repo root, after python run_all.py has trained the models):
#   python src/execution/paper_trader.py check            test the Webull keys and order permissions
#   python src/execution/paper_trader.py run              trade until stopped (Ctrl+C)
#   python src/execution/paper_trader.py run --dry-run    same, but log orders instead of sending them
#   python src/execution/paper_trader.py once             one pass, for running from cron
#   python src/execution/paper_trader.py status           open trades with their current hedged return
#
# Needs WEBULL_APP_KEY and WEBULL_APP_SECRET (paper trading keys) besides the keys run_all.py uses.
# Output: paper_trades.csv (closed trades), paper_signals.csv (every filing it looked at),
#         paper_state.json (open trades and the XPH hedge; delete it only with no trades open)

WEBULL_REGION = "us"
WEBULL_ENDPOINT = os.environ.get("WEBULL_ENDPOINT") or "api.sandbox.webull.com"  # paper trading host
STATE_JSON, TRADES_CSV, SIGNALS_CSV = "paper_state.json", "paper_trades.csv", "paper_signals.csv"
PRICE_DIR = "prices_paper"
MODEL_FILES = ["direction_model.joblib", "exit_model.joblib", "model_config.json"]

POLL_SECONDS = 30             # how often to read the EDGAR feed
FEED_URL = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb="
            "&owner=include&start={start}&count=100&output=atom")
FEED_PAGES = 3                # pages of 100 read when a whole page is new (busy earnings evenings)
MAX_LATE_MINUTES = 15         # skip a filing we can't trade within 15 min of when the backtest would have
EXIT_CHECK_MINUTES = 10       # exits are checked and sent this many minutes before the close
ORDER_TIMEOUT_SECONDS = 30    # a market order not filled by then is cancelled
DRY_RUN_CAPITAL = 100_000     # starting equity in --dry-run; Webull's own balance is used otherwise

# NYSE holidays and 1 pm early closes (NYSE calendar announced Dec 2025). Add 2028 when it comes.
HOLIDAYS = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03",
    "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18", "2027-07-05",
    "2027-09-06", "2027-11-25", "2027-12-24",
}
EARLY_CLOSES = {"2026-11-27", "2026-12-24", "2027-11-26"}

# 8-K item names as EDGAR's filing header spells them, for the MATERIAL_ITEMS filter when the
# submissions API hasn't listed a brand-new filing's item codes yet
ITEM_NAMES = {
    "1.01": "entry into a material definitive agreement",
    "1.02": "termination of a material definitive agreement",
    "1.03": "bankruptcy or receivership",
    "2.01": "completion of acquisition or disposition of assets",
    "2.02": "results of operations and financial condition",
    "2.05": "costs associated with exit or disposal activities",
    "2.06": "material impairments",
    "3.01": "notice of delisting or failure to satisfy",
    "4.02": "non-reliance on previously issued financial statements",
    "7.01": "regulation fd disclosure",
    "8.01": "other events",
}


class OrderError(Exception):
    pass


def log(msg):
    print(f"{datetime.now(ET):%Y-%m-%d %H:%M:%S} {msg}", flush=True)


# ---------- Market calendar ----------

def is_trading_day(d):
    return d.weekday() < 5 and d.isoformat() not in HOLIDAYS


def session(d):
    """Regular-hours open and close of a trading day, as ET timestamps."""
    close = "13:00" if d.isoformat() in EARLY_CLOSES else MARKET_CLOSE.strftime("%H:%M")
    return (pd.Timestamp(f"{d} {MARKET_OPEN.strftime('%H:%M')}", tz=ET), pd.Timestamp(f"{d} {close}", tz=ET))


def next_trading_day(d):
    d += timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def trading_days_between(start, end):
    """Trading days after `start` up to and including `end` (0 on the entry day itself)."""
    n, d = 0, start
    while d < end:
        d += timedelta(days=1)
        n += is_trading_day(d)
    return n


def market_open(now):
    if not is_trading_day(now.date()):
        return False
    o, c = session(now.date())
    return o <= now < c


def tradable_time(accepted):
    """When the backtest would enter: 1 minute after acceptance, or the next regular open."""
    t = accepted + timedelta(minutes=ENTRY_DELAY_MINUTES)
    if is_trading_day(t.date()):
        o, c = session(t.date())
        if t < o:
            return o
        if t < c:
            return t
    return session(next_trading_day(t.date()))[0]


# ---------- Prices (Massive, same source as the backtest) ----------

def last_price(ticker):
    now = pd.Timestamp.now(tz=ET)
    bars = fetch_minute_bars(ticker, now - pd.Timedelta(days=5), now, os.environ["MASSIVE_API_KEY"], adjusted=True)
    if bars.empty:
        raise RuntimeError(f"no recent minute bars for {ticker}")
    return float(bars["close"].iloc[-1])


def daily_bars(ticker):
    """Split-adjusted daily bars for the last ~2 months, refreshed once a day."""
    os.makedirs(PRICE_DIR, exist_ok=True)
    path = os.path.join(PRICE_DIR, f"{ticker}.csv")
    if os.path.exists(path) and date.fromtimestamp(os.path.getmtime(path)) < date.today():
        os.remove(path)
    bl.PRICE_DIR, bl.START, bl.END = PRICE_DIR, (date.today() - timedelta(days=70)).isoformat(), date.today().isoformat()
    return bl.daily_bars(ticker)


def market_features(px, hedge, accepted, entry_day):
    """The market features of build_labels.py / backtest.py, from the 20 trading days before entry_day."""
    if px.empty:
        return None
    hist = px[px.index < entry_day].iloc[-20:]
    if len(hist) < 20:
        return None
    h_hist = hedge.loc[hist.index.intersection(hedge.index)]
    if len(h_hist) < 6:
        return None
    price = hist["close"].iloc[-1]
    dollar_vol = (hist["close"] * hist["volume"]).mean()
    if price < bl.MIN_PRICE or dollar_vol < bl.MIN_DOLLAR_VOLUME:
        return None
    t = accepted.tz_convert(ET).time()
    return {
        "pre_ret_5": hist["close"].iloc[-1] / hist["close"].iloc[-6] - 1 - (h_hist["close"].iloc[-1] / h_hist["close"].iloc[-6] - 1),
        "pre_ret_20": hist["close"].iloc[-1] / hist["close"].iloc[0] - 1 - (h_hist["close"].iloc[-1] / h_hist["close"].iloc[0] - 1),
        "vol_20": hist["close"].pct_change().dropna().std(),
        "log_dollar_volume": np.log10(dollar_vol),
        "log_price": np.log10(price),
        "after_hours": int(t >= MARKET_CLOSE or t < MARKET_OPEN),
    }


# ---------- Brokers ----------

class WebullBroker:
    """Market orders on Webull's paper trading account through the official OpenAPI SDK."""
    name = "Webull paper"

    def __init__(self):
        from webull.core.client import ApiClient
        from webull.trade.trade_client import TradeClient
        key, secret = os.environ.get("WEBULL_APP_KEY"), os.environ.get("WEBULL_APP_SECRET")
        if not key or not secret:
            sys.exit("Set WEBULL_APP_KEY and WEBULL_APP_SECRET (your paper trading keys) first, or use --dry-run.")
        client = ApiClient(key, secret, WEBULL_REGION)
        client.add_endpoint(WEBULL_REGION, WEBULL_ENDPOINT)
        self.trade = TradeClient(client)
        self.account_id = os.environ.get("WEBULL_ACCOUNT_ID") or self.accounts()[0]["account_id"]

    def accounts(self):
        data = self.trade.account_v2.get_account_list().json()
        accounts = data if isinstance(data, list) else data.get("data") or data.get("accounts") or [data]
        if not accounts:
            sys.exit("No Webull accounts for these keys. Make sure they are the paper trading (sandbox) keys.")
        return accounts

    def balance(self):
        return self.trade.account_v2.get_account_balance(self.account_id).json()

    def equity(self):
        return float(self.balance()["total_net_liquidation_value"])

    def positions(self):
        return self.trade.account_v2.get_account_position(self.account_id).json()

    @staticmethod
    def order(symbol, side, qty, cid=None):
        return {"combo_type": "NORMAL", "client_order_id": cid or uuid.uuid4().hex, "symbol": symbol,
                "instrument_type": "EQUITY", "market": "US", "order_type": "MARKET", "quantity": str(qty),
                "support_trading_session": "CORE", "side": side, "time_in_force": "DAY", "entrust_type": "QTY"}

    def preview(self, symbol, side, qty):
        return self.trade.order_v3.preview_order(self.account_id, [self.order(symbol, side, qty)]).json()

    def detail(self, cid):
        data = self.trade.order_v3.get_order_detail(self.account_id, cid).json()
        orders = data.get("orders") if isinstance(data, dict) else data
        return orders[0] if orders else {}

    def place(self, symbol, side, qty):
        """Sends a market order and waits for the fill. Returns (filled quantity, average price, order id)."""
        cid = uuid.uuid4().hex
        try:
            self.trade.order_v3.place_order(self.account_id, [self.order(symbol, side, qty, cid)])
        except Exception as e:
            raise OrderError(f"{side} {qty} {symbol} rejected: {e}")
        deadline, o = time.time() + ORDER_TIMEOUT_SECONDS, {}
        while True:
            time.sleep(2)  # order detail allows 2 requests per 2 seconds
            try:
                o = self.detail(cid)
            except Exception as e:
                log(f"  order status check failed ({e}); trying again")
            if o.get("status") == "FILLED":
                break
            if o.get("status") in ("CANCELLED", "FAILED"):
                raise OrderError(f"{side} {qty} {symbol}: order {o.get('status')}")
            if time.time() > deadline:
                try:
                    self.trade.order_v3.cancel_order(self.account_id, cid)
                    time.sleep(2)
                    o = self.detail(cid)
                except Exception:
                    pass
                break
        filled = float(o.get("filled_quantity") or 0)
        if filled <= 0:
            raise OrderError(f"{side} {qty} {symbol}: not filled within {ORDER_TIMEOUT_SECONDS}s (status {o.get('status')})")
        try:
            price = float(o.get("filled_price") or 0) or last_price(symbol)
        except Exception as e:
            raise OrderError(f"{side} {qty} {symbol} filled {filled:g}, but no fill price: {e}")
        return int(round(filled)), price, cid


class DryRunBroker:
    """Fills every order at the latest minute bar's close and sends nothing."""
    name = "dry run (no orders sent)"

    def __init__(self, state):
        self.state = state

    def equity(self):
        return DRY_RUN_CAPITAL + self.state["realized_pnl"]

    def place(self, symbol, side, qty):
        price = last_price(symbol)
        log(f"  [dry run] {side} {qty} {symbol} @ {price:.2f}")
        return qty, price, f"dry-{uuid.uuid4().hex[:12]}"


# ---------- State ----------

def load_state():
    state = {"seen": [], "pending": [], "positions": [], "xph_qty": 0, "realized_pnl": 0.0, "last_exit_check": ""}
    if os.path.exists(STATE_JSON):
        state.update(json.load(open(STATE_JSON)))
    return state


def save_state(state):
    state["seen"] = state["seen"][-5000:]
    with open(STATE_JSON + ".tmp", "w") as fh:
        json.dump(state, fh, indent=2, default=str)
    os.replace(STATE_JSON + ".tmp", STATE_JSON)


def append_csv(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=not os.path.exists(path))


def record_signal(f, action, **extra):
    append_csv(SIGNALS_CSV, {"time": datetime.now(ET).isoformat(timespec="seconds"), "Accession": f.get("Accession"),
                             "Ticker": f.get("Ticker"), "Company": f.get("Company"), "AcceptedET": f.get("AcceptedET"),
                             "action": action, **{k: extra.get(k) for k in ("p_up", "side", "rule", "event_type")}})


# ---------- 1. New filings from EDGAR ----------

def feed_entries(seen):
    """Original 8-Ks in EDGAR's live feed that we haven't looked at yet, oldest first."""
    found = {}
    for page in range(FEED_PAGES):
        xml = sec.sec_get(FEED_URL.format(start=100 * page))
        entries = xml.split("<entry>")[1:]
        for e in entries:
            title = re.search(r"<title>(.*?)</title>", e, re.S)
            acc = re.search(r"accession-number=([\d-]+)", e)
            updated = re.search(r"<updated>(.*?)</updated>", e)
            m = re.match(r"\s*(\S+) - (.*) \((\d+)\) \((\w+)\)", title.group(1)) if title else None
            if not (m and acc and updated) or m.group(1) != "8-K":
                continue  # 8-K/A amendments are left out, as in training
            found.setdefault(acc.group(1), {"Accession": acc.group(1), "CIK": int(m.group(3)),
                                            "Company": html.unescape(m.group(2)), "Updated": updated.group(1)})
        if len(entries) < 100 or any(acc in seen for acc in found):
            break
    return sorted((f for a, f in found.items() if a not in seen), key=lambda f: f["Updated"])


def company_filings(cik):
    """SIC code and recent filings (accession, form, item codes, acceptance time) from EDGAR's submissions API."""
    data = json.loads(sec.sec_get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"))
    r = data["filings"]["recent"]
    df = pd.DataFrame({k: r.get(k, [""] * len(r["accessionNumber"]))
                       for k in ("accessionNumber", "form", "items", "acceptanceDateTime")})
    return str(data.get("sic", "")), df


def item_codes(items_text):
    text = str(items_text).lower()
    return {code for code, name in ITEM_NAMES.items() if name in text}


def material(codes):
    """Kept if any item is market-moving, or if the items couldn't be read (as in sec_edgar_8k.py)."""
    return not codes or bool(codes & sec.MATERIAL_ITEMS)


def days_since_last_filing(history, f, accepted):
    """Days since the company's previous original, material 8-K (capped at 365, as in training)."""
    h = history[(history["form"] == "8-K") & (history["accessionNumber"] != f["Accession"])].copy()
    h = h[h["items"].fillna("").map(lambda s: material({c.strip() for c in s.split(",") if c.strip()}))]
    ts = pd.to_datetime(h["acceptanceDateTime"], utc=True, errors="coerce")
    ts = ts[ts < accepted]
    return float(min(365, max(0, (accepted - ts.max()).days))) if len(ts) else 365.0


def screen_new_filings(state, ticker_map, sic_cache, http):
    """Reads the feed and turns each new pharma 8-K into a scored filing waiting for its entry time."""
    new = feed_entries(set(state["seen"]))
    if not new:
        return
    now = pd.Timestamp.now(tz=ET)
    unknown = sorted({f["CIK"] for f in new if f["CIK"] not in sic_cache and f["CIK"] in ticker_map})
    if unknown:
        sic_cache.update(sec.load_sic_codes(unknown))
    stop_markers = [f"<TYPE>{t}" for t in sec.STOP_AT_TYPES]

    for f in new:
        state["seen"].append(f["Accession"])
        f["Ticker"] = ticker_map.get(f["CIK"], "")
        if not f["Ticker"] or sic_cache.get(f["CIK"]) not in sec.PHARMA_SIC:
            continue
        # Cheap lateness check from the feed time before downloading and scoring anything
        if now - tradable_time(pd.Timestamp(f["Updated"]).tz_convert(ET)) > pd.Timedelta(minutes=MAX_LATE_MINUTES + 5):
            continue
        try:
            accepted, items, text = sec.parse_filing(sec.sec_get(
                f"https://www.sec.gov/Archives/edgar/data/{f['CIK']}/{f['Accession']}.txt", stop_markers))
            _, history = company_filings(f["CIK"])
        except Exception as e:
            log(f"{f['Ticker']} {f['Accession']}: couldn't download ({e}); skipped")
            continue
        accepted = pd.Timestamp(accepted) if accepted else pd.Timestamp(f["Updated"]).tz_convert(ET)
        listed = history.loc[history["accessionNumber"] == f["Accession"], "items"]
        codes = ({c.strip() for c in listed.iloc[0].split(",") if c.strip()} if len(listed) and listed.iloc[0]
                 else item_codes(items))
        f.update({"AcceptedET": accepted.isoformat(), "Items": items, "Text": text,
                  "TradableAt": tradable_time(accepted).isoformat(),
                  "days_since_last_filing": days_since_last_filing(history, f, accepted)})
        if not material(codes):
            continue
        if len(text) <= 200:  # training only scored filings with real text
            record_signal(f, "skip: no text")
            continue
        if now - pd.Timestamp(f["TradableAt"]) > pd.Timedelta(minutes=MAX_LATE_MINUTES):
            record_signal(f, "skip: seen too late")
            continue

        answers, _, _ = ol.score_batch([{k: f[k] for k in ("Accession", "Company", "Ticker", "Items", "Text")}], http)
        if not answers:
            log(f"{f['Ticker']} {f['Accession']}: LLM scoring failed; skipped")
            record_signal(f, "skip: LLM scoring failed")
            continue
        f["scores"] = {k: v for k, v in answers[0].items() if k != "Accession"}
        del f["Text"]
        state["pending"].append(f)
        log(f"New filing {f['Ticker']} ({f['Company']}) accepted {accepted:%H:%M:%S}, "
            f"{f['scores']['event_type']}, sentiment {f['scores']['sentiment']:+.2f}; trades at {f['TradableAt']}")
    save_state(state)


# ---------- 2. Signals and entries ----------

def load_models():
    missing = [f for f in MODEL_FILES if not os.path.exists(f)]
    if missing:
        sys.exit(f"Missing {', '.join(missing)}. Train first (python run_all.py) and run this from the repo root.")
    return json.load(open("model_config.json")), joblib.load("direction_model.joblib"), joblib.load("exit_model.joblib")


def make_signal(f, models):
    """Direction, probability and exit rule for one filing, or None with the reason it's untradable."""
    cfg, direction, exit_model = models
    accepted = pd.Timestamp(f["AcceptedET"])
    entry_day = pd.Timestamp(pd.Timestamp(f["TradableAt"]).date())
    feats = market_features(daily_bars(f["Ticker"]), daily_bars(bl.HEDGE), accepted, entry_day)
    if feats is None:
        return None, "untradable (price under $5, too little volume or too little history)"
    df = pd.DataFrame([{**feats, **f["scores"], "days_since_last_filing": f["days_since_last_filing"]}])
    df = add_text_features(df)
    # Same Items text as training (EDGAR's header item names), so the item features match what the model saw
    df = add_item_features(df, pd.Series([f["Items"]]))
    X = df[cfg["features"]]
    p_up = float(direction.predict_proba(X)[0, 1])
    side = 1 if p_up >= cfg["cutoff"] else -1 if p_up <= 1 - cfg["cutoff"] else 0
    if side == 0:
        return {"p_up": p_up, "side": 0}, f"no trade (p_up {p_up:.3f} inside the {1 - cfg['cutoff']:.2f}-{cfg['cutoff']:.2f} band)"
    Xe = X.copy()
    Xe["side"] = side
    rule = cfg["rules"][int(exit_model.predict(Xe)[0])]
    tp, sl, max_days = cfg["exit_rules"][rule]
    if tp == "vol":
        tp, sl = 2 * feats["vol_20"] * max_days ** 0.5, feats["vol_20"] * max_days ** 0.5
    return {"p_up": p_up, "side": side, "rule": rule, "tp": tp, "sl": sl, "max_days": max_days,
            "vol_20": feats["vol_20"]}, None


def rebalance_hedge(broker, state):
    """Trades XPH so the account holds the sum of every open trade's hedge. Longs and shorts net out,
    so one XPH position covers them all. Returns the last fill price, or None if nothing traded."""
    delta = sum(p["hedge_qty"] for p in state["positions"]) - state["xph_qty"]
    current = state["xph_qty"]
    if delta > 0:
        cover = min(delta, max(-current, 0))
        legs = [("BUY", cover), ("BUY", delta - cover)]       # cover a short first, then go long
    else:
        sell = min(-delta, max(current, 0))
        legs = [("SELL", sell), ("SHORT", -delta - sell)]     # sell a long first, then go short
    price = None
    for side, qty in legs:
        if qty > 0:
            filled, price, _ = broker.place(bl.HEDGE, side, qty)
            state["xph_qty"] += filled if side == "BUY" else -filled
            save_state(state)
    return price


def enter(f, sig, broker, state):
    """Sizes and opens one trade (stock leg, then the XPH hedge). Returns why it was skipped, or None."""
    equity = broker.equity()
    open_pos = state["positions"]
    if len(open_pos) >= MAX_POSITIONS:
        return f"{MAX_POSITIONS} trades already open"
    if any(p["Ticker"] == f["Ticker"] for p in open_pos):
        return "already holding this ticker"
    room = MAX_GROSS_PCT * equity - sum(p["notional"] for p in open_pos)
    notional = min(POSITION_PCT * equity, room)
    if notional < MIN_POSITION_PCT * equity:
        return "fully invested"
    qty = int(notional // last_price(f["Ticker"]))
    if qty < 1:
        return "position too small for one share"

    side = sig["side"]
    try:
        filled, price, order_id = broker.place(f["Ticker"], "BUY" if side == 1 else "SHORT", qty)
    except OrderError as e:
        return f"order failed: {e}"
    xph_price = last_price(bl.HEDGE)
    pos = {"Accession": f["Accession"], "Ticker": f["Ticker"], "Company": f["Company"], "AcceptedET": f["AcceptedET"],
           "event_type": f["scores"]["event_type"], **sig, "qty": filled, "entry_price": price,
           "entry_time": datetime.now(ET).isoformat(timespec="seconds"), "entry_day": date.today().isoformat(),
           "notional": filled * price, "hedge_qty": -side * round(filled * price / xph_price), "order_id": order_id}
    open_pos.append(pos)
    save_state(state)
    try:
        pos["xph_entry"] = rebalance_hedge(broker, state) or xph_price
    except OrderError as e:
        # Never leave an unhedged trade: undo the stock leg
        log(f"  XPH hedge failed ({e}); closing the {f['Ticker']} leg again")
        open_pos.remove(pos)
        broker.place(f["Ticker"], "SELL" if side == 1 else "BUY", filled)
        save_state(state)
        return f"hedge failed: {e}"
    save_state(state)
    log(f"ENTERED {'LONG' if side == 1 else 'SHORT'} {filled} {f['Ticker']} @ {price:.2f} (${pos['notional']:,.0f}, "
        f"p_up {sig['p_up']:.3f}, exit rule {sig['rule']}), XPH hedge {pos['hedge_qty']:+d} @ {pos['xph_entry']:.2f}")
    return None


def process_pending(state, broker, models):
    now = pd.Timestamp.now(tz=ET)
    for f in list(state["pending"]):
        tradable = pd.Timestamp(f["TradableAt"])
        if tradable > now:
            continue
        if not market_open(now):
            if now - tradable > pd.Timedelta(minutes=MAX_LATE_MINUTES):
                state["pending"].remove(f)
                record_signal(f, "skip: market closed before it could be traded")
            continue
        state["pending"].remove(f)
        if now - tradable > pd.Timedelta(minutes=MAX_LATE_MINUTES):
            record_signal(f, "skip: seen too late")
            continue
        try:
            sig, why = make_signal(f, models)
            if why is None:
                why = enter(f, sig, broker, state)
        except Exception as e:
            sig, why = None, f"error: {e}"
        action = "entered" if why is None else f"skip: {why}"
        if why is not None:
            log(f"{f['Ticker']}: {why}")
        record_signal(f, action, event_type=f["scores"]["event_type"], **(sig or {}))
        save_state(state)


# ---------- 3. Exits ----------

def hedged_return(p, price, xph_price):
    return p["side"] * ((price / p["entry_price"] - 1) - (xph_price / p["xph_entry"] - 1))


def check_exits(state, broker):
    """The backtest's "close" stop check: take-profit, stop or time limit on the hedged return at the close."""
    today = date.today()
    if not state["positions"]:
        state["last_exit_check"] = today.isoformat()
        save_state(state)
        return
    xph_now = last_price(bl.HEDGE)
    closing = []
    for p in state["positions"]:
        try:
            ret = hedged_return(p, last_price(p["Ticker"]), xph_now)
        except Exception as e:
            log(f"{p['Ticker']}: no price for the exit check ({e}); kept open")
            continue
        day = trading_days_between(date.fromisoformat(p["entry_day"]), today)
        reason = ("take_profit" if p["tp"] is not None and ret >= p["tp"] else
                  "stop" if p["sl"] is not None and ret <= -p["sl"] else
                  "time" if day >= p["max_days"] else None)
        target = "none" if p["tp"] is None else f"{p['tp']:+.2%}"
        stop = "none" if p["sl"] is None else f"{-p['sl']:+.2%}"
        log(f"{p['Ticker']} {'long' if p['side'] == 1 else 'short'} day {day}/{p['max_days']}: hedged {ret:+.2%} "
            f"(target {target}, stop {stop})" + (f" -> closing ({reason})" if reason else ""))
        if reason:
            closing.append((p, reason))

    exits, failed = [], False
    for p, reason in closing:
        try:
            _, price, _ = broker.place(p["Ticker"], "SELL" if p["side"] == 1 else "BUY", p["qty"])
        except OrderError as e:
            log(f"  couldn't close {p['Ticker']}: {e}; trying again next pass")
            failed = True
            continue
        state["positions"].remove(p)
        save_state(state)
        exits.append((p, reason, price))
    if not failed:
        state["last_exit_check"] = today.isoformat()  # otherwise the check runs again next pass, until the close
    if not exits:
        save_state(state)
        return

    try:
        xph_exit = rebalance_hedge(broker, state) or xph_now
    except OrderError as e:
        log(f"  XPH hedge adjustment failed ({e}); will retry at the next pass")
        xph_exit = xph_now
    for p, reason, price in exits:
        gross = hedged_return(p, price, xph_exit)
        net = gross - 2 * bl.COST
        pnl = p["notional"] * gross
        state["realized_pnl"] += p["notional"] * net
        append_csv(TRADES_CSV, {
            "entry_time": p["entry_time"], "exit_time": datetime.now(ET).isoformat(timespec="seconds"),
            "Ticker": p["Ticker"], "direction": "long" if p["side"] == 1 else "short", "p_up": p["p_up"],
            "event_type": p["event_type"], "rule": p["rule"], "qty": p["qty"], "entry_price": p["entry_price"],
            "exit_price": price, "xph_entry": p["xph_entry"], "xph_exit": xph_exit, "gross_return": gross,
            "net_return": net, "notional": p["notional"], "pnl": pnl, "exit_reason": reason,
            "Accession": p["Accession"], "broker": broker.name})
        log(f"CLOSED {p['Ticker']} ({reason}): hedged {gross:+.2%}, about ${pnl:+,.0f}")
    save_state(state)


def exit_check_due(state, now):
    if not is_trading_day(now.date()) or state["last_exit_check"] == now.date().isoformat():
        return False
    _, close = session(now.date())
    return close - pd.Timedelta(minutes=EXIT_CHECK_MINUTES) <= now < close


# ---------- Commands ----------

def require_env(dry_run):
    for key in ("MASSIVE_API_KEY", "OSS_API_KEY", "SEC_USER_AGENT"):
        if not os.environ.get(key):
            sys.exit(f"Set {key} first (see .env.example).")
    if not dry_run and not (os.environ.get("WEBULL_APP_KEY") and os.environ.get("WEBULL_APP_SECRET")):
        sys.exit("Set WEBULL_APP_KEY and WEBULL_APP_SECRET first, or use --dry-run.")


class Trader:
    def __init__(self, dry_run):
        require_env(dry_run)
        self.models = load_models()
        self.state = load_state()
        self.broker = DryRunBroker(self.state) if dry_run else WebullBroker()
        self.http = requests.Session()
        self.sic_cache, self.ticker_map, self.ticker_day = {}, {}, None
        log(f"Paper trader started: {self.broker.name}, equity ${self.broker.equity():,.0f}, "
            f"{len(self.state['positions'])} open trades, {len(self.state['pending'])} filings waiting")

    def step(self):
        if self.ticker_day != date.today():  # the SEC's ticker list changes; refresh daily
            self.ticker_map, self.ticker_day = sec.load_ticker_map(), date.today()
        screen_new_filings(self.state, self.ticker_map, self.sic_cache, self.http)
        process_pending(self.state, self.broker, self.models)
        now = pd.Timestamp.now(tz=ET)
        if market_open(now) and sum(p["hedge_qty"] for p in self.state["positions"]) != self.state["xph_qty"]:
            log("XPH hedge doesn't match the open trades (an earlier hedge order failed); fixing")
            rebalance_hedge(self.broker, self.state)
        if exit_check_due(self.state, now):
            log("Exit check")
            check_exits(self.state, self.broker)


def cmd_run(args):
    trader = Trader(args.dry_run)
    while True:
        try:
            trader.step()
        except KeyboardInterrupt:
            raise
        except Exception as e:  # a network hiccup shouldn't stop the bot
            log(f"error: {e!r}; retrying next pass")
        time.sleep(POLL_SECONDS)


def cmd_once(args):
    Trader(args.dry_run).step()


def cmd_check(args):
    broker = WebullBroker()
    print(f"Endpoint: {WEBULL_ENDPOINT}")
    print("Accounts:", json.dumps(broker.accounts(), indent=2))
    print(f"Using account {broker.account_id}")
    b = broker.balance()
    print(f"Net liquidation value: {b.get('total_net_liquidation_value')} {b.get('total_asset_currency', '')}")
    print("Positions:", json.dumps(broker.positions(), indent=2))
    # Preview checks an order without placing it: shows whether long and short market orders are allowed
    for side in ("BUY", "SHORT"):
        try:
            print(f"Preview {side} 1 {bl.HEDGE}:", json.dumps(broker.preview(bl.HEDGE, side, 1)))
        except Exception as e:
            print(f"Preview {side} 1 {bl.HEDGE} refused: {e}")
    if not all(os.path.exists(f) for f in MODEL_FILES):
        print(f"\nNote: {', '.join(MODEL_FILES)} not found here; run python run_all.py before trading.")


def cmd_status(args):
    state = load_state()
    print(f"{len(state['pending'])} filings waiting for their entry time; XPH hedge held: {state['xph_qty']:+d} shares")
    if not state["positions"]:
        print("No open trades.")
        return
    xph_now = last_price(bl.HEDGE)
    today = date.today()
    for p in state["positions"]:
        ret = hedged_return(p, last_price(p["Ticker"]), xph_now)
        day = trading_days_between(date.fromisoformat(p["entry_day"]), today)
        print(f"{p['Ticker']:<6} {'long ' if p['side'] == 1 else 'short'} {p['qty']:>6} @ {p['entry_price']:<9.2f} "
              f"since {p['entry_day']} (day {day}/{p['max_days']}, rule {p['rule']}): hedged {ret:+.2%}")
    if os.path.exists(TRADES_CSV):
        tr = pd.read_csv(TRADES_CSV)
        print(f"\nClosed trades: {len(tr)}, win rate {(tr['net_return'] > 0).mean():.0%}, "
              f"avg net {tr['net_return'].mean():+.2%}, P&L about ${tr['pnl'].sum():,.0f}")


def main():
    parser = argparse.ArgumentParser(description="Paper trade the pharma 8-K strategy on Webull.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("run", cmd_run), ("once", cmd_once)):
        p = sub.add_parser(name)
        p.add_argument("--dry-run", action="store_true", help="log orders instead of sending them to Webull")
        p.set_defaults(fn=fn)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
