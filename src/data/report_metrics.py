import json
import os
from datetime import date
from statistics import NormalDist

import numpy as np
import pandas as pd

import build_labels as bl
import train_models as tm

# Numbers the hackathon report asks for that the training and backtest scripts don't print.
# Run after train_models.py and backtest.py, from the repo root:
#   python src/data/report_metrics.py
#
# In-sample (IS)  = walk-forward 2020-2024: each year predicted by a model trained only on earlier years.
# Out-of-sample (OOS) = the 2025-02-15 onward backtest (backtest_trades.csv, backtest_equity.csv).
# Prints IS and OOS side by side, the same with costs doubled, turnover, capacity (% of dollar ADV),
# a factor regression on SPY and XPH, and the Deflated Sharpe Ratio. Saves report_metrics.json.

TRADES_CSV, EQUITY_CSV = "backtest_trades.csv", "backtest_equity.csv"
OOS_FEATURES_CSV = "test_features.csv"     # has log_dollar_volume for each 2025+ filing
TUNING_CSV = "tuning_results.csv"
OUT_JSON = "report_metrics.json"
EXTRA_VARIANTS = 10      # versions tried before the tuning grid (old models, stop and sizing variants); edit to your count
POSITION_PCT, START_CAPITAL = 0.05, 100_000
FACTOR_DIR = "prices_factors"
ADV_LIMITS = (0.01, 0.05)   # trade size as a share of the stock's average daily dollar volume


def stats_from_daily(pnl_pct, turnover):
    """Annual return, volatility, Sharpe and max drawdown from daily returns."""
    eq = (1 + pnl_pct).cumprod()
    years = len(pnl_pct) / 252
    vol = pnl_pct.std() * np.sqrt(252)
    return {"annual_return": eq.iloc[-1] ** (1 / years) - 1, "annual_vol": vol,
            "sharpe": pnl_pct.mean() * 252 / vol if vol > 0 else np.nan,
            "max_drawdown": (eq / eq.cummax() - 1).min(), "turnover_per_year": turnover}


# ---------- In-sample: walk-forward 2020-2024 with the chosen setting ----------

def in_sample(cfg, extra_cost):
    df = tm.load()
    rows = []
    for (valid, p, rl, rs) in tm.out_of_fold(df, cfg["settings"]):
        rets, sides = tm.trade_returns(valid, p, cfg["cutoff"], rl, rs)
        taken = (p >= cfg["cutoff"]) | (p <= 1 - cfg["cutoff"])
        rows.append(pd.DataFrame({"day": valid["EntryDay"].to_numpy()[taken], "ret": rets - extra_cost,
                                  "adv": 10 ** valid["log_dollar_volume"].to_numpy()[taken]}))
    tr = pd.concat(rows)
    # Approximate daily P&L: each trade is 5% of $100k, booked on its entry day (labels don't store exit days)
    days = pd.bdate_range(tr["day"].min(), tr["day"].max())
    pnl = (tr.groupby("day")["ret"].sum() * POSITION_PCT).reindex(days, fill_value=0.0)
    years = len(days) / 252
    out = stats_from_daily(pnl, turnover=len(tr) * 2 * POSITION_PCT / years)
    out.update({"trades": len(tr), "avg_trade": tr["ret"].mean(), "win_rate": (tr["ret"] > 0).mean(),
                "by_year": tr.groupby(tr["day"].dt.year)["ret"].mean().round(4).to_dict()})
    return out, tr


# ---------- Out-of-sample: the 2025+ backtest ----------

def out_of_sample(extra_cost):
    tr = pd.read_csv(TRADES_CSV)
    eq = pd.read_csv(EQUITY_CSV, index_col=0, parse_dates=True)["equity"]
    # Doubling costs: take the extra cost off each trade's P&L on its exit day (entry day if still open)
    when = pd.to_datetime(tr["exit_time"].where(tr["exit_reason"] != "open", tr["entry_time"]).str[:10])
    extra = (tr["notional"] * extra_cost).groupby(when).sum().reindex(eq.index, fill_value=0).cumsum()
    eq = eq - extra
    years = len(eq) / 252
    turnover = 2 * tr["notional"].sum() / eq.mean() / years
    out = stats_from_daily(eq.pct_change().dropna(), turnover)
    net = tr["net_return"] - extra_cost
    out.update({"trades": len(tr), "avg_trade": net.mean(), "win_rate": (net > 0).mean(),
                "total_return": eq.iloc[-1] / START_CAPITAL - 1})
    return out, tr, eq


# ---------- Capacity ----------

def capacity(tr_oos):
    feats = pd.read_csv(OOS_FEATURES_CSV, usecols=["Accession", "log_dollar_volume"]).drop_duplicates("Accession")
    adv = tr_oos["Accession"].map(feats.set_index("Accession")["log_dollar_volume"]).pipe(lambda s: 10 ** s)
    share = (tr_oos["notional"] / adv).dropna()
    capital = START_CAPITAL
    out = {"median_pct_adv": share.median(), "p90_pct_adv": share.quantile(0.9),
           "median_adv_dollars": adv.median()}
    # Trade sizes grow in line with capital, so the share of ADV does too
    out["capital_median_at_1pct_adv"] = capital * ADV_LIMITS[0] / share.median()
    out["capital_p90_at_5pct_adv"] = capital * ADV_LIMITS[1] / share.quantile(0.9)
    return out


# ---------- Factor regression ----------

def factor_regression(eq):
    bl.PRICE_DIR, bl.START, bl.END = FACTOR_DIR, eq.index[0].date().isoformat(), date.today().isoformat()
    f = pd.DataFrame({t: bl.daily_bars(t)["close"] for t in ("SPY", "XPH")}).pct_change()
    f.index = pd.to_datetime(f.index).tz_localize(None).normalize()
    y = eq.pct_change().rename("strategy")
    d = pd.concat([y, f], axis=1, join="inner").dropna()
    X = np.column_stack([np.ones(len(d)), d[["SPY", "XPH"]].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, d["strategy"].to_numpy(), rcond=None)
    resid = d["strategy"].to_numpy() - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var(ddof=3))
    r2 = 1 - resid.var() / d["strategy"].var(ddof=0)
    return {"alpha_annual": beta[0] * 252, "alpha_t": beta[0] / se[0], "beta_SPY": beta[1], "t_SPY": beta[1] / se[1],
            "beta_XPH": beta[2], "t_XPH": beta[2] / se[2], "r2": r2, "days": len(d)}


# ---------- Deflated Sharpe Ratio (Bailey and Lopez de Prado, 2014) ----------

def deflated_sharpe(is_trades):
    """Per-trade Sharpe of the chosen IS strategy, deflated for how many variants were tried."""
    res = pd.read_csv(TUNING_CSV)
    res = res[res["trades"] >= 30]
    trial_sr = res["score"] / np.sqrt(res["trades"])   # score = mean/std*sqrt(n), so this is mean/std per trade
    n_trials = len(res) + EXTRA_VARIANTS
    r = is_trades["ret"].to_numpy()
    sr, T = r.mean() / r.std(ddof=1), len(r)
    skew = pd.Series(r).skew()
    kurt = pd.Series(r).kurt() + 3
    z, g = NormalDist().inv_cdf, 0.5772156649
    sr0 = np.sqrt(trial_sr.var()) * ((1 - g) * z(1 - 1 / n_trials) + g * z(1 - 1 / (n_trials * np.e)))
    dsr = NormalDist().cdf((sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return {"variants_tried": n_trials, "sharpe_per_trade": sr, "expected_max_sharpe_from_luck": sr0,
            "deflated_sharpe_prob": dsr, "trades": T}


def show(title, rows):
    print(f"\n{title}\n" + pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.4f}"))


def main():
    cfg = json.load(open("model_config.json"))
    print(f"Chosen setting {cfg['settings']}, cutoff {cfg['cutoff']}; costs {bl.COST * 1e4:.0f} bps per side")
    result = {}
    for label, extra in (("costs", 0.0), ("costs_doubled", 2 * bl.COST)):
        is_stats, is_trades = in_sample(cfg, extra)
        oos_stats, oos_trades, eq = out_of_sample(extra)
        result[label] = {"IS_2020_2024": is_stats, "OOS_2025_on": oos_stats}
        show(f"{'Base costs' if extra == 0 else 'Costs doubled'} (IS daily P&L is approximate: 5% per trade, booked at entry)",
             {"IS 2020-2024": {k: v for k, v in is_stats.items() if k != "by_year"}, "OOS 2025+": oos_stats})
        if extra == 0:
            base_is, base_oos_trades, base_eq = is_trades, oos_trades, eq
            print("IS avg trade by year:", is_stats["by_year"])

    result["capacity"] = capacity(base_oos_trades)
    show("Capacity (OOS trades)", {"value": result["capacity"]})
    try:
        result["factors"] = factor_regression(base_eq)
        show("Factor regression of OOS daily returns on SPY and XPH", {"value": result["factors"]})
    except Exception as e:
        print(f"\nFactor regression skipped: {e}")
    result["deflated_sharpe"] = deflated_sharpe(base_is)
    show("Deflated Sharpe Ratio (IS walk-forward, per trade)", {"value": result["deflated_sharpe"]})

    json.dump(result, open(OUT_JSON, "w"), indent=2, default=float)
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
