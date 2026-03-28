"""
execution/paper_logger.py
Nova — paper trading logger

Runs every morning after 10:00 AM ET to generate signals,
then every evening after 4:00 PM ET to record outcomes.

All trades stored in CSV — no broker connection needed.
This is pure forward-testing: real signals, no real money.

Usage:
    python -m execution.paper_logger --mode morning   # log signals
    python -m execution.paper_logger --mode evening   # record outcomes
    python -m execution.paper_logger --mode report    # print summary
"""

import argparse
import csv
import os
from datetime import datetime, date
from pathlib import Path
from loguru import logger
import pandas as pd
import yfinance as yf

from strategy.bias    import get_daily_bias
from strategy.scanner import scan_stocks_in_play, NASDAQ_WATCHLIST
from strategy.orb     import detect_breakout
from risk.manager     import AccountState, approve_signal
from data.fetcher     import fetch_daily
from utils.indicators import compute_atr

# ── Config ─────────────────────────────────────────────────
PAPER_LOG     = "reports/paper_trades.csv"
EQUITY_START  = 10_000.0
VIX_MAX       = 25.0

# Pruned watchlist — backtested negative tickers removed
NOVA_WATCHLIST = [
    "NFLX", "MSFT", "SNOW", "CRWD", "META",
    "NVDA", "AMD",  "AAPL", "MU",   "TSLA", "QCOM",
]

# CSV columns
COLUMNS = [
    "date", "ticker", "direction", "bias",
    "entry", "stop", "target",
    "rvol", "gap_pct", "atr",
    "shares", "dollar_risk",
    "exit_price", "exit_reason",
    "pnl_r", "pnl_dollar",
    "status",            # OPEN | CLOSED
    "logged_at",
    "closed_at",
]


# ── Helpers ────────────────────────────────────────────────
def load_log() -> pd.DataFrame:
    """Load existing paper trade log or create empty one."""
    Path(PAPER_LOG).parent.mkdir(exist_ok=True)

    if not Path(PAPER_LOG).exists():
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(PAPER_LOG, index=False)
        logger.info(f"Created new paper log: {PAPER_LOG}")
        return df

    return pd.read_csv(PAPER_LOG)


def save_log(df: pd.DataFrame):
    """Save paper trade log to CSV."""
    df.to_csv(PAPER_LOG, index=False)


def get_vix() -> float:
    """Fetch current VIX level."""
    try:
        vix = yf.download("^VIX", period="2d", interval="1d",
                          auto_adjust=True, progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix = vix.droplevel(level=1, axis=1)
        return float(vix["Close"].iloc[-1])
    except Exception:
        return 0.0


def get_equity(df: pd.DataFrame) -> float:
    """Compute current account equity from closed trades."""
    closed = df[df["status"] == "CLOSED"]
    if closed.empty:
        return EQUITY_START
    total_pnl = closed["pnl_dollar"].astype(float).sum()
    return round(EQUITY_START + total_pnl, 2)


# ── Morning session ────────────────────────────────────────
def run_morning():
    """
    Morning routine — run after 10:00 AM ET.
    Generates signals and logs them as OPEN trades.
    """
    today     = str(date.today())
    logged_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("━" * 55)
    logger.info("  NOVA PAPER TRADER — MORNING SESSION")
    logger.info(f"  Date   : {today}")
    logger.info(f"  Time   : {logged_at}")
    logger.info("━" * 55)

    df_log = load_log()

    # Check if already ran today
    if today in df_log["date"].values:
        logger.warning(f"Already logged signals for {today} — skipping")
        return

    # VIX check
    vix = get_vix()
    logger.info(f"VIX current: {vix:.1f}")
    if vix > VIX_MAX:
        logger.warning(f"VIX {vix:.1f} > {VIX_MAX} — NO TRADE today (macro regime)")
        logger.info("Nova sits out. No signals logged.")
        return

    # Layer 1: bias
    bias_result = get_daily_bias("QQQ")
    bias        = bias_result["bias"]

    if bias == "NO_TRADE":
        logger.warning("No clear bias — Nova sits out today")
        return

    logger.info(f"Bias: {bias} | QQQ ${bias_result['price']} "
                f"vs VWAP ${bias_result['vwap']} "
                f"({bias_result['pct_diff']:+}%)")

    # Layer 2: stocks in play
    in_play = scan_stocks_in_play(watchlist=NOVA_WATCHLIST)

    if in_play.empty:
        logger.warning("No stocks in play today")
        return

    # Current equity
    equity  = get_equity(df_log)
    account = AccountState(
        equity         = equity,
        peak_equity    = max(equity, EQUITY_START),
        daily_pnl      = 0.0,
        open_positions = 0,
    )

    new_rows = []

    for _, row in in_play.iterrows():
        ticker = row["ticker"]

        if account.open_positions >= 3:
            break

        # ATR
        df_daily = fetch_daily(ticker, period="3mo")
        if df_daily.empty:
            continue
        if isinstance(df_daily.columns, pd.MultiIndex):
            df_daily = df_daily.droplevel(1, axis=1)
        atr_val = float(compute_atr(df_daily).iloc[-1])

        # ORB signal
        signal = detect_breakout(ticker, bias, atr_val)
        if signal is None:
            continue

        # Risk approval
        sized = approve_signal(signal, account)
        if sized is None:
            continue

        account.open_positions  += 1
        account.daily_risk_used += sized.dollar_risk

        new_rows.append({
            "date"       : today,
            "ticker"     : ticker,
            "direction"  : signal.direction,
            "bias"       : bias,
            "entry"      : signal.entry,
            "stop"       : signal.stop,
            "target"     : signal.target,
            "rvol"       : row["rvol"],
            "gap_pct"    : row["gap_pct"],
            "atr"        : round(atr_val, 2),
            "shares"     : sized.shares,
            "dollar_risk": sized.dollar_risk,
            "exit_price" : "",
            "exit_reason": "",
            "pnl_r"      : "",
            "pnl_dollar" : "",
            "status"     : "OPEN",
            "logged_at"  : logged_at,
            "closed_at"  : "",
        })

        logger.info(f"LOGGED → {ticker} {signal.direction} | "
                    f"Entry ${signal.entry} | Stop ${signal.stop} | "
                    f"Target ${signal.target} | "
                    f"Shares {sized.shares} | Risk ${sized.dollar_risk}")

    if not new_rows:
        logger.warning("No signals passed all filters today")
        return

    df_new = pd.DataFrame(new_rows)
    df_log = pd.concat([df_log, df_new], ignore_index=True)
    save_log(df_log)

    logger.info(f"\n  {len(new_rows)} signal(s) logged to {PAPER_LOG}")
    logger.info("  Run evening session after market close to record outcomes.")


# ── Evening session ────────────────────────────────────────
def run_evening():
    """
    Evening routine — run after 4:00 PM ET.
    Checks actual price action and closes open trades.
    """
    today      = str(date.today())
    closed_at  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("━" * 55)
    logger.info("  NOVA PAPER TRADER — EVENING SESSION")
    logger.info(f"  Date: {today}")
    logger.info("━" * 55)

    df_log   = load_log()
    open_trades = df_log[
        (df_log["status"] == "OPEN") &
        (df_log["date"]   == today)
    ]

    if open_trades.empty:
        logger.info("No open trades to close today")
        return

    closed_count = 0

    for idx, trade in open_trades.iterrows():
        ticker    = trade["ticker"]
        direction = trade["direction"]
        entry     = float(trade["entry"])
        stop      = float(trade["stop"])
        target    = float(trade["target"])
        shares    = int(trade["shares"])

        # Fetch today's actual OHLCV
        try:
            df = yf.download(ticker, period="2d", interval="1d",
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(level=1, axis=1)

            today_bar = df.iloc[-1]
            high      = float(today_bar["High"])
            low       = float(today_bar["Low"])
            close     = float(today_bar["Close"])

        except Exception as e:
            logger.warning(f"Could not fetch {ticker}: {e}")
            continue

        # Determine outcome
        if direction == "LONG":
            if high >= target:
                exit_price, exit_reason = target, "TARGET"
            elif low <= stop:
                exit_price, exit_reason = stop, "STOP"
            else:
                exit_price, exit_reason = close, "TIME"
            pnl_r = (exit_price - entry) / (entry - stop)

        else:  # SHORT
            if low <= target:
                exit_price, exit_reason = target, "TARGET"
            elif high >= stop:
                exit_price, exit_reason = stop, "STOP"
            else:
                exit_price, exit_reason = close, "TIME"
            pnl_r = (entry - exit_price) / (stop - entry)

        pnl_dollar = round(pnl_r * float(trade["dollar_risk"]), 2)

        # Update log
        df_log.at[idx, "exit_price"]  = round(exit_price, 2)
        df_log.at[idx, "exit_reason"] = exit_reason
        df_log.at[idx, "pnl_r"]       = round(pnl_r, 4)
        df_log.at[idx, "pnl_dollar"]  = pnl_dollar
        df_log.at[idx, "status"]      = "CLOSED"
        df_log.at[idx, "closed_at"]   = closed_at

        result_icon = "✅" if pnl_r > 0 else "❌"
        logger.info(f"{result_icon} {ticker} {direction} | "
                    f"Exit ${exit_price:.2f} ({exit_reason}) | "
                    f"P&L {pnl_r:+.3f}R / ${pnl_dollar:+.2f}")

        closed_count += 1

    save_log(df_log)
    logger.info(f"\n  {closed_count} trade(s) closed and recorded")

    # Quick daily summary
    today_closed = df_log[
        (df_log["date"] == today) &
        (df_log["status"] == "CLOSED")
    ]
    if not today_closed.empty:
        daily_r      = today_closed["pnl_r"].astype(float).sum()
        daily_dollar = today_closed["pnl_dollar"].astype(float).sum()
        logger.info(f"  Today's P&L: {daily_r:+.3f}R / ${daily_dollar:+.2f}")


# ── Report session ─────────────────────────────────────────
def run_report():
    """Print full forward-test summary from the paper log."""
    df_log = load_log()
    closed = df_log[df_log["status"] == "CLOSED"].copy()

    if closed.empty:
        logger.info("No closed trades yet — run morning + evening sessions first")
        return

    closed["pnl_r"]      = closed["pnl_r"].astype(float)
    closed["pnl_dollar"] = closed["pnl_dollar"].astype(float)

    wins   = closed[closed["pnl_r"] > 0]
    losses = closed[closed["pnl_r"] <= 0]

    equity = EQUITY_START
    curve  = []
    for pnl in closed["pnl_dollar"]:
        equity += pnl
        curve.append(equity)
    closed["equity"] = curve
    peak = closed["equity"].cummax()
    dd   = ((closed["equity"] - peak) / peak * 100).min()

    gross_profit = wins["pnl_r"].sum()        if len(wins)   > 0 else 0
    gross_loss   = abs(losses["pnl_r"].sum()) if len(losses) > 0 else 1

    sep = "━" * 55
    print(f"\n{sep}")
    print("  NOVA — PAPER TRADING REPORT")
    print(sep)
    print(f"  Period        : {closed['date'].iloc[0]} → {closed['date'].iloc[-1]}")
    print(f"  Total trades  : {len(closed)}")
    print(f"  Win rate      : {len(wins)/len(closed)*100:.1f}%")
    print(f"  Expectancy    : {closed['pnl_r'].mean():+.4f}R")
    print(f"  Profit factor : {gross_profit/gross_loss:.2f}")
    print(f"  Total R       : {closed['pnl_r'].sum():+.3f}R")
    print(f"  Total P&L     : ${closed['pnl_dollar'].sum():+.2f}")
    print(f"  Final equity  : ${equity:,.2f}")
    print(f"  Max drawdown  : {dd:.2f}%")
    print(f"\n  {'Date':<12} {'Ticker':<7} {'Dir':<6} "
          f"{'Exit':<8} {'Reason':<8} {'R':>7} {'P&L':>9}")
    print(f"  {'─'*60}")
    for _, row in closed.iterrows():
        icon = "✅" if row["pnl_r"] > 0 else "❌"
        print(f"  {row['date']:<12} {row['ticker']:<7} "
              f"{row['direction']:<6} ${float(row['exit_price']):<7.2f} "
              f"{row['exit_reason']:<8} "
              f"{float(row['pnl_r']):>+7.3f}R "
              f"${float(row['pnl_dollar']):>+8.2f}  {icon}")
    print(f"{sep}\n")


# ── Entry point ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nova Paper Trader")
    parser.add_argument(
        "--mode",
        choices=["morning", "evening", "report"],
        default="morning",
        help="morning: log signals | evening: record outcomes | report: summary",
    )
    args = parser.parse_args()

    if args.mode == "morning":
        run_morning()
    elif args.mode == "evening":
        run_evening()
    elif args.mode == "report":
        run_report()