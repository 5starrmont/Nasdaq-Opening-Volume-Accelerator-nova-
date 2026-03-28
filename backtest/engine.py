"""
backtest/engine.py
Nova — historical backtest engine (daily-bar simulation)

Uses 2 years of daily OHLCV data (free via yfinance) to simulate
Nova's 3-layer ORB logic. The 5-min opening range is approximated
as ±20% of the 14-day ATR from the daily open price.

This is the standard academic approximation when intraday data
is unavailable. The daily High/Low determines trade outcome.

Coverage: ~500 trading days × 10 tickers = statistically robust sample.
"""

import pandas as pd
import numpy as np
from loguru import logger
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf

# ── Strategy constants ─────────────────────────────────────
RISK_PER_TRADE  = 0.01      # 1% account risk per trade
RR_RATIO        = 2.0       # 2:1 reward to risk
ORB_ATR_FRAC    = 0.20      # opening range ≈ 20% of daily ATR
ATR_BUFFER_PCT  = 0.10      # entry buffer = 10% of ATR
NEUTRAL_BAND    = 0.0005    # 0.05% VWAP neutral band
MIN_RVOL        = 1.5       # relative volume threshold
MIN_GAP_PCT     = 0.003     # 0.3% minimum gap


# ── Data classes ───────────────────────────────────────────
@dataclass
class BacktestTrade:
    date         : str
    ticker       : str
    direction    : str
    bias         : str
    entry        : float
    stop         : float
    target       : float
    exit_price   : float
    exit_reason  : str       # TARGET | STOP | TIME
    pnl_r        : float     # P&L in R units
    pnl_pct      : float     # P&L as % of account
    rvol         : float
    gap_pct      : float
    atr          : float


@dataclass
class BacktestResult:
    trades           : list  = field(default_factory=list)
    equity_curve     : list  = field(default_factory=list)
    total_return     : float = 0.0
    sharpe_ratio     : float = 0.0
    sortino_ratio    : float = 0.0
    max_drawdown     : float = 0.0
    avg_drawdown     : float = 0.0
    win_rate         : float = 0.0
    profit_factor    : float = 0.0
    expectancy_r     : float = 0.0
    total_trades     : int   = 0
    winning_trades   : int   = 0
    losing_trades    : int   = 0
    time_exits       : int   = 0
    avg_win_r        : float = 0.0
    avg_loss_r       : float = 0.0
    max_consec_wins  : int   = 0
    max_consec_losses: int   = 0
    long_win_rate    : float = 0.0
    short_win_rate   : float = 0.0


# ── Data fetching ──────────────────────────────────────────
def fetch_daily_bulk(tickers: list, period: str = "2y") -> dict:
    """Fetch 2 years of daily OHLCV for all tickers."""
    logger.info(f"Fetching {period} daily data for {len(tickers)} tickers...")
    data = {}

    for ticker in tickers:
        try:
            df = yf.download(
                ticker, period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if df.empty or len(df) < 60:
                logger.warning(f"{ticker} — insufficient data")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(level=1, axis=1)

            df.index = pd.to_datetime(df.index)
            data[ticker] = df
            logger.success(f"{ticker} | {len(df)} daily bars | "
                           f"{df.index[0].date()} → {df.index[-1].date()}")

        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")

    return data


# ── Per-day calculations ───────────────────────────────────
def compute_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_rvol_series(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    return df["Volume"] / df["Volume"].rolling(lookback).mean()


def compute_gap_series(df: pd.DataFrame) -> pd.Series:
    return (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)


def compute_daily_bias(df_qqq: pd.DataFrame) -> pd.Series:
    """
    Approximate daily bias using open vs prior close.
    Gap up  → institutional buying pressure → LONG bias
    Gap down → institutional selling pressure → SHORT bias
    Near flat → NO_TRADE
    """
    gap = compute_gap_series(df_qqq)
    bias = pd.Series("NO_TRADE", index=df_qqq.index)
    bias[gap >  NEUTRAL_BAND] = "LONG"
    bias[gap < -NEUTRAL_BAND] = "SHORT"
    return bias


# ── Trade simulation ───────────────────────────────────────
def simulate_daily_trade(
    row       : pd.Series,
    direction : str,
    atr       : float,
) -> dict | None:
    """
    Simulate one ORB trade using daily OHLCV bar.

    Opening range approximation:
        ORB_high = Open + ATR × ORB_ATR_FRAC
        ORB_low  = Open - ATR × ORB_ATR_FRAC

    Outcome determined by whether daily High/Low
    reached the target or stop before close.
    """
    o = float(row["Open"])
    h = float(row["High"])
    l = float(row["Low"])
    c = float(row["Close"])

    orb_high = o + atr * ORB_ATR_FRAC
    orb_low  = o - atr * ORB_ATR_FRAC
    buffer   = atr * ATR_BUFFER_PCT

    if direction == "LONG":
        # First 5-min candle must be bullish (close > open approximated by gap up)
        if c < o:  # day closed down — skip
            return None
        entry  = orb_high + buffer
        stop   = orb_low
        risk   = entry - stop
        if risk <= 0:
            return None
        target = entry + risk * RR_RATIO

        # Simulate outcome using daily High/Low
        # Assume price tests both stop and target intraday —
        # if High >= target: TARGET hit
        # elif Low <= stop: STOP hit
        # else: TIME exit at Close
        if h >= target:
            exit_price  = target
            exit_reason = "TARGET"
        elif l <= stop:
            exit_price  = stop
            exit_reason = "STOP"
        else:
            exit_price  = c
            exit_reason = "TIME"

        pnl_r = (exit_price - entry) / risk

    else:  # SHORT
        if c > o:  # day closed up — skip
            return None
        entry  = orb_low - buffer
        stop   = orb_high
        risk   = stop - entry
        if risk <= 0:
            return None
        target = entry - risk * RR_RATIO

        if l <= target:
            exit_price  = target
            exit_reason = "TARGET"
        elif h >= stop:
            exit_price  = stop
            exit_reason = "STOP"
        else:
            exit_price  = c
            exit_reason = "TIME"

        pnl_r = (entry - exit_price) / risk

    return {
        "entry"      : round(entry, 2),
        "stop"       : round(stop, 2),
        "target"     : round(target, 2),
        "exit_price" : round(exit_price, 2),
        "exit_reason": exit_reason,
        "pnl_r"      : round(pnl_r, 4),
        "pnl_pct"    : round(pnl_r * RISK_PER_TRADE, 6),
        "atr"        : round(atr, 2),
    }


# ── Metrics ────────────────────────────────────────────────
def compute_metrics(trades: list, initial_equity: float) -> BacktestResult:
    if not trades:
        logger.warning("No trades to compute metrics on")
        return BacktestResult()

    df = pd.DataFrame([t.__dict__ for t in trades])
    df = df.sort_values("date").reset_index(drop=True)

    # Equity curve
    equity = [initial_equity]
    for pct in df["pnl_pct"]:
        equity.append(equity[-1] * (1 + pct))
    df["equity"] = equity[1:]
    df["peak"]   = df["equity"].cummax()
    df["dd"]     = (df["equity"] - df["peak"]) / df["peak"] * 100

    wins   = df[df["pnl_r"] > 0]
    losses = df[df["pnl_r"] <= 0]
    longs  = df[df["direction"] == "LONG"]
    shorts = df[df["direction"] == "SHORT"]

    gross_profit  = wins["pnl_r"].sum()        if len(wins)   > 0 else 0
    gross_loss    = abs(losses["pnl_r"].sum()) if len(losses) > 0 else 1
    profit_factor = gross_profit / gross_loss  if gross_loss  > 0 else 0

    # Sharpe & Sortino (annualized)
    daily_ret = df.groupby("date")["pnl_pct"].sum()
    mean_r    = daily_ret.mean()
    std_r     = daily_ret.std()
    down_std  = daily_ret[daily_ret < 0].std()
    sharpe    = mean_r / std_r   * np.sqrt(252) if std_r   > 0 else 0
    sortino   = mean_r / down_std* np.sqrt(252) if down_std > 0 else 0

    # Consecutive wins/losses
    streaks     = (df["pnl_r"] > 0).astype(int)
    max_wins    = max((sum(1 for _ in g) for k, g in
                       __import__("itertools").groupby(streaks) if k == 1), default=0)
    max_losses  = max((sum(1 for _ in g) for k, g in
                       __import__("itertools").groupby(streaks) if k == 0), default=0)

    total_return = (df["equity"].iloc[-1] - initial_equity) / initial_equity

    return BacktestResult(
        trades            = trades,
        equity_curve      = df["equity"].tolist(),
        total_return      = round(total_return * 100, 2),
        sharpe_ratio      = round(sharpe, 2),
        sortino_ratio     = round(sortino, 2),
        max_drawdown      = round(df["dd"].min(), 2),
        avg_drawdown      = round(df["dd"][df["dd"] < 0].mean(), 2) if any(df["dd"] < 0) else 0.0,
        win_rate          = round(len(wins) / len(df) * 100, 2),
        profit_factor     = round(profit_factor, 2),
        expectancy_r      = round(df["pnl_r"].mean(), 4),
        total_trades      = len(df),
        winning_trades    = len(wins),
        losing_trades     = len(losses),
        time_exits        = len(df[df["exit_reason"] == "TIME"]),
        avg_win_r         = round(wins["pnl_r"].mean(), 4)   if len(wins)   > 0 else 0,
        avg_loss_r        = round(losses["pnl_r"].mean(), 4) if len(losses) > 0 else 0,
        max_consec_wins   = max_wins,
        max_consec_losses = max_losses,
        long_win_rate     = round(len(longs[longs["pnl_r"] > 0]) / len(longs) * 100, 2) if len(longs)  > 0 else 0,
        short_win_rate    = round(len(shorts[shorts["pnl_r"] > 0]) / len(shorts) * 100, 2) if len(shorts) > 0 else 0,
    )


# ── Rich reporting ─────────────────────────────────────────
def print_full_report(result: BacktestResult):
    """Print a comprehensive quant research report."""

    df = pd.DataFrame([t.__dict__ for t in result.trades])
    df = df.sort_values("date").reset_index(drop=True)
    df["equity"] = result.equity_curve

    sep = "━" * 60

    print(f"\n{sep}")
    print("  NOVA — FULL BACKTEST REPORT")
    print(sep)

    # Overview
    print(f"\n  {'OVERVIEW':─<50}")
    print(f"  Period          : {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
    print(f"  Total trades    : {result.total_trades}")
    print(f"  Trading days    : {df['date'].nunique()}")
    print(f"  Avg trades/month: {result.total_trades / max(1, df['date'].nunique() / 21):.1f}")

    # Returns
    print(f"\n  {'RETURNS':─<50}")
    print(f"  Total return    : {result.total_return:+.2f}%")
    print(f"  Expectancy      : {result.expectancy_r:+.4f}R per trade")
    print(f"  Profit factor   : {result.profit_factor}")
    print(f"  Avg win         : {result.avg_win_r:+.4f}R")
    print(f"  Avg loss        : {result.avg_loss_r:+.4f}R")

    # Risk
    print(f"\n  {'RISK':─<50}")
    print(f"  Sharpe ratio    : {result.sharpe_ratio}")
    print(f"  Sortino ratio   : {result.sortino_ratio}")
    print(f"  Max drawdown    : {result.max_drawdown:.2f}%")
    print(f"  Avg drawdown    : {result.avg_drawdown:.2f}%")

    # Win/Loss
    print(f"\n  {'WIN / LOSS':─<50}")
    print(f"  Win rate        : {result.win_rate}%")
    print(f"  Winning trades  : {result.winning_trades}")
    print(f"  Losing trades   : {result.losing_trades}")
    print(f"  Time exits      : {result.time_exits}")
    print(f"  Max consec wins : {result.max_consec_wins}")
    print(f"  Max consec loss : {result.max_consec_losses}")

    # Direction breakdown
    print(f"\n  {'DIRECTION BREAKDOWN':─<50}")
    print(f"  LONG  win rate  : {result.long_win_rate}%")
    print(f"  SHORT win rate  : {result.short_win_rate}%")
    long_trades  = len(df[df["direction"] == "LONG"])
    short_trades = len(df[df["direction"] == "SHORT"])
    print(f"  LONG  trades    : {long_trades}")
    print(f"  SHORT trades    : {short_trades}")

    # Exit breakdown
    exits = df["exit_reason"].value_counts()
    print(f"\n  {'EXIT BREAKDOWN':─<50}")
    for reason, count in exits.items():
        pct = count / len(df) * 100
        print(f"  {reason:<16}: {count:>4} ({pct:.1f}%)")

    # Monthly P&L
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    monthly = df.groupby("month").agg(
        trades   = ("pnl_r", "count"),
        total_r  = ("pnl_r", "sum"),
        win_rate = ("pnl_r", lambda x: (x > 0).mean() * 100),
    ).round(2)

    print(f"\n  {'MONTHLY BREAKDOWN':─<50}")
    print(f"  {'Month':<10} {'Trades':>7} {'Total R':>9} {'Win%':>8}")
    print(f"  {'─'*38}")
    for month, row in monthly.iterrows():
        bar = "█" * int(abs(row["total_r"]) * 3)
        sign = "+" if row["total_r"] >= 0 else ""
        print(f"  {str(month):<10} {int(row['trades']):>7} "
              f"{sign}{row['total_r']:>8.3f}R  {row['win_rate']:>6.1f}%  {bar}")

    # Per-ticker breakdown
    ticker_stats = df.groupby("ticker").agg(
        trades   = ("pnl_r", "count"),
        total_r  = ("pnl_r", "sum"),
        win_rate = ("pnl_r", lambda x: (x > 0).mean() * 100),
        avg_r    = ("pnl_r", "mean"),
    ).sort_values("total_r", ascending=False).round(3)

    print(f"\n  {'PER-TICKER BREAKDOWN':─<50}")
    print(f"  {'Ticker':<8} {'Trades':>7} {'Total R':>9} {'Win%':>8} {'Avg R':>8}")
    print(f"  {'─'*44}")
    for ticker, row in ticker_stats.iterrows():
        sign = "+" if row["total_r"] >= 0 else ""
        print(f"  {ticker:<8} {int(row['trades']):>7} "
              f"{sign}{row['total_r']:>8.3f}R  {row['win_rate']:>6.1f}%  "
              f"{sign}{row['avg_r']:>7.4f}R")

    print(f"\n{sep}\n")


# ── Equity curve chart ─────────────────────────────────────
def plot_equity_curve(
    result         : BacktestResult,
    initial_equity : float,
    output_path    : str = "reports/nova_equity_curve.png",
):
    Path(output_path).parent.mkdir(exist_ok=True)

    df = pd.DataFrame([t.__dict__ for t in result.trades])
    df = df.sort_values("date").reset_index(drop=True)
    df["equity"] = result.equity_curve
    df["peak"]   = df["equity"].cummax()
    df["dd"]     = (df["equity"] - df["peak"]) / df["peak"] * 100
    df["date_dt"] = pd.to_datetime(df["date"])

    colors = df["exit_reason"].map({
        "TARGET": "#2ecc71",
        "STOP"  : "#e74c3c",
        "TIME"  : "#95a5a6",
    })

    fig = plt.figure(figsize=(16, 11), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[3, 1, 1],
        hspace=0.45, wspace=0.3,
    )

    ax_eq  = fig.add_subplot(gs[0, :])   # equity curve — full width
    ax_dd  = fig.add_subplot(gs[1, :])   # drawdown — full width
    ax_rr  = fig.add_subplot(gs[2, 0])   # R distribution
    ax_mon = fig.add_subplot(gs[2, 1])   # monthly P&L bar

    BG = "#0d0d0d"
    for ax in [ax_eq, ax_dd, ax_rr, ax_mon]:
        ax.set_facecolor(BG)
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a2a")

    fig.suptitle(
        "Nova — Full Backtest Report",
        fontsize=16, fontweight="bold", color="white", y=0.98,
    )

    # ── Equity curve ───────────────────────────────────────
    x = range(len(df))
    ax_eq.plot(x, df["equity"], color="#00d4ff", linewidth=1.8, zorder=2)
    ax_eq.fill_between(x, initial_equity, df["equity"],
                       where=df["equity"] >= initial_equity,
                       alpha=0.15, color="#2ecc71")
    ax_eq.fill_between(x, initial_equity, df["equity"],
                       where=df["equity"] < initial_equity,
                       alpha=0.20, color="#e74c3c")
    ax_eq.scatter(x, df["equity"], c=colors, s=22, zorder=3, alpha=0.85)
    ax_eq.axhline(initial_equity, color="#444", linewidth=0.8, linestyle="--")

    ax_eq.set_ylabel("Account Value ($)", color="#aaaaaa", fontsize=9)
    ax_eq.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax_eq.set_title("Equity Curve", color="#888888", fontsize=9, pad=4)

    # Annotate final
    final = df["equity"].iloc[-1]
    ret   = (final - initial_equity) / initial_equity * 100
    ax_eq.annotate(
        f"  ${final:,.0f}  ({ret:+.1f}%)",
        xy=(len(df)-1, final),
        color="#00d4ff", fontsize=10, fontweight="bold",
    )

    # Legend
    from matplotlib.lines import Line2D
    ax_eq.legend(
        handles=[
            Line2D([0],[0], marker="o", color="w",
                   markerfacecolor="#2ecc71", markersize=8, label="Target"),
            Line2D([0],[0], marker="o", color="w",
                   markerfacecolor="#e74c3c", markersize=8, label="Stop"),
            Line2D([0],[0], marker="o", color="w",
                   markerfacecolor="#95a5a6", markersize=8, label="Time"),
        ],
        facecolor="#1a1a1a", edgecolor="#333",
        labelcolor="white", fontsize=8, loc="upper left",
    )

    # ── Drawdown ───────────────────────────────────────────
    ax_dd.fill_between(x, df["dd"], 0, alpha=0.55, color="#e74c3c")
    ax_dd.plot(x, df["dd"], color="#e74c3c", linewidth=1.0)
    ax_dd.axhline(0, color="#444", linewidth=0.6)
    ax_dd.set_ylabel("Drawdown (%)", color="#aaaaaa", fontsize=9)
    ax_dd.set_xlabel("Trade #", color="#aaaaaa", fontsize=8)
    ax_dd.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax_dd.set_title("Drawdown", color="#888888", fontsize=9, pad=4)

    # ── R distribution ─────────────────────────────────────
    wins_r   = df[df["pnl_r"] > 0]["pnl_r"]
    losses_r = df[df["pnl_r"] <= 0]["pnl_r"]

    if len(wins_r):
        ax_rr.hist(wins_r,   bins=20, color="#2ecc71",
                   alpha=0.7, label="Winners", edgecolor="#0d0d0d")
    if len(losses_r):
        ax_rr.hist(losses_r, bins=20, color="#e74c3c",
                   alpha=0.7, label="Losers",  edgecolor="#0d0d0d")

    ax_rr.axvline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax_rr.axvline(df["pnl_r"].mean(), color="#00d4ff",
                  linewidth=1.2, linestyle="--",
                  label=f"Mean {df['pnl_r'].mean():+.3f}R")
    ax_rr.set_title("R Distribution", color="#888888", fontsize=9, pad=4)
    ax_rr.set_xlabel("R multiple", color="#aaaaaa", fontsize=8)
    ax_rr.set_ylabel("Frequency",   color="#aaaaaa", fontsize=8)
    ax_rr.legend(facecolor="#1a1a1a", edgecolor="#333",
                 labelcolor="white", fontsize=7)

    # ── Monthly P&L bar chart ──────────────────────────────
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    monthly_r   = df.groupby("month")["pnl_r"].sum()
    bar_colors  = ["#2ecc71" if v >= 0 else "#e74c3c" for v in monthly_r]

    ax_mon.bar(range(len(monthly_r)), monthly_r.values,
               color=bar_colors, alpha=0.8, edgecolor="#0d0d0d", width=0.6)
    ax_mon.axhline(0, color="#aaaaaa", linewidth=0.6)
    ax_mon.set_xticks(range(len(monthly_r)))
    ax_mon.set_xticklabels(
        [str(m) for m in monthly_r.index],
        rotation=45, ha="right", fontsize=7, color="#aaaaaa",
    )
    ax_mon.set_title("Monthly P&L (R)", color="#888888", fontsize=9, pad=4)
    ax_mon.set_ylabel("Total R",        color="#aaaaaa", fontsize=8)
    ax_mon.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:+.1f}R"))

    plt.savefig(output_path, dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    logger.success(f"Chart saved → {output_path}")


# ── Main backtest runner ───────────────────────────────────
def run_backtest(
    watchlist      : list  = None,
    qqq_ticker     : str   = "QQQ",
    initial_equity : float = 10_000,
) -> BacktestResult:

    if watchlist is None:
        watchlist = [
            "META",  "NVDA", "TSLA", "AAPL", "MSFT",
            "AMZN",  "AMD",  "GOOGL","CRWD", "PLTR",
            "PANW",  "NFLX", "SNOW", "MU",   "QCOM",
        ]

    all_tickers = [qqq_ticker] + watchlist

    logger.info("━" * 60)
    logger.info("  NOVA BACKTEST ENGINE — Daily Bar Simulation")
    logger.info(f"  Tickers    : {len(watchlist)} stocks")
    logger.info(f"  Period     : 2 years daily OHLCV")
    logger.info(f"  Equity     : ${initial_equity:,.0f}")
    logger.info(f"  RVOL min   : {MIN_RVOL}×")
    logger.info(f"  Gap min    : {MIN_GAP_PCT*100:.1f}%")
    logger.info(f"  ORB frac   : {ORB_ATR_FRAC*100:.0f}% of ATR")
    logger.info("━" * 60)

    # Fetch all data
    data = fetch_daily_bulk(all_tickers, period="2y")

    if qqq_ticker not in data:
        logger.error("QQQ data missing")
        return BacktestResult()

    # Compute QQQ bias series
    qqq        = data[qqq_ticker]
    qqq_bias   = compute_daily_bias(qqq)
    qqq_atr    = compute_atr_series(qqq)

    all_trades  = []
    days_scanned = 0
    days_traded  = 0

    # Precompute indicators for each stock
    stock_data = {}
    for ticker in watchlist:
        if ticker not in data:
            continue
        df          = data[ticker]
        atr         = compute_atr_series(df)
        rvol        = compute_rvol_series(df)
        gap         = compute_gap_series(df)
        stock_data[ticker] = {
            "df"  : df,
            "atr" : atr,
            "rvol": rvol,
            "gap" : gap,
        }

    # Iterate over every trading day
    common_dates = sorted(set(qqq.index) & set(list(stock_data.values())[0]["df"].index))

    for date in common_dates:
        days_scanned += 1

        bias = qqq_bias.get(date, "NO_TRADE")
        if bias == "NO_TRADE":
            continue

        traded_today = 0

        for ticker, sd in stock_data.items():
            df   = sd["df"]
            if date not in df.index:
                continue

            rvol_val = sd["rvol"].get(date, 0)
            gap_val  = sd["gap"].get(date, 0)
            atr_val  = sd["atr"].get(date, 0)

            if rvol_val < MIN_RVOL:
                continue
            if abs(gap_val) < MIN_GAP_PCT:
                continue
            if atr_val <= 0:
                continue

            row = df.loc[date]
            result = simulate_daily_trade(row, bias, atr_val)

            if result is None:
                continue

            all_trades.append(BacktestTrade(
                date        = str(date.date()),
                ticker      = ticker,
                direction   = bias,
                bias        = bias,
                entry       = result["entry"],
                stop        = result["stop"],
                target      = result["target"],
                exit_price  = result["exit_price"],
                exit_reason = result["exit_reason"],
                pnl_r       = result["pnl_r"],
                pnl_pct     = result["pnl_pct"],
                rvol        = round(rvol_val, 2),
                gap_pct     = round(gap_val * 100, 3),
                atr         = result["atr"],
            ))
            traded_today += 1

        if traded_today > 0:
            days_traded += 1

    logger.info(f"Days scanned : {days_scanned}")
    logger.info(f"Days traded  : {days_traded}")
    logger.info(f"Total trades : {len(all_trades)}")

    # Compute metrics
    result = compute_metrics(all_trades, initial_equity)

    # Full report
    print_full_report(result)

    # Chart
    plot_equity_curve(result, initial_equity)

    return result


if __name__ == "__main__":
    run_backtest(initial_equity=10_000)