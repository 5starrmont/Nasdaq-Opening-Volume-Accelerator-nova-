"""
backtest/engine.py
Nova — historical backtest engine (daily-bar simulation)

Filters:
    1. Gap direction must align with QQQ MA20 trend
    2. VIX < 25 (no macro panic trading)
    3. RVOL >= 1.5x + gap >= 0.3%
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

# ── Constants ──────────────────────────────────────────────
RISK_PER_TRADE  = 0.01
RR_RATIO        = 2.0
ORB_ATR_FRAC    = 0.20
ATR_BUFFER_PCT  = 0.10
NEUTRAL_BAND    = 0.003
MIN_RVOL        = 1.5
MIN_GAP_PCT     = 0.003
VIX_MAX         = 25.0


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
    exit_reason  : str
    pnl_r        : float
    pnl_pct      : float
    rvol         : float
    gap_pct      : float
    atr          : float


@dataclass
class BacktestResult:
    trades            : list  = field(default_factory=list)
    equity_curve      : list  = field(default_factory=list)
    total_return      : float = 0.0
    sharpe_ratio      : float = 0.0
    sortino_ratio     : float = 0.0
    max_drawdown      : float = 0.0
    avg_drawdown      : float = 0.0
    win_rate          : float = 0.0
    profit_factor     : float = 0.0
    expectancy_r      : float = 0.0
    total_trades      : int   = 0
    winning_trades    : int   = 0
    losing_trades     : int   = 0
    time_exits        : int   = 0
    avg_win_r         : float = 0.0
    avg_loss_r        : float = 0.0
    max_consec_wins   : int   = 0
    max_consec_losses : int   = 0
    long_win_rate     : float = 0.0
    short_win_rate    : float = 0.0


# ── Data fetching ──────────────────────────────────────────
def fetch_daily_bulk(tickers: list, period: str = "2y") -> dict:
    logger.info(f"Fetching {period} daily data for {len(tickers)} tickers...")
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, interval="1d",
                             auto_adjust=True, progress=False)
            if df.empty or len(df) < 60:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(level=1, axis=1)
            df.index = pd.to_datetime(df.index)
            data[ticker] = df
            logger.success(f"{ticker} | {len(df)} bars | "
                           f"{df.index[0].date()} → {df.index[-1].date()}")
        except Exception as e:
            logger.warning(f"Failed {ticker}: {e}")
    return data


# ── Indicators ─────────────────────────────────────────────
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


def compute_daily_bias(
    df_qqq : pd.DataFrame,
    df_vix : pd.DataFrame,
) -> pd.Series:
    """
    Layer 1 bias — three conditions must all be true:
        1. Daily gap direction is meaningful (> NEUTRAL_BAND)
        2. Gap direction aligns with QQQ 20-day MA trend
        3. VIX is below VIX_MAX (not macro panic)
    """
    gap   = compute_gap_series(df_qqq)
    ma20  = df_qqq["Close"].rolling(20).mean()
    trend = df_qqq["Close"] > ma20          # True = BULL, False = BEAR

    vix_aligned = df_vix["Close"].reindex(df_qqq.index, method="ffill")

    bias = pd.Series("NO_TRADE", index=df_qqq.index)

    for date in df_qqq.index:
        g   = gap.get(date, 0)
        t   = trend.get(date, False)
        vix = vix_aligned.get(date, 0)

        if pd.isna(vix) or vix > VIX_MAX:
            continue
        if pd.isna(t) or pd.isna(g):
            continue

        if g > NEUTRAL_BAND and t:          # gap UP + BULL trend → LONG
            bias[date] = "LONG"
        elif g < -NEUTRAL_BAND and not t:   # gap DOWN + BEAR trend → SHORT
            bias[date] = "SHORT"

    return bias


# ── Trade simulation ───────────────────────────────────────
def simulate_daily_trade(
    row       : pd.Series,
    direction : str,
    atr       : float,
) -> dict | None:
    o = float(row["Open"])
    h = float(row["High"])
    l = float(row["Low"])
    c = float(row["Close"])

    orb_high = o + atr * ORB_ATR_FRAC
    orb_low  = o - atr * ORB_ATR_FRAC
    buffer   = atr * ATR_BUFFER_PCT

    if direction == "LONG":
        if c < o:
            return None
        entry  = orb_high + buffer
        stop   = orb_low
        risk   = entry - stop
        if risk <= 0:
            return None
        target = entry + risk * RR_RATIO

        if h >= target:
            exit_price, exit_reason = target, "TARGET"
        elif l <= stop:
            exit_price, exit_reason = stop, "STOP"
        else:
            exit_price, exit_reason = c, "TIME"

        pnl_r = (exit_price - entry) / risk

    else:
        if c > o:
            return None
        entry  = orb_low - buffer
        stop   = orb_high
        risk   = stop - entry
        if risk <= 0:
            return None
        target = entry - risk * RR_RATIO

        if l <= target:
            exit_price, exit_reason = target, "TARGET"
        elif h >= stop:
            exit_price, exit_reason = stop, "STOP"
        else:
            exit_price, exit_reason = c, "TIME"

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

    daily_ret = df.groupby("date")["pnl_pct"].sum()
    mean_r    = daily_ret.mean()
    std_r     = daily_ret.std()
    down_std  = daily_ret[daily_ret < 0].std()
    sharpe    = mean_r / std_r    * np.sqrt(252) if std_r    > 0 else 0
    sortino   = mean_r / down_std * np.sqrt(252) if down_std > 0 else 0

    import itertools
    streaks    = (df["pnl_r"] > 0).astype(int).tolist()
    max_wins   = max((sum(1 for _ in g)
                      for k, g in itertools.groupby(streaks) if k == 1), default=0)
    max_losses = max((sum(1 for _ in g)
                      for k, g in itertools.groupby(streaks) if k == 0), default=0)

    total_return = (df["equity"].iloc[-1] - initial_equity) / initial_equity

    return BacktestResult(
        trades            = trades,
        equity_curve      = df["equity"].tolist(),
        total_return      = round(total_return * 100, 2),
        sharpe_ratio      = round(sharpe, 2),
        sortino_ratio     = round(sortino, 2),
        max_drawdown      = round(df["dd"].min(), 2),
        avg_drawdown      = round(df["dd"][df["dd"] < 0].mean(), 2)
                            if any(df["dd"] < 0) else 0.0,
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
        long_win_rate     = round(len(longs[longs["pnl_r"] > 0])
                            / len(longs)  * 100, 2) if len(longs)  > 0 else 0,
        short_win_rate    = round(len(shorts[shorts["pnl_r"] > 0])
                            / len(shorts) * 100, 2) if len(shorts) > 0 else 0,
    )


# ── Report ─────────────────────────────────────────────────
def print_full_report(result: BacktestResult):
    df = pd.DataFrame([t.__dict__ for t in result.trades])
    df = df.sort_values("date").reset_index(drop=True)
    df["equity"] = result.equity_curve
    sep = "━" * 60

    print(f"\n{sep}")
    print("  NOVA — FULL BACKTEST REPORT")
    print(sep)

    print(f"\n  {'OVERVIEW':─<50}")
    print(f"  Period          : {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
    print(f"  Total trades    : {result.total_trades}")
    print(f"  Trading days    : {df['date'].nunique()}")
    print(f"  Avg trades/month: "
          f"{result.total_trades / max(1, df['date'].nunique() / 21):.1f}")

    print(f"\n  {'RETURNS':─<50}")
    print(f"  Total return    : {result.total_return:+.2f}%")
    print(f"  Expectancy      : {result.expectancy_r:+.4f}R per trade")
    print(f"  Profit factor   : {result.profit_factor}")
    print(f"  Avg win         : {result.avg_win_r:+.4f}R")
    print(f"  Avg loss        : {result.avg_loss_r:+.4f}R")

    print(f"\n  {'RISK':─<50}")
    print(f"  Sharpe ratio    : {result.sharpe_ratio}")
    print(f"  Sortino ratio   : {result.sortino_ratio}")
    print(f"  Max drawdown    : {result.max_drawdown:.2f}%")
    print(f"  Avg drawdown    : {result.avg_drawdown:.2f}%")

    print(f"\n  {'WIN / LOSS':─<50}")
    print(f"  Win rate        : {result.win_rate}%")
    print(f"  Winning trades  : {result.winning_trades}")
    print(f"  Losing trades   : {result.losing_trades}")
    print(f"  Time exits      : {result.time_exits}")
    print(f"  Max consec wins : {result.max_consec_wins}")
    print(f"  Max consec loss : {result.max_consec_losses}")

    print(f"\n  {'DIRECTION BREAKDOWN':─<50}")
    print(f"  LONG  win rate  : {result.long_win_rate}%")
    print(f"  SHORT win rate  : {result.short_win_rate}%")
    print(f"  LONG  trades    : "
          f"{len(df[df['direction'] == 'LONG'])}")
    print(f"  SHORT trades    : "
          f"{len(df[df['direction'] == 'SHORT'])}")

    exits = df["exit_reason"].value_counts()
    print(f"\n  {'EXIT BREAKDOWN':─<50}")
    for reason, count in exits.items():
        print(f"  {reason:<16}: {count:>4} ({count/len(df)*100:.1f}%)")

    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    monthly = df.groupby("month").agg(
        trades   = ("pnl_r", "count"),
        total_r  = ("pnl_r", "sum"),
        win_rate = ("pnl_r", lambda x: (x > 0).mean() * 100),
    ).round(2)

    print(f"\n  {'MONTHLY BREAKDOWN':─<50}")
    print(f"  {'Month':<10} {'Trades':>7} {'Total R':>10} {'Win%':>7}")
    print(f"  {'─'*38}")
    for month, row in monthly.iterrows():
        sign = "+" if row["total_r"] >= 0 else ""
        bar  = "█" * int(abs(row["total_r"]) * 2)
        print(f"  {str(month):<10} {int(row['trades']):>7} "
              f"  {sign}{row['total_r']:>7.3f}R  {row['win_rate']:>5.1f}%  {bar}")

    ticker_stats = df.groupby("ticker").agg(
        trades   = ("pnl_r", "count"),
        total_r  = ("pnl_r", "sum"),
        win_rate = ("pnl_r", lambda x: (x > 0).mean() * 100),
        avg_r    = ("pnl_r", "mean"),
    ).sort_values("total_r", ascending=False).round(3)

    print(f"\n  {'PER-TICKER BREAKDOWN':─<50}")
    print(f"  {'Ticker':<8} {'Trades':>7} {'Total R':>10} "
          f"{'Win%':>7} {'Avg R':>9}")
    print(f"  {'─'*46}")
    for ticker, row in ticker_stats.iterrows():
        sign = "+" if row["total_r"] >= 0 else ""
        print(f"  {ticker:<8} {int(row['trades']):>7}  "
              f"{sign}{row['total_r']:>8.3f}R  {row['win_rate']:>5.1f}%  "
              f"{sign}{row['avg_r']:>7.4f}R")

    print(f"\n{sep}\n")


# ── Chart ──────────────────────────────────────────────────
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

    colors = df["exit_reason"].map({
        "TARGET": "#2ecc71",
        "STOP"  : "#e74c3c",
        "TIME"  : "#95a5a6",
    })

    BG  = "#0d0d0d"
    fig = plt.figure(figsize=(16, 11), facecolor=BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            height_ratios=[3, 1, 1],
                            hspace=0.45, wspace=0.3)

    ax_eq  = fig.add_subplot(gs[0, :])
    ax_dd  = fig.add_subplot(gs[1, :])
    ax_rr  = fig.add_subplot(gs[2, 0])
    ax_mon = fig.add_subplot(gs[2, 1])

    for ax in [ax_eq, ax_dd, ax_rr, ax_mon]:
        ax.set_facecolor(BG)
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a2a")

    fig.suptitle("Nova — Full Backtest Report (MA20 + VIX filter)",
                 fontsize=16, fontweight="bold", color="white", y=0.98)

    x = range(len(df))

    # Equity
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

    final = df["equity"].iloc[-1]
    ret   = (final - initial_equity) / initial_equity * 100
    ax_eq.annotate(f"  ${final:,.0f}  ({ret:+.1f}%)",
                   xy=(len(df)-1, final),
                   color="#00d4ff", fontsize=10, fontweight="bold")

    from matplotlib.lines import Line2D
    ax_eq.legend(handles=[
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor="#2ecc71", markersize=8, label="Target"),
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor="#e74c3c", markersize=8, label="Stop"),
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor="#95a5a6", markersize=8, label="Time"),
    ], facecolor="#1a1a1a", edgecolor="#333",
       labelcolor="white", fontsize=8, loc="upper left")

    # Drawdown
    ax_dd.fill_between(x, df["dd"], 0, alpha=0.55, color="#e74c3c")
    ax_dd.plot(x, df["dd"], color="#e74c3c", linewidth=1.0)
    ax_dd.axhline(0, color="#444", linewidth=0.6)
    ax_dd.set_ylabel("Drawdown (%)", color="#aaaaaa", fontsize=9)
    ax_dd.set_xlabel("Trade #",      color="#aaaaaa", fontsize=8)
    ax_dd.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax_dd.set_title("Drawdown", color="#888888", fontsize=9, pad=4)

    # R distribution
    wins_r   = df[df["pnl_r"] > 0]["pnl_r"]
    losses_r = df[df["pnl_r"] <= 0]["pnl_r"]
    if len(wins_r):
        ax_rr.hist(wins_r,   bins=20, color="#2ecc71",
                   alpha=0.7, label="Winners", edgecolor=BG)
    if len(losses_r):
        ax_rr.hist(losses_r, bins=20, color="#e74c3c",
                   alpha=0.7, label="Losers",  edgecolor=BG)
    ax_rr.axvline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax_rr.axvline(df["pnl_r"].mean(), color="#00d4ff", linewidth=1.2,
                  linestyle="--", label=f"Mean {df['pnl_r'].mean():+.3f}R")
    ax_rr.set_title("R Distribution",  color="#888888", fontsize=9, pad=4)
    ax_rr.set_xlabel("R multiple",     color="#aaaaaa", fontsize=8)
    ax_rr.set_ylabel("Frequency",      color="#aaaaaa", fontsize=8)
    ax_rr.legend(facecolor="#1a1a1a", edgecolor="#333",
                 labelcolor="white", fontsize=7)

    # Monthly bars
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    monthly_r   = df.groupby("month")["pnl_r"].sum()
    bar_colors  = ["#2ecc71" if v >= 0 else "#e74c3c" for v in monthly_r]
    ax_mon.bar(range(len(monthly_r)), monthly_r.values,
               color=bar_colors, alpha=0.8, edgecolor=BG, width=0.6)
    ax_mon.axhline(0, color="#aaaaaa", linewidth=0.6)
    ax_mon.set_xticks(range(len(monthly_r)))
    ax_mon.set_xticklabels([str(m) for m in monthly_r.index],
                           rotation=45, ha="right",
                           fontsize=7, color="#aaaaaa")
    ax_mon.set_title("Monthly P&L (R)", color="#888888", fontsize=9, pad=4)
    ax_mon.set_ylabel("Total R",        color="#aaaaaa", fontsize=8)
    ax_mon.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:+.1f}R"))

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    logger.success(f"Chart saved → {output_path}")


# ── Main runner ────────────────────────────────────────────
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
    logger.info("  NOVA BACKTEST ENGINE — MA20 + VIX Filter")
    logger.info(f"  Tickers : {len(watchlist)} stocks | Period: 2yr daily")
    logger.info(f"  Filters : RVOL≥{MIN_RVOL} | Gap≥{MIN_GAP_PCT*100:.1f}% "
                f"| MA20 aligned | VIX<{VIX_MAX}")
    logger.info("━" * 60)

    # Fetch stock data
    data = fetch_daily_bulk(all_tickers, period="2y")
    if qqq_ticker not in data:
        logger.error("QQQ data missing")
        return BacktestResult()

    # Fetch VIX
    logger.info("Fetching VIX regime filter...")
    df_vix = yf.download("^VIX", period="2y", interval="1d",
                          auto_adjust=True, progress=False)
    if isinstance(df_vix.columns, pd.MultiIndex):
        df_vix = df_vix.droplevel(level=1, axis=1)
    df_vix.index = pd.to_datetime(df_vix.index)
    logger.success(f"VIX | {len(df_vix)} bars | "
                   f"current: {df_vix['Close'].iloc[-1]:.1f}")

    # Compute bias
    qqq      = data[qqq_ticker]
    qqq_bias = compute_daily_bias(qqq, df_vix)
    qqq_atr  = compute_atr_series(qqq)

    # Precompute stock indicators
    stock_data = {}
    for ticker in watchlist:
        if ticker not in data:
            continue
        df = data[ticker]
        stock_data[ticker] = {
            "df"  : df,
            "atr" : compute_atr_series(df),
            "rvol": compute_rvol_series(df),
            "gap" : compute_gap_series(df),
        }

    # Bias stats
    bias_counts = qqq_bias.value_counts()
    logger.info(f"Bias distribution: "
                f"LONG={bias_counts.get('LONG',0)} | "
                f"SHORT={bias_counts.get('SHORT',0)} | "
                f"NO_TRADE={bias_counts.get('NO_TRADE',0)}")

    common_dates = sorted(
        set(qqq.index) & set(list(stock_data.values())[0]["df"].index)
    )

    all_trades   = []
    days_traded  = 0

    for date in common_dates:
        bias = qqq_bias.get(date, "NO_TRADE")
        if bias == "NO_TRADE":
            continue

        traded_today = 0

        for ticker, sd in stock_data.items():
            df = sd["df"]
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

            row    = df.loc[date]
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

    logger.info(f"Days scanned : {len(common_dates)}")
    logger.info(f"Days traded  : {days_traded}")
    logger.info(f"Total trades : {len(all_trades)}")

    result = compute_metrics(all_trades, initial_equity)
    print_full_report(result)
    plot_equity_curve(result, initial_equity)

    return result


if __name__ == "__main__":
    run_backtest(initial_equity=10_000)