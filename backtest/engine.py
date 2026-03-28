"""
backtest/engine.py
Nova — historical backtest engine
"""

import pandas as pd
import numpy as np
from loguru import logger
from dataclasses import dataclass, field
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import yfinance as yf

# ── Constants ──────────────────────────────────────────────
RISK_PER_TRADE  = 0.01
RR_RATIO        = 2.0
ATR_BUFFER_PCT  = 0.10
NEUTRAL_BAND    = 0.0005
MIN_RVOL        = 1.5
MIN_GAP_PCT     = 0.003
OPEN_BAR        = "13:30"
BIAS_BAR        = "14:00"
EOD_EXIT        = "19:30"


@dataclass
class BacktestTrade:
    date        : str
    ticker      : str
    direction   : str
    entry       : float
    stop        : float
    target      : float
    exit_price  : float
    exit_reason : str
    pnl_r       : float
    pnl_pct     : float


@dataclass
class BacktestResult:
    trades          : list  = field(default_factory=list)
    total_return    : float = 0.0
    sharpe_ratio    : float = 0.0
    max_drawdown    : float = 0.0
    win_rate        : float = 0.0
    profit_factor   : float = 0.0
    expectancy_r    : float = 0.0
    total_trades    : int   = 0
    winning_trades  : int   = 0
    losing_trades   : int   = 0


def fetch_data_bulk(tickers: list, period: str = "2y") -> dict:
    logger.info(f"Fetching bulk data for {len(tickers)} tickers...")
    data = {}

    for ticker in tickers:
        try:
            daily = yf.download(
                ticker, period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            intraday = yf.download(
                ticker, period="60d",
                interval="5m",
                auto_adjust=True,
                progress=False,
            )

            if daily.empty or intraday.empty:
                continue

            if isinstance(daily.columns, pd.MultiIndex):
                daily = daily.droplevel(level=1, axis=1)
            if isinstance(intraday.columns, pd.MultiIndex):
                intraday = intraday.droplevel(level=1, axis=1)

            intraday.index = pd.to_datetime(intraday.index, utc=True)

            data[ticker] = {"daily": daily, "intraday": intraday}
            logger.success(f"{ticker} | {len(daily)} daily | {len(intraday)} intraday bars")

        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")

    return data


def get_bias_for_day(intraday: pd.DataFrame, date: pd.Timestamp) -> str:
    day_bars = intraday[intraday.index.date == date.date()]

    if day_bars.empty:
        return "NO_TRADE"

    tp   = (day_bars["High"] + day_bars["Low"] + day_bars["Close"]) / 3
    vwap = (tp * day_bars["Volume"]).cumsum() / day_bars["Volume"].cumsum()

    bias_bars = day_bars[day_bars.index.strftime("%H:%M") <= BIAS_BAR]

    if bias_bars.empty:
        return "NO_TRADE"

    idx      = bias_bars.index[-1]
    price    = float(bias_bars.loc[idx, "Close"])
    vwap_val = float(vwap.loc[idx])
    pct_diff = (price - vwap_val) / vwap_val

    if abs(pct_diff) <= NEUTRAL_BAND:
        return "NO_TRADE"
    return "LONG" if price > vwap_val else "SHORT"


def get_rvol_for_day(daily: pd.DataFrame, date: pd.Timestamp,
                     lookback: int = 20) -> float:
    date_naive = date.tz_localize(None) if date.tzinfo else date
    past       = daily[daily.index < date_naive]

    if len(past) < lookback:
        return 0.0

    avg_vol   = past["Volume"].iloc[-lookback:].mean()
    today_vol = daily.loc[daily.index == date_naive, "Volume"]

    if today_vol.empty or avg_vol == 0:
        return 0.0

    return float(today_vol.iloc[0]) / avg_vol


def get_gap_for_day(daily: pd.DataFrame, date: pd.Timestamp) -> float:
    date_naive = date.tz_localize(None) if date.tzinfo else date
    past       = daily[daily.index <= date_naive]

    if len(past) < 2:
        return 0.0

    prev_close  = float(past["Close"].iloc[-2])
    todays_open = float(past["Open"].iloc[-1])

    if prev_close == 0:
        return 0.0

    return (todays_open - prev_close) / prev_close


def simulate_trade(
    intraday  : pd.DataFrame,
    date      : pd.Timestamp,
    direction : str,
    atr_daily : float,
) -> BacktestTrade | None:
    day_bars = intraday[intraday.index.date == date.date()]

    if day_bars.empty:
        return None

    open_bars = day_bars[day_bars.index.strftime("%H:%M") == OPEN_BAR]

    if open_bars.empty:
        return None

    orb_bar  = open_bars.iloc[0]
    orb_high = float(orb_bar["High"])
    orb_low  = float(orb_bar["Low"])
    orb_dir  = "LONG" if orb_bar["Close"] >= orb_bar["Open"] else "SHORT"

    if orb_dir != direction:
        return None

    buffer         = ATR_BUFFER_PCT * atr_daily
    long_trigger   = orb_high + buffer
    short_trigger  = orb_low  - buffer

    post_open = day_bars[
        (day_bars.index.strftime("%H:%M") > OPEN_BAR) &
        (day_bars.index.strftime("%H:%M") <= EOD_EXIT)
    ]

    entry = stop = target = None
    entry_ts = None

    for ts, bar in post_open.iterrows():
        if direction == "LONG" and float(bar["Close"]) > long_trigger:
            entry    = long_trigger
            stop     = orb_low
            risk     = entry - stop
            target   = entry + risk * RR_RATIO
            entry_ts = ts
            break
        elif direction == "SHORT" and float(bar["Close"]) < short_trigger:
            entry    = short_trigger
            stop     = orb_high
            risk     = stop - entry
            target   = entry - risk * RR_RATIO
            entry_ts = ts
            break

    if entry is None:
        return None

    risk_per_share = abs(entry - stop)
    if risk_per_share == 0:
        return None

    remaining   = post_open[post_open.index > entry_ts]
    exit_price  = None
    exit_reason = "TIME"

    for ts2, bar2 in remaining.iterrows():
        high = float(bar2["High"])
        low  = float(bar2["Low"])

        if direction == "LONG":
            if low <= stop:
                exit_price  = stop
                exit_reason = "STOP"
                break
            if high >= target:
                exit_price  = target
                exit_reason = "TARGET"
                break
        else:
            if high >= stop:
                exit_price  = stop
                exit_reason = "STOP"
                break
            if low <= target:
                exit_price  = target
                exit_reason = "TARGET"
                break

    if exit_price is None:
        last_bar    = remaining.iloc[-1] if not remaining.empty else post_open.iloc[-1]
        exit_price  = float(last_bar["Close"])
        exit_reason = "TIME"

    if direction == "LONG":
        pnl_r = (exit_price - entry) / risk_per_share
    else:
        pnl_r = (entry - exit_price) / risk_per_share

    pnl_pct = pnl_r * RISK_PER_TRADE

    return BacktestTrade(
        date        = str(date.date()),
        ticker      = "",
        direction   = direction,
        entry       = round(entry, 2),
        stop        = round(stop, 2),
        target      = round(target, 2),
        exit_price  = round(exit_price, 2),
        exit_reason = exit_reason,
        pnl_r       = round(pnl_r, 3),
        pnl_pct     = round(pnl_pct, 5),
    )


def compute_metrics(trades: list, initial_equity: float = 10_000) -> BacktestResult:
    if not trades:
        return BacktestResult()

    df          = pd.DataFrame([t.__dict__ for t in trades])
    df["equity"] = initial_equity * (1 + df["pnl_pct"]).cumprod()
    df["peak"]   = df["equity"].cummax()
    df["dd"]     = (df["equity"] - df["peak"]) / df["peak"]

    wins   = df[df["pnl_r"] > 0]
    losses = df[df["pnl_r"] <= 0]

    win_rate      = len(wins) / len(df)
    gross_profit  = wins["pnl_r"].sum()        if len(wins)   > 0 else 0
    gross_loss    = abs(losses["pnl_r"].sum()) if len(losses) > 0 else 1
    profit_factor = gross_profit / gross_loss  if gross_loss  > 0 else 0
    expectancy_r  = df["pnl_r"].mean()

    daily_returns = df.groupby("date")["pnl_pct"].sum()
    sharpe = (
        daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        if daily_returns.std() > 0 else 0
    )

    total_return = (df["equity"].iloc[-1] - initial_equity) / initial_equity

    return BacktestResult(
        trades         = trades,
        total_return   = round(total_return * 100, 2),
        sharpe_ratio   = round(sharpe, 2),
        max_drawdown   = round(df["dd"].min() * 100, 2),
        win_rate       = round(win_rate * 100, 2),
        profit_factor  = round(profit_factor, 2),
        expectancy_r   = round(expectancy_r, 3),
        total_trades   = len(df),
        winning_trades = len(wins),
        losing_trades  = len(losses),
    )


def plot_equity_curve(trades: list, initial_equity: float, output_dir: str = "logs"):
    """Save equity curve + drawdown chart as a PNG."""
    if not trades:
        return

    Path(output_dir).mkdir(exist_ok=True)

    df           = pd.DataFrame([t.__dict__ for t in trades])
    df["date"]   = pd.to_datetime(df["date"])
    df           = df.sort_values("date").reset_index(drop=True)
    df["equity"] = initial_equity * (1 + df["pnl_pct"]).cumprod()
    df["peak"]   = df["equity"].cummax()
    df["dd"]     = (df["equity"] - df["peak"]) / df["peak"] * 100

    # Color each point by outcome
    colors = df["exit_reason"].map({"TARGET": "#2ecc71", "STOP": "#e74c3c", "TIME": "#95a5a6"})

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor="#0f0f0f",
    )

    fig.suptitle(
        "Nova — Equity Curve & Drawdown",
        fontsize=15, fontweight="bold",
        color="white", y=0.98,
    )

    # ── Equity curve ───────────────────────────────────────
    ax1.set_facecolor("#0f0f0f")
    ax1.plot(df.index, df["equity"],
             color="#00d4ff", linewidth=1.6, zorder=2, label="Nova equity")
    ax1.fill_between(df.index, initial_equity, df["equity"],
                     where=df["equity"] >= initial_equity,
                     alpha=0.15, color="#2ecc71")
    ax1.fill_between(df.index, initial_equity, df["equity"],
                     where=df["equity"] < initial_equity,
                     alpha=0.15, color="#e74c3c")

    ax1.scatter(df.index, df["equity"], c=colors, s=18, zorder=3, alpha=0.8)
    ax1.axhline(initial_equity, color="#555", linewidth=0.8, linestyle="--")

    ax1.set_ylabel("Account Value ($)", color="white", fontsize=11)
    ax1.tick_params(colors="white")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71",
               markersize=8, label="Target hit"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markersize=8, label="Stop hit"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#95a5a6",
               markersize=8, label="Time exit"),
    ]
    ax1.legend(handles=legend_elements, facecolor="#1a1a1a",
               edgecolor="#333", labelcolor="white", fontsize=9)

    # Annotate final equity
    final = df["equity"].iloc[-1]
    ret   = (final - initial_equity) / initial_equity * 100
    ax1.annotate(
        f"  ${final:,.0f}  ({ret:+.1f}%)",
        xy=(df.index[-1], final),
        color="#00d4ff", fontsize=10, fontweight="bold",
    )

    # ── Drawdown ────────────────────────────────────────────
    ax2.set_facecolor("#0f0f0f")
    ax2.fill_between(df.index, df["dd"], 0, alpha=0.5, color="#e74c3c")
    ax2.plot(df.index, df["dd"], color="#e74c3c", linewidth=1.2)
    ax2.axhline(0, color="#555", linewidth=0.6)

    ax2.set_ylabel("Drawdown (%)", color="white", fontsize=10)
    ax2.set_xlabel("Trade #", color="white", fontsize=10)
    ax2.tick_params(colors="white")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()
    out_path = f"{output_dir}/nova_equity_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close()
    logger.success(f"Equity curve saved → {out_path}")


def run_backtest(
    watchlist      : list  = None,
    qqq_ticker     : str   = "QQQ",
    initial_equity : float = 10_000,
) -> BacktestResult:

    if watchlist is None:
        watchlist = ["META", "NVDA", "TSLA", "AAPL", "MSFT",
                     "AMZN", "AMD",  "GOOGL","CRWD", "PLTR"]

    all_tickers = [qqq_ticker] + watchlist

    logger.info("━" * 55)
    logger.info("  NOVA BACKTEST ENGINE")
    logger.info(f"  Tickers : {watchlist}")
    logger.info(f"  Equity  : ${initial_equity:,.0f}")
    logger.info("━" * 55)

    data = fetch_data_bulk(all_tickers, period="2y")

    if qqq_ticker not in data:
        logger.error("QQQ data missing — cannot run backtest")
        return BacktestResult()

    qqq_intraday  = data[qqq_ticker]["intraday"]
    trading_days  = sorted(set(qqq_intraday.index.date))
    logger.info(f"Backtesting {len(trading_days)} trading days...")

    all_trades = []

    for day in trading_days:
        day_ts = pd.Timestamp(day, tz="UTC")

        bias = get_bias_for_day(qqq_intraday, day_ts)
        if bias == "NO_TRADE":
            continue

        for ticker in watchlist:
            if ticker not in data:
                continue

            daily    = data[ticker]["daily"]
            intraday = data[ticker]["intraday"]

            rvol = get_rvol_for_day(daily, day_ts)
            if rvol < MIN_RVOL:
                continue

            gap = get_gap_for_day(daily, day_ts)
            if abs(gap) < MIN_GAP_PCT:
                continue

            date_naive = day_ts.tz_localize(None)
            past_daily = daily[daily.index < date_naive]

            if len(past_daily) < 15:
                continue

            close      = past_daily["Close"]
            prev_close = close.shift(1)
            high       = past_daily["High"]
            low        = past_daily["Low"]

            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low  - prev_close).abs(),
            ], axis=1).max(axis=1)

            atr_daily = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])

            trade = simulate_trade(intraday, day_ts, bias, atr_daily)

            if trade is None:
                continue

            trade.ticker = ticker
            all_trades.append(trade)

    logger.info(f"Backtest complete | {len(all_trades)} trades simulated")

    result = compute_metrics(all_trades, initial_equity)

    logger.info("\n" + "━" * 55)
    logger.info("  NOVA BACKTEST RESULTS")
    logger.info("━" * 55)
    logger.info(f"  Total trades    : {result.total_trades}")
    logger.info(f"  Win rate        : {result.win_rate}%")
    logger.info(f"  Winning trades  : {result.winning_trades}")
    logger.info(f"  Losing trades   : {result.losing_trades}")
    logger.info(f"  Expectancy      : {result.expectancy_r:+.3f}R per trade")
    logger.info(f"  Profit factor   : {result.profit_factor}")
    logger.info(f"  Sharpe ratio    : {result.sharpe_ratio}")
    logger.info(f"  Max drawdown    : {result.max_drawdown}%")
    logger.info(f"  Total return    : {result.total_return:+.2f}%")
    logger.info("━" * 55)

    # Save equity curve
    plot_equity_curve(all_trades, initial_equity)

    # Trade log
    if all_trades:
        df_log = pd.DataFrame([t.__dict__ for t in all_trades])
        logger.info(f"\n  Full trade log:\n{df_log.to_string(index=False)}")

    return result


if __name__ == "__main__":
    result = run_backtest(initial_equity=10_000)

    print(f"\n{'━'*55}")
    print(f"  NOVA — BACKTEST SUMMARY")
    print(f"{'━'*55}")
    print(f"  Trades       : {result.total_trades}")
    print(f"  Win rate     : {result.win_rate}%")
    print(f"  Expectancy   : {result.expectancy_r:+.3f}R")
    print(f"  Profit factor: {result.profit_factor}")
    print(f"  Sharpe ratio : {result.sharpe_ratio}")
    print(f"  Max drawdown : {result.max_drawdown}%")
    print(f"  Total return : {result.total_return:+.2f}%")
    print(f"{'━'*55}")
    print(f"\n  Equity curve → logs/nova_equity_curve.png\n")