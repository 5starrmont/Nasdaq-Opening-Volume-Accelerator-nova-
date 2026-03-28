"""
strategy/scanner.py
Nova — Stocks in Play scanner (Layer 2)

Scans a watchlist of Nasdaq stocks and ranks them by Relative Volume.
A stock is "In Play" when unusual volume signals a fundamental catalyst
— earnings, news, analyst action — that concentrates institutional flow.

Criteria (from Zarattini et al. 2024):
    RVOL  >= 2.0x   (twice normal volume)
    Price >= $10     (no penny stocks)
    Market cap > $500M (liquidity)
    Gap   >= 0.5%   (pre-market catalyst present)
"""

import pandas as pd
import yfinance as yf
from loguru import logger

from data.fetcher import fetch_daily
from utils.indicators import compute_rvol


# --- Default Nasdaq watchlist ---
# These are the most liquid, most traded Nasdaq 100 names.
# In production this gets replaced by a live pre-market scanner.
NASDAQ_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "TSLA", "GOOGL", "AVGO", "COST", "NFLX",
    "AMD",  "ADBE", "QCOM", "INTC", "MU",
    "PANW", "CRWD", "SNOW", "MSTR", "PLTR",
]

# --- Thresholds ---
MIN_RVOL      = 2.0    # minimum relative volume to qualify
MIN_PRICE     = 10.0   # minimum share price
MIN_GAP_PCT   = 0.005  # minimum 0.5% gap from prior close
TOP_N         = 5      # max stocks to trade per day


def get_gap_pct(ticker: str) -> float:
    """
    Compute today's pre-market gap for a ticker.

    Gap % = (today's open - yesterday's close) / yesterday's close

    Args:
        ticker: stock symbol

    Returns:
        gap as a decimal (0.02 = 2% gap up, -0.015 = 1.5% gap down)
    """
    try:
        df = fetch_daily(ticker, period="5d")

        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(level=1, axis=1)

        if len(df) < 2:
            return 0.0

        prev_close  = float(df["Close"].iloc[-2])
        todays_open = float(df["Open"].iloc[-1])
        gap         = (todays_open - prev_close) / prev_close
        return round(gap, 5)

    except Exception as e:
        logger.warning(f"{ticker} gap calculation failed: {e}")
        return 0.0


def scan_stocks_in_play(
    watchlist: list = NASDAQ_WATCHLIST,
    min_rvol: float = MIN_RVOL,
    min_price: float = MIN_PRICE,
    min_gap_pct: float = MIN_GAP_PCT,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """
    Scan watchlist and return ranked Stocks in Play.

    Args:
        watchlist   : list of ticker symbols to scan
        min_rvol    : minimum relative volume threshold
        min_price   : minimum share price filter
        min_gap_pct : minimum absolute gap % from prior close
        top_n       : max number of stocks to return

    Returns:
        DataFrame ranked by RVOL with columns:
        ticker, price, gap_pct, rvol, qualifies
    """
    logger.info(f"Scanning {len(watchlist)} stocks for In Play criteria...")
    results = []

    for ticker in watchlist:
        try:
            # Fetch daily data
            df = fetch_daily(ticker, period="3mo")

            if df.empty or len(df) < 22:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(level=1, axis=1)

            # Current price and gap
            price   = float(df["Close"].iloc[-1])
            gap_pct = get_gap_pct(ticker)
            rvol    = compute_rvol(df, lookback=20).iloc[-1]

            # Apply filters
            qualifies = (
                rvol      >= min_rvol      and
                price     >= min_price     and
                abs(gap_pct) >= min_gap_pct
            )

            results.append({
                "ticker"    : ticker,
                "price"     : round(price, 2),
                "gap_pct"   : round(gap_pct * 100, 2),
                "rvol"      : round(rvol, 2),
                "qualifies" : qualifies,
            })

        except Exception as e:
            logger.warning(f"Failed to scan {ticker}: {e}")
            continue

    if not results:
        logger.error("No results from scan — check data connection")
        return pd.DataFrame()

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("rvol", ascending=False)

    # Stocks that pass all filters
    in_play = df_results[df_results["qualifies"]].head(top_n)

    logger.info("=" * 55)
    logger.info("  NOVA SCAN RESULTS — ALL STOCKS")
    logger.info("=" * 55)
    logger.info(f"\n{df_results.to_string(index=False)}")
    logger.info("=" * 55)
    logger.info(f"  STOCKS IN PLAY (top {top_n})")
    logger.info("=" * 55)

    if in_play.empty:
        logger.warning("  No stocks passed all filters today — no trades")
    else:
        logger.info(f"\n{in_play.to_string(index=False)}")

    return in_play


if __name__ == "__main__":
    in_play = scan_stocks_in_play()

    print(f"\n{'='*55}")
    print(f"  STOCKS IN PLAY TODAY")
    print(f"{'='*55}")

    if in_play.empty:
        print("  None — Nova sits out today.")
    else:
        for _, row in in_play.iterrows():
            direction = "GAP UP" if row["gap_pct"] > 0 else "GAP DOWN"
            print(f"  {row['ticker']:<6} | "
                  f"${row['price']:<8} | "
                  f"RVOL {row['rvol']}x | "
                  f"{direction} {abs(row['gap_pct'])}%")
    print(f"{'='*55}\n")