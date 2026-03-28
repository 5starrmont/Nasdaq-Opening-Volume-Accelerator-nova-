"""
strategy/bias.py
Nova — market bias filter (Layer 1)

Determines intraday directional bias by comparing QQQ price to VWAP.
This runs once at 10:00 AM ET and governs all trades for the day.

Rules:
    price > VWAP  →  LONG bias  (only take long ORB signals)
    price < VWAP  →  SHORT bias (only take short ORB signals)
    within 0.05%  →  NO TRADE   (too close to call, sit out)
"""

import pandas as pd
from loguru import logger

from data.fetcher import fetch_intraday
from utils.indicators import compute_vwap


# --- Constants ---
MARKET_OPEN_UTC  = "13:30"   # 9:30 AM Eastern = 13:30 UTC
BIAS_WINDOW_UTC  = "14:00"   # Read bias at 10:00 AM Eastern = 14:00 UTC
NEUTRAL_BAND_PCT = 0.0005    # 0.05% — inside this band = no trade


def get_daily_bias(ticker: str = "QQQ") -> dict:
    """
    Compute today's market bias for the given ticker.

    Fetches today's intraday bars, computes VWAP, then reads the
    price vs VWAP relationship at the bias window time (10:00 AM ET).

    Returns:
        dict with keys:
            bias        : "LONG", "SHORT", or "NO_TRADE"
            price       : QQQ price at bias window
            vwap        : VWAP at bias window
            pct_diff    : % difference between price and VWAP
            timestamp   : the bar timestamp used for the reading
    """
    logger.info(f"Computing daily bias for {ticker}...")

    # Fetch today's 5-min bars
    df = fetch_intraday(ticker, period="1d", interval="5m")

    if df.empty:
        logger.error("No intraday data — cannot determine bias")
        return {"bias": "NO_TRADE", "price": None, "vwap": None,
                "pct_diff": None, "timestamp": None}

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(level=1, axis=1)

    # Compute VWAP
    vwap = compute_vwap(df)
    df["VWAP"] = vwap

    # Find the bar closest to bias window (10:00 AM ET = 14:00 UTC)
    today_str  = df.index[-1].strftime("%Y-%m-%d")
    window_str = f"{today_str} {BIAS_WINDOW_UTC}"

    # Filter bars up to and including the bias window
    df_utc      = df.copy()
    df_utc.index = df_utc.index.tz_convert("UTC")

    bias_bars = df_utc[df_utc.index.strftime("%H:%M") <= BIAS_WINDOW_UTC]

    if bias_bars.empty:
        # Market hasn't reached bias window yet — use latest bar
        logger.warning("Bias window not yet reached — using latest available bar")
        bias_bar = df_utc.iloc[-1]
    else:
        bias_bar = bias_bars.iloc[-1]

    price     = float(bias_bar["Close"])
    vwap_val  = float(bias_bar["VWAP"])
    pct_diff  = (price - vwap_val) / vwap_val
    timestamp = bias_bar.name

    # Determine bias
    if abs(pct_diff) <= NEUTRAL_BAND_PCT:
        bias = "NO_TRADE"
    elif price > vwap_val:
        bias = "LONG"
    else:
        bias = "SHORT"

    result = {
        "bias"      : bias,
        "price"     : round(price, 2),
        "vwap"      : round(vwap_val, 2),
        "pct_diff"  : round(pct_diff * 100, 3),
        "timestamp" : timestamp,
    }

    # Log clearly so you can read it at a glance every morning
    logger.info("=" * 50)
    logger.info(f"  NOVA DAILY BIAS — {ticker}")
    logger.info(f"  Timestamp : {timestamp}")
    logger.info(f"  Price     : ${price:.2f}")
    logger.info(f"  VWAP      : ${vwap_val:.2f}")
    logger.info(f"  Diff      : {pct_diff*100:+.3f}%")
    logger.info(f"  BIAS      : >>> {bias} <<<")
    logger.info("=" * 50)

    return result


if __name__ == "__main__":
    result = get_daily_bias("QQQ")

    print(f"\n{'='*40}")
    print(f"  Bias      : {result['bias']}")
    print(f"  QQQ Price : ${result['price']}")
    print(f"  VWAP      : ${result['vwap']}")
    print(f"  Diff      : {result['pct_diff']:+}%")
    print(f"  Time      : {result['timestamp']}")
    print(f"{'='*40}\n")

    if result["bias"] == "LONG":
        print("Nova is scanning for LONG setups only today.")
    elif result["bias"] == "SHORT":
        print("Nova is scanning for SHORT setups only today.")
    else:
        print("Nova sees no clear bias. No trades today.")