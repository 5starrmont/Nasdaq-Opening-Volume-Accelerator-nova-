"""
strategy/orb.py
Nova — Opening Range Breakout engine (Layer 3)

For each Stock in Play, this module:
    1. Identifies the 5-minute opening range (9:30–9:35 AM ET)
    2. Detects when price breaks above/below that range
    3. Generates a full trade signal: entry, stop, target, R:R

Only fires signals that agree with the daily bias from Layer 1.
"""

import pandas as pd
from loguru import logger
from dataclasses import dataclass

from data.fetcher import fetch_intraday
from utils.indicators import compute_vwap, compute_atr


# --- Constants ---
OPEN_BAR_UTC   = "13:30"    # 9:30 AM ET = 13:30 UTC (first 5-min bar)
ATR_BUFFER_PCT = 0.10       # entry buffer = 10% of daily ATR above ORB high
RR_RATIO       = 2.0        # reward-to-risk ratio for profit target


@dataclass
class TradeSignal:
    """
    A complete, actionable trade signal from Nova.
    Every field must be populated before a signal is valid.
    """
    ticker      : str
    direction   : str    # "LONG" or "SHORT"
    entry       : float  # price to enter the trade
    stop        : float  # price where we are wrong — exit immediately
    target      : float  # price to take profit
    risk        : float  # dollar risk per share (entry - stop)
    reward      : float  # dollar reward per share (target - entry)
    rr_ratio    : float  # reward / risk — must be >= 2.0
    orb_high    : float  # top of the opening range
    orb_low     : float  # bottom of the opening range
    signal_bar  : str    # timestamp of the bar that triggered the signal
    bias        : str    # the market bias that approved this signal


def get_opening_range(df: pd.DataFrame) -> dict:
    """
    Extract the high and low of the first 5-minute candle.

    Args:
        df: intraday DataFrame with UTC timestamps, already fetched

    Returns:
        dict with orb_high, orb_low, orb_open, orb_close, orb_direction
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(level=1, axis=1)

    # Filter to only the opening bar (13:30 UTC = 9:30 AM ET)
    df_utc   = df.copy()
    df_utc.index = df_utc.index.tz_convert("UTC")
    open_bars = df_utc[df_utc.index.strftime("%H:%M") == OPEN_BAR_UTC]

    if open_bars.empty:
        logger.warning("Opening bar not found — market may be closed")
        return {}

    bar = open_bars.iloc[0]

    orb = {
        "orb_high"      : round(float(bar["High"]),  2),
        "orb_low"       : round(float(bar["Low"]),   2),
        "orb_open"      : round(float(bar["Open"]),  2),
        "orb_close"     : round(float(bar["Close"]), 2),
        "orb_direction" : "LONG" if bar["Close"] >= bar["Open"] else "SHORT",
        "orb_range"     : round(float(bar["High"] - bar["Low"]), 2),
    }

    logger.info(f"Opening range | High={orb['orb_high']} "
                f"Low={orb['orb_low']} "
                f"Range=${orb['orb_range']} "
                f"Direction={orb['orb_direction']}")
    return orb


def detect_breakout(
    ticker: str,
    bias: str,
    atr_daily: float,
) -> TradeSignal | None:
    """
    Watch intraday bars for an ORB breakout and generate a trade signal.

    Args:
        ticker     : stock symbol e.g. "META"
        bias       : "LONG" or "SHORT" from Layer 1
        atr_daily  : daily ATR for this stock (for entry buffer)

    Returns:
        TradeSignal if breakout detected, None if no signal today
    """
    if bias not in ("LONG", "SHORT"):
        logger.warning(f"Invalid bias '{bias}' — skipping {ticker}")
        return None

    logger.info(f"Scanning {ticker} for {bias} ORB breakout...")

    # Fetch today's 5-min bars
    df = fetch_intraday(ticker, period="1d", interval="5m")

    if df.empty:
        logger.warning(f"No intraday data for {ticker}")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(level=1, axis=1)

    # Get the opening range
    orb = get_opening_range(df)
    if not orb:
        return None

    # Only trade in the direction of both ORB and bias
    if orb["orb_direction"] != bias:
        logger.info(f"{ticker} | ORB direction={orb['orb_direction']} "
                    f"disagrees with bias={bias} — skipping")
        return None

    # Entry buffer — small cushion above/below ORB to avoid false breaks
    buffer = round(ATR_BUFFER_PCT * atr_daily, 2)

    # Define entry triggers
    long_entry_trigger  = orb["orb_high"] + buffer
    short_entry_trigger = orb["orb_low"]  - buffer

    # Scan bars after the opening range for a breakout
    df_utc = df.copy()
    df_utc.index = df_utc.index.tz_convert("UTC")

    # Bars after 9:35 AM ET (13:35 UTC)
    post_open = df_utc[df_utc.index.strftime("%H:%M") > OPEN_BAR_UTC]

    # Also exclude bars after 3:30 PM ET (19:30 UTC) — no late entries
    post_open = post_open[post_open.index.strftime("%H:%M") <= "19:30"]

    signal = None

    for ts, bar in post_open.iterrows():
        if bias == "LONG":
            # A bar closes ABOVE the long entry trigger
            if float(bar["Close"]) > long_entry_trigger:
                entry  = long_entry_trigger
                stop   = orb["orb_low"]
                risk   = round(entry - stop, 2)
                reward = round(risk * RR_RATIO, 2)
                target = round(entry + reward, 2)

                signal = TradeSignal(
                    ticker     = ticker,
                    direction  = "LONG",
                    entry      = round(entry, 2),
                    stop       = stop,
                    target     = target,
                    risk       = risk,
                    reward     = reward,
                    rr_ratio   = RR_RATIO,
                    orb_high   = orb["orb_high"],
                    orb_low    = orb["orb_low"],
                    signal_bar = str(ts),
                    bias       = bias,
                )
                break

        elif bias == "SHORT":
            # A bar closes BELOW the short entry trigger
            if float(bar["Close"]) < short_entry_trigger:
                entry  = short_entry_trigger
                stop   = orb["orb_high"]
                risk   = round(stop - entry, 2)
                reward = round(risk * RR_RATIO, 2)
                target = round(entry - reward, 2)

                signal = TradeSignal(
                    ticker     = ticker,
                    direction  = "SHORT",
                    entry      = round(entry, 2),
                    stop       = stop,
                    target     = target,
                    risk       = risk,
                    reward     = reward,
                    rr_ratio   = RR_RATIO,
                    orb_high   = orb["orb_high"],
                    orb_low    = orb["orb_low"],
                    signal_bar = str(ts),
                    bias       = bias,
                )
                break

    if signal:
        logger.info("=" * 55)
        logger.info(f"  NOVA SIGNAL — {signal.ticker} {signal.direction}")
        logger.info(f"  Entry      : ${signal.entry}")
        logger.info(f"  Stop       : ${signal.stop}  (risk ${signal.risk}/share)")
        logger.info(f"  Target     : ${signal.target} (reward ${signal.reward}/share)")
        logger.info(f"  R:R        : {signal.rr_ratio}:1")
        logger.info(f"  Signal bar : {signal.signal_bar}")
        logger.info("=" * 55)
    else:
        logger.info(f"{ticker} | No breakout detected today")

    return signal


if __name__ == "__main__":
    from data.fetcher import fetch_daily
    from utils.indicators import compute_atr

    # Test with META — highest RVOL from yesterday's scan
    # Use SHORT bias to match yesterday's market direction
    TICKER = "META"
    BIAS   = "SHORT"

    # Get daily ATR for buffer calculation
    df_daily = fetch_daily(TICKER, period="3mo")
    if isinstance(df_daily.columns, pd.MultiIndex):
        df_daily = df_daily.droplevel(level=1, axis=1)

    atr_daily = float(compute_atr(df_daily).iloc[-1])
    logger.info(f"{TICKER} daily ATR: ${atr_daily:.2f}")

    # Run the signal engine
    signal = detect_breakout(TICKER, BIAS, atr_daily)

    if signal:
        print(f"\n{'='*55}")
        print(f"  SIGNAL GENERATED")
        print(f"{'='*55}")
        print(f"  Ticker    : {signal.ticker}")
        print(f"  Direction : {signal.direction}")
        print(f"  Entry     : ${signal.entry}")
        print(f"  Stop      : ${signal.stop}")
        print(f"  Target    : ${signal.target}")
        print(f"  Risk/share: ${signal.risk}")
        print(f"  R:R       : {signal.rr_ratio}:1")
        print(f"  Triggered : {signal.signal_bar}")
        print(f"{'='*55}\n")
    else:
        print(f"\n  No signal on {TICKER} today — Nova waits.\n")