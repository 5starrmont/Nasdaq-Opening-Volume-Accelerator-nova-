"""
utils/indicators.py
Nova — technical indicators
All indicators used across the Nova system live here.
"""

import pandas as pd
import numpy as np
from loguru import logger


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Compute intraday VWAP (Volume Weighted Average Price).

    VWAP resets every trading day — it is meaningless on daily bars.
    Only use this on intraday data (1m, 5m, 15m).

    Formula: VWAP = cumulative(typical_price × volume) / cumulative(volume)
    Typical price = (High + Low + Close) / 3

    Args:
        df: intraday OHLCV DataFrame with DatetimeIndex

    Returns:
        pd.Series of VWAP values, one per bar
    """
    # Flatten multi-level columns if yfinance returned them
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(level=1, axis=1)

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_x_vol      = typical_price * df["Volume"]

    vwap = (
        tp_x_vol
        .groupby(df.index.date)
        .cumsum()
        /
        df["Volume"]
        .groupby(df.index.date)
        .cumsum()
    )

    logger.info(f"VWAP computed | {len(vwap)} bars | "
                f"{df.index[0].date()} → {df.index[-1].date()}")
    return vwap


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Compute Exponential Moving Average.

    Args:
        series: typically the Close price series
        period: number of bars (e.g. 9 for 9-bar EMA)

    Returns:
        pd.Series of EMA values
    """
    return series.ewm(span=period, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average True Range — measures volatility.
    Used to size our entry buffer above the ORB high.

    True Range = max of:
        High - Low
        |High - Previous Close|
        |Low  - Previous Close|

    Args:
        df: OHLCV DataFrame
        period: lookback period (default 14)

    Returns:
        pd.Series of ATR values
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(level=1, axis=1)

    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    return atr


def compute_rvol(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Compute Relative Volume — today's volume vs the N-day average.

    RVOL = today's volume / average daily volume over last N days
    RVOL >= 2.0 means twice the normal volume — a Stock in Play.

    Args:
        df: daily OHLCV DataFrame
        lookback: number of days for the average (default 20)

    Returns:
        pd.Series of RVOL values
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(level=1, axis=1)

    avg_volume = df["Volume"].rolling(window=lookback).mean()
    rvol       = df["Volume"] / avg_volume

    logger.info(f"RVOL computed | lookback={lookback} days | latest={rvol.iloc[-1]:.2f}x")
    return rvol


if __name__ == "__main__":
    from data.fetcher import fetch_daily, fetch_intraday

    # --- Test VWAP on intraday data ---
    df_5m = fetch_intraday("QQQ", period="5d", interval="5m")
    vwap  = compute_vwap(df_5m)

    print("\n--- VWAP (last 10 bars) ---")
    print(vwap.tail(10).round(2))

    # --- Test RVOL on daily data ---
    df_daily = fetch_daily("QQQ", period="3mo")
    rvol     = compute_rvol(df_daily)

    print("\n--- RVOL (last 10 days) ---")
    print(rvol.tail(10).round(2))

    # --- Test ATR on daily data ---
    atr = compute_atr(df_daily)
    print("\n--- ATR (last 5 days) ---")
    print(atr.tail(5).round(2))