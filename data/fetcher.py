"""
data/fetcher.py
Nova — market data ingestion
Fetches OHLCV data from Yahoo Finance via yfinance
"""

import yfinance as yf
import pandas as pd
from loguru import logger


def fetch_daily(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch daily OHLCV bars for a ticker.

    Args:
        ticker: e.g. "QQQ", "AAPL"
        period: how far back — "1y", "2y", "5y"

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    logger.info(f"Fetching daily data for {ticker} | period={period}")

    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)

    if df.empty:
        logger.error(f"No data returned for {ticker}")
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index)
    logger.success(f"{ticker} | {len(df)} daily bars fetched | "
                   f"{df.index[0].date()} → {df.index[-1].date()}")
    return df


def fetch_intraday(ticker: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    """
    Fetch intraday OHLCV bars for a ticker.

    Args:
        ticker: e.g. "QQQ"
        period: max 60d for 1m, max 730d for 1h
        interval: "1m", "5m", "15m", "1h"

    Returns:
        DataFrame with intraday OHLCV bars
    """
    logger.info(f"Fetching {interval} intraday data for {ticker} | period={period}")

    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    if df.empty:
        logger.error(f"No data returned for {ticker}")
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index)
    logger.success(f"{ticker} | {len(df)} {interval} bars fetched | "
                   f"{df.index[0]} → {df.index[-1]}")
    return df


if __name__ == "__main__":
    # Quick test — run this file directly to verify data is flowing
    df_daily = fetch_daily("QQQ", period="1y")
    print("\n--- Daily bars (last 5) ---")
    print(df_daily.tail())

    df_intraday = fetch_intraday("QQQ", period="5d", interval="5m")
    print("\n--- 5-min intraday bars (last 5) ---")
    print(df_intraday.tail())