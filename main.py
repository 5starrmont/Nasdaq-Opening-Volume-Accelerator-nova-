"""
main.py
Nova — Nasdaq Opening Volume Accelerator

Morning routine orchestrator. Runs the full 3-layer pipeline:
    Layer 1 → Market bias    (strategy/bias.py)
    Layer 2 → Stocks in Play (strategy/scanner.py)
    Layer 3 → ORB signals    (strategy/orb.py)
              Risk approval  (risk/manager.py)

Usage:
    python main.py                        # live mode, default $10k account
    python main.py --equity 25000         # custom account size
    python main.py --ticker META --bias SHORT  # force a specific test
"""

import argparse
from loguru import logger

from strategy.bias    import get_daily_bias
from strategy.scanner import scan_stocks_in_play
from strategy.orb     import detect_breakout
from risk.manager     import AccountState, approve_signal
from data.fetcher     import fetch_daily
from utils.indicators import compute_atr


# ── logging setup ──────────────────────────────────────────
logger.remove()  # remove default handler
logger.add(
    "logs/nova_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    format="{time:HH:mm:ss} | {level:<8} | {message}",
)
logger.add(
    lambda msg: print(msg, end=""),
    level="INFO",
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
)


def run_nova(equity: float = 10_000.0):
    """
    Run Nova's complete morning pipeline.

    Args:
        equity: starting account equity in dollars
    """

    logger.info("━" * 55)
    logger.info("  NOVA — Nasdaq Opening Volume Accelerator")
    logger.info("  Starting morning pipeline...")
    logger.info("━" * 55)

    # ── Initialize account ──────────────────────────────────
    account = AccountState(
        equity         = equity,
        peak_equity    = equity,
        daily_pnl      = 0.0,
        open_positions = 0,
    )
    logger.info(f"Account initialized | Equity=${equity:,.2f}")

    # ── Layer 1: Market bias ────────────────────────────────
    logger.info("\n[ LAYER 1 ] Computing market bias...")
    bias_result = get_daily_bias("QQQ")
    bias        = bias_result["bias"]

    if bias == "NO_TRADE":
        logger.warning("Bias is NO_TRADE — QQQ too close to VWAP.")
        logger.warning("Nova sits out today. No trades.")
        return

    logger.info(f"Bias confirmed: {bias} | "
                f"QQQ ${bias_result['price']} vs "
                f"VWAP ${bias_result['vwap']} "
                f"({bias_result['pct_diff']:+}%)")

    # ── Layer 2: Stocks in Play ─────────────────────────────
    logger.info("\n[ LAYER 2 ] Scanning for Stocks in Play...")
    in_play = scan_stocks_in_play()

    if in_play.empty:
        logger.warning("No Stocks in Play today — no trades.")
        return

    logger.info(f"{len(in_play)} stock(s) in play: "
                f"{list(in_play['ticker'])}")

    # ── Layer 3: ORB signals + risk approval ────────────────
    logger.info("\n[ LAYER 3 ] Scanning for ORB breakouts...")

    approved_trades = []

    for _, row in in_play.iterrows():
        ticker = row["ticker"]

        if account.open_positions >= 3:
            logger.warning("Max positions reached — stopping scan")
            break

        # Get daily ATR for this stock
        df_daily = fetch_daily(ticker, period="3mo")
        if df_daily.empty:
            continue

        if hasattr(df_daily.columns, "levels"):
            df_daily = df_daily.droplevel(level=1, axis=1)

        atr_daily = float(compute_atr(df_daily).iloc[-1])

        # Detect breakout
        signal = detect_breakout(ticker, bias, atr_daily)

        if signal is None:
            continue

        # Risk approval
        sized = approve_signal(signal, account)

        if sized is None:
            continue

        # Update account state
        account.open_positions  += 1
        account.daily_risk_used += sized.dollar_risk
        account.trades_today    += 1
        approved_trades.append(sized)

    # ── Summary ─────────────────────────────────────────────
    logger.info("\n" + "━" * 55)
    logger.info("  NOVA DAILY SUMMARY")
    logger.info("━" * 55)
    logger.info(f"  Market bias   : {bias}")
    logger.info(f"  Stocks scanned: {len(in_play)}")
    logger.info(f"  Signals found : {len(approved_trades)}")

    if not approved_trades:
        logger.info("  No trades approved today — Nova waits.")
    else:
        logger.info(f"\n  {'TICKER':<8} {'DIR':<6} "
                    f"{'ENTRY':>8} {'STOP':>8} "
                    f"{'TARGET':>8} {'SHARES':>7} "
                    f"{'RISK$':>8} {'REWARD$':>9}")
        logger.info("  " + "-" * 68)

        total_risk   = 0.0
        total_reward = 0.0

        for s in approved_trades:
            sig = s.signal
            logger.info(
                f"  {sig.ticker:<8} {sig.direction:<6} "
                f"{sig.entry:>8.2f} {sig.stop:>8.2f} "
                f"{sig.target:>8.2f} {s.shares:>7} "
                f"{s.dollar_risk:>8.2f} {s.dollar_target:>9.2f}"
            )
            total_risk   += s.dollar_risk
            total_reward += s.dollar_target

        logger.info("  " + "-" * 68)
        logger.info(f"  {'TOTAL':<8} {'':6} {'':>8} {'':>8} "
                    f"{'':>8} {'':>7} "
                    f"{total_risk:>8.2f} {total_reward:>9.2f}")
        logger.info(f"\n  Total account risk today: "
                    f"{total_risk/account.equity*100:.2f}%")

    logger.info("━" * 55)
    return approved_trades


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nova Trading System")
    parser.add_argument(
        "--equity",
        type=float,
        default=10_000.0,
        help="Account equity in dollars (default: 10000)",
    )
    args = parser.parse_args()

    run_nova(equity=args.equity)