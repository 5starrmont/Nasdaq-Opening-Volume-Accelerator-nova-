"""
risk/manager.py
Nova — position sizing and risk controls

Every trade goes through this module before any order is placed.
No signal from strategy/orb.py is acted on without risk approval.

Rules (from edge research):
    Risk per trade  : 1% of account equity
    Max positions   : 3 simultaneous trades
    Max daily risk  : 2% of account
    Daily loss limit: -2% of account → stop trading for the day
    Drawdown limit  : -10% from peak → halve position size
                      -20% from peak → stop trading entirely
"""

from dataclasses import dataclass, field
from loguru import logger
from strategy.orb import TradeSignal


# --- Risk parameters ---
RISK_PER_TRADE_PCT  = 0.01   # 1% of account per trade
MAX_POSITIONS       = 3      # max open trades at once
MAX_DAILY_RISK_PCT  = 0.02   # 2% total account at risk simultaneously
DAILY_LOSS_LIMIT    = -0.02  # -2% daily P&L → halt trading
DRAWDOWN_HALF_PCT   = -0.10  # -10% from peak → cut size in half
DRAWDOWN_STOP_PCT   = -0.20  # -20% from peak → stop entirely
MIN_SHARES          = 1      # minimum shares per trade


@dataclass
class AccountState:
    """
    Tracks the current state of the trading account.
    Updated after every trade and every session.
    """
    equity          : float          # current account value in dollars
    peak_equity     : float          # highest account value ever reached
    daily_pnl       : float = 0.0   # today's realized P&L
    open_positions  : int   = 0      # number of currently open trades
    daily_risk_used : float = 0.0   # total risk committed today
    trades_today    : int   = 0      # number of trades taken today
    halted          : bool  = False  # True = no new trades allowed

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak equity as a decimal."""
        return (self.equity - self.peak_equity) / self.peak_equity

    @property
    def daily_pnl_pct(self) -> float:
        """Today's P&L as a percentage of starting equity."""
        starting = self.equity - self.daily_pnl
        if starting == 0:
            return 0.0
        return self.daily_pnl / starting


@dataclass
class SizedSignal:
    """
    A TradeSignal that has been approved and sized by the risk module.
    This is what gets sent to the execution engine.
    """
    signal          : TradeSignal
    shares          : int     # number of shares to trade
    dollar_risk     : float   # total dollar risk on this trade
    dollar_target   : float   # total dollar profit if target hit
    account_risk_pct: float   # risk as % of account


def compute_position_size(
    signal  : TradeSignal,
    account : AccountState,
) -> int:
    """
    Compute how many shares to trade based on 1% account risk rule.

    Formula:
        dollar_risk_allowed = account.equity × RISK_PER_TRADE_PCT
        shares = dollar_risk_allowed / risk_per_share

    Args:
        signal  : validated TradeSignal from ORB engine
        account : current AccountState

    Returns:
        integer number of shares (never fractional, never negative)
    """
    dollar_risk_allowed = account.equity * RISK_PER_TRADE_PCT
    risk_per_share      = signal.risk  # already computed in TradeSignal

    if risk_per_share <= 0:
        logger.error(f"Invalid risk per share: {risk_per_share}")
        return 0

    raw_shares = dollar_risk_allowed / risk_per_share
    shares     = max(MIN_SHARES, int(raw_shares))  # floor to whole shares

    # Apply drawdown-based size reduction
    if account.drawdown <= DRAWDOWN_HALF_PCT:
        shares = max(MIN_SHARES, shares // 2)
        logger.warning(f"Drawdown {account.drawdown*100:.1f}% — "
                       f"position size halved to {shares} shares")

    logger.info(f"Position size | "
                f"Equity=${account.equity:,.0f} | "
                f"Risk allowed=${dollar_risk_allowed:.0f} | "
                f"Risk/share=${risk_per_share:.2f} | "
                f"Shares={shares}")
    return shares


def approve_signal(
    signal  : TradeSignal,
    account : AccountState,
) -> SizedSignal | None:
    """
    Run all risk checks and size the position.
    Returns a SizedSignal if approved, None if rejected.

    Checks (in order):
        1. Account not halted
        2. Daily loss limit not breached
        3. Drawdown not at stop level
        4. Max positions not exceeded
        5. Daily risk budget not exceeded
        6. Signal has valid R:R

    Args:
        signal  : TradeSignal from ORB engine
        account : current AccountState

    Returns:
        SizedSignal if all checks pass, None if any check fails
    """
    ticker = signal.ticker
    logger.info(f"Risk check for {ticker} {signal.direction}...")

    # --- Check 1: Account halted ---
    if account.halted:
        logger.warning(f"{ticker} REJECTED — account is halted for today")
        return None

    # --- Check 2: Daily loss limit ---
    if account.daily_pnl_pct <= DAILY_LOSS_LIMIT:
        logger.warning(f"{ticker} REJECTED — daily loss limit hit "
                       f"({account.daily_pnl_pct*100:.2f}%). "
                       f"No more trades today.")
        account.halted = True
        return None

    # --- Check 3: Drawdown stop ---
    if account.drawdown <= DRAWDOWN_STOP_PCT:
        logger.warning(f"{ticker} REJECTED — max drawdown hit "
                       f"({account.drawdown*100:.1f}%). "
                       f"Stop trading and review system.")
        account.halted = True
        return None

    # --- Check 4: Max positions ---
    if account.open_positions >= MAX_POSITIONS:
        logger.warning(f"{ticker} REJECTED — max positions reached "
                       f"({account.open_positions}/{MAX_POSITIONS})")
        return None

    # --- Check 5: Daily risk budget ---
    trade_risk    = account.equity * RISK_PER_TRADE_PCT
    new_daily_risk = account.daily_risk_used + trade_risk

    if new_daily_risk / account.equity > MAX_DAILY_RISK_PCT:
        logger.warning(f"{ticker} REJECTED — daily risk budget exceeded "
                       f"(used {account.daily_risk_used/account.equity*100:.1f}% "
                       f"of {MAX_DAILY_RISK_PCT*100:.0f}% limit)")
        return None

    # --- Check 6: Minimum R:R ---
    if signal.rr_ratio < 2.0:
        logger.warning(f"{ticker} REJECTED — R:R {signal.rr_ratio} below 2.0 minimum")
        return None

    # --- All checks passed — size the position ---
    shares       = compute_position_size(signal, account)
    dollar_risk  = round(shares * signal.risk, 2)
    dollar_target = round(shares * signal.reward, 2)
    acct_risk_pct = round(dollar_risk / account.equity * 100, 3)

    sized = SizedSignal(
        signal           = signal,
        shares           = shares,
        dollar_risk      = dollar_risk,
        dollar_target    = dollar_target,
        account_risk_pct = acct_risk_pct,
    )

    logger.info("=" * 55)
    logger.info(f"  RISK APPROVED — {ticker} {signal.direction}")
    logger.info(f"  Shares        : {shares}")
    logger.info(f"  Dollar risk   : ${dollar_risk:,.2f}")
    logger.info(f"  Dollar target : ${dollar_target:,.2f}")
    logger.info(f"  Account risk  : {acct_risk_pct}%")
    logger.info(f"  Open positions: {account.open_positions + 1}/{MAX_POSITIONS}")
    logger.info("=" * 55)

    return sized


if __name__ == "__main__":
    from strategy.orb import TradeSignal

    # Simulate the META SHORT signal from Step 10
    test_signal = TradeSignal(
        ticker     = "META",
        direction  = "SHORT",
        entry      = 535.65,
        stop       = 543.60,
        target     = 519.75,
        risk       = 7.95,
        reward     = 15.90,
        rr_ratio   = 2.0,
        orb_high   = 543.60,
        orb_low    = 537.79,
        signal_bar = "2026-03-27 13:55:00+00:00",
        bias       = "SHORT",
    )

    # Simulate a $10,000 account — realistic starting point
    account = AccountState(
        equity         = 10_000.0,
        peak_equity    = 10_000.0,
        daily_pnl      = 0.0,
        open_positions = 0,
    )

    print(f"\n{'='*55}")
    print(f"  ACCOUNT STATE")
    print(f"{'='*55}")
    print(f"  Equity     : ${account.equity:,.2f}")
    print(f"  Peak       : ${account.peak_equity:,.2f}")
    print(f"  Drawdown   : {account.drawdown*100:.1f}%")
    print(f"  Daily P&L  : ${account.daily_pnl:,.2f}")
    print(f"  Positions  : {account.open_positions}/{MAX_POSITIONS}")
    print(f"{'='*55}\n")

    sized = approve_signal(test_signal, account)

    if sized:
        print(f"\n{'='*55}")
        print(f"  SIZED TRADE — READY FOR EXECUTION")
        print(f"{'='*55}")
        print(f"  {sized.signal.ticker} {sized.signal.direction}")
        print(f"  Shares   : {sized.shares}")
        print(f"  Entry    : ${sized.signal.entry}")
        print(f"  Stop     : ${sized.signal.stop}")
        print(f"  Target   : ${sized.signal.target}")
        print(f"  Risk     : ${sized.dollar_risk:,.2f} ({sized.account_risk_pct}% of account)")
        print(f"  Reward   : ${sized.dollar_target:,.2f}")
        print(f"{'='*55}\n")
    else:
        print("\n  Signal rejected by risk manager.\n")