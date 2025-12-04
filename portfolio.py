import logging
from dataclasses import dataclass

from driftpy.constants.numeric_constants import QUOTE_PRECISION
from driftpy.math.margin import MarginCategory

from trading import Account, DriftClientManager

logger = logging.getLogger(__name__)


@dataclass
class Portfolio:
    """Portfolio summary."""
    total_collateral: float  # Total account value in USD
    free_collateral: float   # Available margin in USD
    net_usd_value: float     # Net USD value including unrealized PnL
    margin_requirement: float  # Current margin requirement
    leverage: float          # Current leverage ratio
    health: int              # Account health (0-100)


async def get_portfolio(account: Account) -> Portfolio:
    """
    Get portfolio summary for an account.
    
    Args:
        account: Trading account
    
    Returns:
        Portfolio object with account metrics
    
    Raises:
        ValueError: If Drift account doesn't exist for this wallet
    """
    drift_client = await DriftClientManager.get_client(account)
    user = drift_client.get_user()
    
    # Check if user account exists
    user_account_and_slot = user.account_subscriber.get_user_account_and_slot()
    if user_account_and_slot is None or user_account_and_slot.data is None:
        raise ValueError(
            f"No Drift account found for wallet {account.name}. "
            f"Please go to drift.trade and create a trading account first."
        )
    
    # Get collateral values (in QUOTE_PRECISION = 1e6)
    total_collateral = user.get_total_collateral(MarginCategory.INITIAL)
    free_collateral = user.get_free_collateral(MarginCategory.INITIAL)
    net_usd_value = user.get_net_usd_value()
    
    # Get margin requirement
    margin_requirement = user.get_margin_requirement(MarginCategory.INITIAL)
    
    # Get leverage (returned as basis points, so divide by 10000)
    leverage_bps = user.get_leverage()
    leverage = leverage_bps / 10000
    
    # Get account health (0-100)
    health = user.get_health()
    
    return Portfolio(
        total_collateral=total_collateral / QUOTE_PRECISION,
        free_collateral=free_collateral / QUOTE_PRECISION,
        net_usd_value=net_usd_value / QUOTE_PRECISION,
        margin_requirement=margin_requirement / QUOTE_PRECISION,
        leverage=leverage,
        health=health,
    )


async def get_account_summary(account: Account) -> dict:
    """
    Get a detailed account summary.
    
    Args:
        account: Trading account
    
    Returns:
        dict with account details
    """
    portfolio = await get_portfolio(account)
    
    return {
        "total_value": f"${portfolio.total_collateral:.2f}",
        "available": f"${portfolio.free_collateral:.2f}",
        "net_value": f"${portfolio.net_usd_value:.2f}",
        "margin_used": f"${portfolio.margin_requirement:.2f}",
        "leverage": f"{portfolio.leverage:.2f}x",
        "health": f"{portfolio.health}%",
    }
