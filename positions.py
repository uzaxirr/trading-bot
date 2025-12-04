import logging
from typing import List
from dataclasses import dataclass

from driftpy.constants.numeric_constants import BASE_PRECISION, PRICE_PRECISION, QUOTE_PRECISION

from trading import Account, DriftClientManager
from markets import get_market_by_index, MARKETS

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a perp position."""
    symbol: str
    market_index: int
    position: float  # Base asset amount (positive = long, negative = short)
    sign: str  # "LONG" or "SHORT"
    entry_price: float
    unrealized_pnl: float
    liquidation_price: float


async def get_open_positions(account: Account) -> List[Position]:
    """
    Fetch all open perp positions for an account.
    
    Args:
        account: Trading account
    
    Returns:
        List of Position objects
    """
    drift_client = await DriftClientManager.get_client(account)
    user = drift_client.get_user()
    
    active_positions = user.get_active_perp_positions()
    positions = []
    
    for perp_pos in active_positions:
        if perp_pos.base_asset_amount == 0:
            continue
        
        market_index = perp_pos.market_index
        
        # Find symbol for this market
        symbol = None
        for sym, market in MARKETS.items():
            if market.market_index == market_index:
                symbol = sym
                break
        
        if symbol is None:
            symbol = f"MARKET_{market_index}"
        
        # Calculate position details
        base_amount = perp_pos.base_asset_amount / BASE_PRECISION
        is_long = perp_pos.base_asset_amount > 0
        
        # Get entry price from quote_entry_amount / base_asset_amount
        if perp_pos.base_asset_amount != 0:
            entry_price = abs(perp_pos.quote_entry_amount) / abs(perp_pos.base_asset_amount) * BASE_PRECISION / QUOTE_PRECISION
        else:
            entry_price = 0
        
        # Get unrealized PnL
        try:
            unrealized_pnl = user.get_unrealized_pnl(
                with_funding=True,
                market_index=market_index
            ) / QUOTE_PRECISION
        except Exception:
            unrealized_pnl = 0
        
        # Get liquidation price
        try:
            liq_price = user.get_perp_liq_price(market_index)
            if liq_price > 0:
                liq_price = liq_price / PRICE_PRECISION
            else:
                liq_price = 0
        except Exception:
            liq_price = 0
        
        positions.append(Position(
            symbol=symbol,
            market_index=market_index,
            position=base_amount,
            sign="LONG" if is_long else "SHORT",
            entry_price=entry_price,
            unrealized_pnl=unrealized_pnl,
            liquidation_price=liq_price,
        ))
    
    return positions


async def get_position(account: Account, symbol: str) -> Position | None:
    """
    Get a specific position by symbol.
    
    Args:
        account: Trading account
        symbol: Market symbol (e.g., "SOL")
    
    Returns:
        Position object or None if no position
    """
    if symbol not in MARKETS:
        return None
    
    market = MARKETS[symbol]
    positions = await get_open_positions(account)
    
    for pos in positions:
        if pos.market_index == market.market_index:
            return pos
    
    return None
