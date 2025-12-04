import base64
import logging
from dataclasses import dataclass
from typing import Optional

from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from anchorpy.provider import Wallet

from driftpy.drift_client import DriftClient
from driftpy.drift_user import DriftUser
from driftpy.types import (
    OrderParams,
    OrderType,
    MarketType,
    PositionDirection,
)
from driftpy.constants.numeric_constants import BASE_PRECISION, PRICE_PRECISION

from config import RPC_URL, DRIFT_ENV
from markets import MARKETS, get_market_by_index

logger = logging.getLogger(__name__)


@dataclass
class Account:
    """Trading account configuration."""
    private_key: str  # Base58 or JSON array of bytes
    name: str
    model_name: str
    invocation_count: int
    id: str


def load_keypair(private_key: str) -> Keypair:
    """Load a Solana keypair from various formats."""
    try:
        # Try base58 format first
        return Keypair.from_base58_string(private_key)
    except Exception:
        pass
    
    try:
        # Try JSON array format [1,2,3,...]
        import json
        key_bytes = bytes(json.loads(private_key))
        return Keypair.from_bytes(key_bytes)
    except Exception:
        pass
    
    try:
        # Try base64 format
        key_bytes = base64.b64decode(private_key)
        return Keypair.from_bytes(key_bytes)
    except Exception:
        pass
    
    raise ValueError("Could not parse private key. Expected base58, JSON array, or base64 format.")


class DriftClientManager:
    """Manages DriftClient instances for accounts."""
    
    _clients: dict[str, DriftClient] = {}
    _connections: dict[str, AsyncClient] = {}
    
    @classmethod
    async def get_client(cls, account: Account) -> DriftClient:
        """Get or create a DriftClient for the given account."""
        if account.id in cls._clients:
            return cls._clients[account.id]
        
        keypair = load_keypair(account.private_key)
        wallet = Wallet(keypair)
        
        connection = AsyncClient(RPC_URL)
        cls._connections[account.id] = connection
        
        drift_client = DriftClient(
            connection,
            wallet,
            env=DRIFT_ENV,
        )
        
        await drift_client.subscribe()
        cls._clients[account.id] = drift_client
        
        logger.info(f"Created DriftClient for account {account.name} ({wallet.public_key})")
        return drift_client
    
    @classmethod
    async def close_client(cls, account_id: str):
        """Close and remove a DriftClient."""
        if account_id in cls._clients:
            # DriftClient doesn't have a close method, but we close the connection
            pass
        if account_id in cls._connections:
            await cls._connections[account_id].close()
            del cls._connections[account_id]
        if account_id in cls._clients:
            del cls._clients[account_id]
    
    @classmethod
    async def close_all(cls):
        """Close all clients."""
        for account_id in list(cls._clients.keys()):
            await cls.close_client(account_id)


async def get_current_price(drift_client: DriftClient, market_index: int) -> float:
    """Get the current oracle price for a market."""
    oracle_price_data = drift_client.get_oracle_price_data_for_perp_market(market_index)
    if oracle_price_data is None:
        raise ValueError(f"No oracle price data for market {market_index}")
    return oracle_price_data.price / PRICE_PRECISION


async def create_position(
    account: Account,
    symbol: str,
    side: str,
    quantity: float,
) -> dict:
    """
    Create a new perp position on Drift.
    
    Args:
        account: Trading account
        symbol: Market symbol (e.g., "SOL", "BTC")
        side: "LONG" or "SHORT"
        quantity: Position size in base asset units
    
    Returns:
        dict with success status and transaction signature
    """
    if symbol not in MARKETS:
        raise ValueError(f"Unknown market symbol: {symbol}")
    
    market = MARKETS[symbol]
    drift_client = await DriftClientManager.get_client(account)
    
    # Get current oracle price for slippage calculation
    current_price = await get_current_price(drift_client, market.market_index)
    
    # Set price with slippage tolerance (1%)
    # For longs, we set max price; for shorts, we set min price
    slippage = 0.01
    if side == "LONG":
        limit_price = int(current_price * (1 + slippage) * PRICE_PRECISION)
    else:
        limit_price = int(current_price * (1 - slippage) * PRICE_PRECISION)
    
    # Convert quantity to base precision
    base_asset_amount = int(abs(quantity) * BASE_PRECISION)
    
    order_params = OrderParams(
        order_type=OrderType.Market(),
        market_type=MarketType.Perp(),
        direction=PositionDirection.Long() if side == "LONG" else PositionDirection.Short(),
        base_asset_amount=base_asset_amount,
        market_index=market.market_index,
        price=limit_price,
    )
    
    logger.info(f"Placing {side} order for {quantity} {symbol} at ~${current_price:.2f}")
    
    try:
        tx_sig = await drift_client.place_perp_order(order_params)
        logger.info(f"Order placed successfully. TX: {tx_sig}")
        
        return {
            "success": True,
            "tx_signature": str(tx_sig),
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": current_price,
        }
    except Exception as e:
        logger.error(f"Failed to place order: {e}")
        raise


async def close_position(
    account: Account,
    symbol: str,
) -> dict:
    """
    Close a specific perp position.
    
    Args:
        account: Trading account
        symbol: Market symbol to close
    
    Returns:
        dict with success status
    """
    if symbol not in MARKETS:
        raise ValueError(f"Unknown market symbol: {symbol}")
    
    market = MARKETS[symbol]
    drift_client = await DriftClientManager.get_client(account)
    
    # Get current position
    user = drift_client.get_user()
    position = user.get_perp_position(market.market_index)
    
    if position is None or position.base_asset_amount == 0:
        logger.info(f"No open position in {symbol}")
        return {"success": True, "message": "No position to close"}
    
    # Determine direction to close (opposite of current position)
    base_amount = position.base_asset_amount
    is_long = base_amount > 0
    
    # Get current price for slippage
    current_price = await get_current_price(drift_client, market.market_index)
    slippage = 0.01
    
    # To close a long, we sell (short); to close a short, we buy (long)
    if is_long:
        direction = PositionDirection.Short()
        limit_price = int(current_price * (1 - slippage) * PRICE_PRECISION)
    else:
        direction = PositionDirection.Long()
        limit_price = int(current_price * (1 + slippage) * PRICE_PRECISION)
    
    order_params = OrderParams(
        order_type=OrderType.Market(),
        market_type=MarketType.Perp(),
        direction=direction,
        base_asset_amount=abs(base_amount),
        market_index=market.market_index,
        price=limit_price,
        reduce_only=True,
    )
    
    logger.info(f"Closing position in {symbol}: {base_amount / BASE_PRECISION:.4f}")
    
    try:
        tx_sig = await drift_client.place_perp_order(order_params)
        logger.info(f"Position closed successfully. TX: {tx_sig}")
        
        return {
            "success": True,
            "tx_signature": str(tx_sig),
            "symbol": symbol,
            "closed_amount": abs(base_amount) / BASE_PRECISION,
        }
    except Exception as e:
        logger.error(f"Failed to close position: {e}")
        raise


async def close_all_positions(account: Account) -> dict:
    """
    Close all open perp positions.
    
    Args:
        account: Trading account
    
    Returns:
        dict with success status and list of closed positions
    """
    drift_client = await DriftClientManager.get_client(account)
    user = drift_client.get_user()
    
    active_positions = user.get_active_perp_positions()
    closed = []
    
    for position in active_positions:
        if position.base_asset_amount == 0:
            continue
        
        market = get_market_by_index(position.market_index)
        if market is None:
            logger.warning(f"Unknown market index: {position.market_index}")
            continue
        
        # Find the symbol for this market
        symbol = None
        for sym, m in MARKETS.items():
            if m.market_index == position.market_index:
                symbol = sym
                break
        
        if symbol is None:
            continue
        
        try:
            result = await close_position(account, symbol)
            if result.get("success"):
                closed.append(symbol)
        except Exception as e:
            logger.error(f"Failed to close position in market {position.market_index}: {e}")
    
    return {
        "success": True,
        "closed_positions": closed,
    }
