import logging
import time
from typing import List
from dataclasses import dataclass

import httpx
from driftpy.constants.numeric_constants import PRICE_PRECISION

from config import DRIFT_ENV, BIRDEYE_API_KEY
from trading import Account, DriftClientManager
from markets import MARKETS
from indicators import get_ema, get_macd, Indicators

logger = logging.getLogger(__name__)

# Birdeye API for historical price data (free tier available)
BIRDEYE_API_URL = "https://public-api.birdeye.so"

# Token addresses for price lookups
TOKEN_ADDRESSES = {
    "SOL": "So11111111111111111111111111111111111111112",
    "BTC": "3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh",  # Wrapped BTC
    "ETH": "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",  # Wrapped ETH
    "DOGE": "A5gVMLiRTG1fheq8rMZYgPxLzPA8qxLx3DAQEi3LBAmS",
    "WIF": "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
    "JUP": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
    "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    "HYPE": "HYPERfwdTjyJ2SCaKHmpF2MtrXqWxrsotYDsTrshHWq8",
}


@dataclass
class PriceData:
    """Price data point."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


async def get_oracle_price(account: Account, symbol: str) -> float:
    """
    Get current oracle price from Drift.
    
    Args:
        account: Trading account (needed for DriftClient)
        symbol: Market symbol
    
    Returns:
        Current price in USD
    """
    if symbol not in MARKETS:
        raise ValueError(f"Unknown symbol: {symbol}")
    
    market = MARKETS[symbol]
    drift_client = await DriftClientManager.get_client(account)
    
    oracle_data = drift_client.get_oracle_price_data_for_perp_market(market.market_index)
    if oracle_data is None:
        raise ValueError(f"No oracle data for {symbol}")
    
    return oracle_data.price / PRICE_PRECISION


async def get_historical_prices_birdeye(
    symbol: str,
    interval: str = "15m",
    limit: int = 50,
) -> List[PriceData]:
    """
    Fetch historical price data from Birdeye API.
    
    Args:
        symbol: Token symbol
        interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d)
        limit: Number of candles to fetch
    
    Returns:
        List of PriceData objects
    """
    if symbol not in TOKEN_ADDRESSES:
        logger.warning(f"No token address for {symbol}, using mock data")
        return []
    
    token_address = TOKEN_ADDRESSES[symbol]
    
    # Map interval to Birdeye format
    interval_map = {
        "1m": "1m",
        "5m": "5m", 
        "15m": "15m",
        "1h": "1H",
        "4h": "4H",
        "1d": "1D",
    }
    birdeye_interval = interval_map.get(interval, "15m")
    
    # Calculate time range
    now = int(time.time())
    interval_seconds = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
    }
    duration = interval_seconds.get(interval, 900) * limit
    start_time = now - duration
    
    url = f"{BIRDEYE_API_URL}/defi/ohlcv"
    params = {
        "address": token_address,
        "type": birdeye_interval,
        "time_from": start_time,
        "time_to": now,
    }
    
    try:
        headers = {}
        if BIRDEYE_API_KEY:
            headers["X-API-KEY"] = BIRDEYE_API_KEY
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Birdeye API error: {response.status_code}")
                return []
            
            data = response.json()
            items = data.get("data", {}).get("items", [])
            
            prices = []
            for item in items:
                prices.append(PriceData(
                    timestamp=item.get("unixTime", 0),
                    open=item.get("o", 0),
                    high=item.get("h", 0),
                    low=item.get("l", 0),
                    close=item.get("c", 0),
                    volume=item.get("v", 0),
                ))
            
            return prices
    except Exception as e:
        logger.warning(f"Failed to fetch Birdeye data for {symbol}: {e}")
        return []


async def get_indicators(
    account: Account,
    symbol: str,
    interval: str = "15m",
) -> Indicators:
    """
    Get technical indicators for a symbol.
    
    Args:
        account: Trading account
        symbol: Market symbol
        interval: Time interval
    
    Returns:
        Indicators object with mid_prices, ema20s, and macd
    """
    # Try to get historical prices
    prices_data = await get_historical_prices_birdeye(symbol, interval, 50)
    
    if len(prices_data) < 26:
        # Fallback: use current oracle price repeated
        # This is not ideal but ensures we have some data
        try:
            current_price = await get_oracle_price(account, symbol)
            mid_prices = [current_price] * 30
        except Exception:
            return Indicators(mid_prices=[], macd=[], ema20s=[])
    else:
        mid_prices = [(p.open + p.close) / 2 for p in prices_data]
    
    # Calculate indicators
    try:
        macd = get_macd(mid_prices)[-10:]
        ema20s = get_ema(mid_prices, 20)[-10:]
        mid_prices = [round(p, 4) for p in mid_prices[-10:]]
        macd = [round(m, 4) for m in macd]
        ema20s = [round(e, 4) for e in ema20s]
    except Exception as e:
        logger.warning(f"Failed to calculate indicators for {symbol}: {e}")
        return Indicators(
            mid_prices=[round(p, 4) for p in mid_prices[-10:]],
            macd=[],
            ema20s=[],
        )
    
    return Indicators(
        mid_prices=mid_prices,
        macd=macd,
        ema20s=ema20s,
    )


async def get_all_market_data(account: Account) -> dict:
    """
    Get market data for all configured markets.
    
    Args:
        account: Trading account
    
    Returns:
        dict mapping symbol to market data
    """
    market_data = {}
    
    for symbol in MARKETS.keys():
        try:
            # Get current price
            current_price = await get_oracle_price(account, symbol)
            
            # Get short-term indicators
            short_term = await get_indicators(account, symbol, "5m")
            
            # Get long-term indicators  
            long_term = await get_indicators(account, symbol, "4h")
            
            market_data[symbol] = {
                "current_price": current_price,
                "short_term": short_term,
                "long_term": long_term,
            }
        except Exception as e:
            logger.warning(f"Failed to get market data for {symbol}: {e}")
            market_data[symbol] = {
                "current_price": 0,
                "short_term": Indicators([], [], []),
                "long_term": Indicators([], [], []),
            }
    
    return market_data
