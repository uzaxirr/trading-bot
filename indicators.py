from typing import List
from dataclasses import dataclass


@dataclass
class Indicators:
    mid_prices: List[float]
    macd: List[float]
    ema20s: List[float]


def get_ema(prices: List[float], period: int) -> List[float]:
    """Calculate Exponential Moving Average."""
    if len(prices) < period:
        raise ValueError(f"Not enough prices provided. Need {period}, got {len(prices)}")
    
    multiplier = 2 / (period + 1)
    
    # Initial SMA
    sma = sum(prices[:period]) / period
    emas = [sma]
    
    # Calculate EMA for remaining prices
    for i in range(period, len(prices)):
        ema = emas[-1] * (1 - multiplier) + prices[i] * multiplier
        emas.append(ema)
    
    return emas


def get_mid_prices(candlesticks: list) -> List[float]:
    """Calculate mid prices from candlesticks."""
    return [round((float(c.open) + float(c.close)) / 2, 3) for c in candlesticks]


def get_macd(prices: List[float]) -> List[float]:
    """Calculate MACD (EMA12 - EMA26)."""
    if len(prices) < 26:
        raise ValueError(f"Not enough prices for MACD. Need 26, got {len(prices)}")
    
    ema26 = get_ema(prices, 26)
    ema12 = get_ema(prices, 12)
    
    # Align lengths - ema12 has more values, trim to match ema26
    ema12 = ema12[-len(ema26):]
    
    return [e12 - e26 for e12, e26 in zip(ema12, ema26)]


