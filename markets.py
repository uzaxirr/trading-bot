from dataclasses import dataclass
from typing import Dict


@dataclass
class Market:
    """Drift perp market configuration."""
    market_index: int
    symbol: str
    base_asset_symbol: str
    leverage: int  # Max leverage for this market


# Drift mainnet perp markets
# Full list: https://github.com/drift-labs/protocol-v2/blob/master/sdk/src/constants/perpMarkets.ts
MARKETS: Dict[str, Market] = {
    "SOL": Market(
        market_index=0,
        symbol="SOL-PERP",
        base_asset_symbol="SOL",
        leverage=20,
    ),
    "BTC": Market(
        market_index=1,
        symbol="BTC-PERP",
        base_asset_symbol="BTC",
        leverage=20,
    ),
    "ETH": Market(
        market_index=2,
        symbol="ETH-PERP",
        base_asset_symbol="ETH",
        leverage=20,
    ),
    "DOGE": Market(
        market_index=7,
        symbol="DOGE-PERP",
        base_asset_symbol="DOGE",
        leverage=10,
    ),
    "WIF": Market(
        market_index=23,
        symbol="WIF-PERP",
        base_asset_symbol="WIF",
        leverage=10,
    ),
    "JUP": Market(
        market_index=24,
        symbol="JUP-PERP",
        base_asset_symbol="JUP",
        leverage=10,
    ),
    "BONK": Market(
        market_index=4,
        symbol="1MBONK-PERP",
        base_asset_symbol="1MBONK",
        leverage=10,
    ),
    "HYPE": Market(
        market_index=59,
        symbol="HYPE-PERP",
        base_asset_symbol="HYPE",
        leverage=10,
    ),
}


def get_market_by_index(market_index: int) -> Market | None:
    """Get market by its index."""
    for market in MARKETS.values():
        if market.market_index == market_index:
            return market
    return None
