"""
Backtesting engine for strategy validation.
Tests strategies against historical price data from Birdeye.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import httpx

from config import BIRDEYE_API_KEY
from indicators import get_ema, get_macd

logger = logging.getLogger(__name__)

# Token addresses for Birdeye API
TOKEN_ADDRESSES = {
    "SOL": "So11111111111111111111111111111111111111112",
    "BTC": "3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh",
    "ETH": "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",
    "WIF": "A5gVMLiRTG1fheq8rMZYgPxLzPA8qxLx3DAQEi3LBAmS",
    "DOGE": "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
    "JUP": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
    "HYPE": "HYPERfwdTjyJ2SCaKHmpF2MtrXqWxrsotYDsTrshHWq8",
}

BIRDEYE_API_URL = "https://public-api.birdeye.so"


@dataclass
class Candle:
    """OHLCV candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class BacktestTrade:
    """A simulated trade in the backtest."""
    entry_time: datetime
    entry_price: float
    side: str  # "LONG" or "SHORT"
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_percent: Optional[float] = None
    is_win: Optional[bool] = None


@dataclass
class BacktestResult:
    """Results of a backtest."""
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl_percent: float
    avg_win_percent: float
    avg_loss_percent: float
    profit_factor: float
    max_drawdown_percent: float
    sharpe_ratio: float
    trades: List[BacktestTrade] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "total_pnl_percent": round(self.total_pnl_percent, 2),
            "avg_win_percent": round(self.avg_win_percent, 2),
            "avg_loss_percent": round(self.avg_loss_percent, 2),
            "profit_factor": round(self.profit_factor, 2),
            "max_drawdown_percent": round(self.max_drawdown_percent, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
        }


async def fetch_historical_candles(
    symbol: str,
    timeframe: str = "15m",
    days: int = 30,
) -> List[Candle]:
    """Fetch historical candle data from Birdeye."""
    token_address = TOKEN_ADDRESSES.get(symbol)
    if not token_address:
        logger.warning(f"Unknown symbol: {symbol}")
        return []
    
    # Convert timeframe to Birdeye format
    tf_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}
    birdeye_tf = tf_map.get(timeframe, "15m")
    
    now = int(datetime.now().timestamp())
    start = now - (days * 24 * 60 * 60)
    
    url = f"{BIRDEYE_API_URL}/defi/ohlcv"
    params = {
        "address": token_address,
        "type": birdeye_tf,
        "time_from": start,
        "time_to": now,
    }
    headers = {"X-API-KEY": BIRDEYE_API_KEY}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=headers, timeout=30)
            
            if response.status_code != 200:
                logger.warning(f"Birdeye API error: {response.status_code}")
                return []
            
            data = response.json()
            items = data.get("data", {}).get("items", [])
            
            candles = []
            for item in items:
                candles.append(Candle(
                    timestamp=datetime.fromtimestamp(item["unixTime"]),
                    open=item["o"],
                    high=item["h"],
                    low=item["l"],
                    close=item["c"],
                    volume=item.get("v", 0),
                ))
            
            return sorted(candles, key=lambda x: x.timestamp)
    
    except Exception as e:
        logger.error(f"Failed to fetch candles: {e}")
        return []


def calculate_indicators_for_candles(candles: List[Candle]) -> Dict[str, List[float]]:
    """Calculate technical indicators for a list of candles."""
    closes = [c.close for c in candles]
    
    # Need enough data for indicators
    if len(closes) < 50:
        return {"close": closes}
    
    indicators = {
        "close": closes,
    }
    
    # Calculate EMAs
    try:
        ema_9 = get_ema(closes, 9)
        ema_21 = get_ema(closes, 21)
        ema_50 = get_ema(closes, 50)
        
        # Pad to match length
        indicators["ema_9"] = [ema_9[0]] * (len(closes) - len(ema_9)) + ema_9
        indicators["ema_21"] = [ema_21[0]] * (len(closes) - len(ema_21)) + ema_21
        indicators["ema_50"] = [ema_50[0]] * (len(closes) - len(ema_50)) + ema_50
    except Exception:
        indicators["ema_9"] = closes
        indicators["ema_21"] = closes
        indicators["ema_50"] = closes
    
    # Calculate MACD
    try:
        macd = get_macd(closes)
        macd_signal = get_ema(macd, 9) if len(macd) >= 9 else macd
        
        # Pad to match length
        indicators["macd"] = [0] * (len(closes) - len(macd)) + macd
        indicators["macd_signal"] = [0] * (len(closes) - len(macd_signal)) + macd_signal
    except Exception:
        indicators["macd"] = [0] * len(closes)
        indicators["macd_signal"] = [0] * len(closes)
    
    # Calculate RSI
    indicators["rsi"] = calculate_rsi(closes)
    
    return indicators


def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """Calculate RSI indicator."""
    if len(prices) < period + 1:
        return [50.0] * len(prices)
    
    rsi_values = [50.0] * period  # Default for initial values
    
    gains = []
    losses = []
    
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        gains.append(max(change, 0))
        losses.append(abs(min(change, 0)))
    
    # Calculate initial averages
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    for i in range(period, len(prices)):
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))
        
        if i < len(gains):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    return rsi_values


def evaluate_condition(condition: Dict, indicators: Dict, index: int) -> bool:
    """Evaluate a single condition against indicators at a specific index."""
    indicator = condition.get("indicator", "").lower()
    operator = condition.get("operator", "")
    value = condition.get("value")
    compare_to = condition.get("compare_to", "").lower()
    
    # Get indicator value
    if indicator not in indicators:
        return False
    
    ind_values = indicators[indicator]
    if index >= len(ind_values):
        return False
    
    current_value = ind_values[index]
    
    # Get comparison value
    if compare_to and compare_to in indicators:
        if index >= len(indicators[compare_to]):
            return False
        compare_value = indicators[compare_to][index]
    elif value is not None:
        compare_value = float(value)
    else:
        return False
    
    # Evaluate operator
    if operator == ">" or operator == "above":
        return current_value > compare_value
    elif operator == "<" or operator == "below":
        return current_value < compare_value
    elif operator == "cross_above":
        if index < 1:
            return False
        prev_value = ind_values[index - 1]
        if compare_to in indicators:
            prev_compare = indicators[compare_to][index - 1]
        else:
            prev_compare = compare_value
        return prev_value <= prev_compare and current_value > compare_value
    elif operator == "cross_below":
        if index < 1:
            return False
        prev_value = ind_values[index - 1]
        if compare_to in indicators:
            prev_compare = indicators[compare_to][index - 1]
        else:
            prev_compare = compare_value
        return prev_value >= prev_compare and current_value < compare_value
    
    return False


def evaluate_conditions(conditions: List[Dict], indicators: Dict, index: int, logic: str = "AND") -> bool:
    """Evaluate multiple conditions with AND/OR logic."""
    if not conditions:
        return False
    
    results = [evaluate_condition(c, indicators, index) for c in conditions]
    
    if logic.upper() == "AND":
        return all(results)
    else:  # OR
        return any(results)


async def backtest_strategy(
    symbol: str,
    side: str,
    entry_conditions: List[Dict],
    exit_conditions: List[Dict],
    stop_loss_percent: float = 5.0,
    take_profit_percent: float = 10.0,
    timeframe: str = "15m",
    days: int = 30,
) -> BacktestResult:
    """
    Run a backtest on a strategy.
    
    Args:
        symbol: Trading symbol (e.g., "SOL")
        side: "LONG" or "SHORT"
        entry_conditions: List of entry condition dicts
        exit_conditions: List of exit condition dicts
        stop_loss_percent: Stop loss percentage
        take_profit_percent: Take profit percentage
        timeframe: Candle timeframe
        days: Number of days to backtest
    
    Returns:
        BacktestResult with performance metrics
    """
    logger.info(f"Starting backtest for {symbol} {side} strategy over {days} days")
    
    # Fetch historical data
    candles = await fetch_historical_candles(symbol, timeframe, days)
    
    if len(candles) < 50:
        logger.warning(f"Not enough candles for backtest: {len(candles)}")
        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            start_date=datetime.now() - timedelta(days=days),
            end_date=datetime.now(),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            total_pnl_percent=0,
            avg_win_percent=0,
            avg_loss_percent=0,
            profit_factor=0,
            max_drawdown_percent=0,
            sharpe_ratio=0,
        )
    
    # Calculate indicators
    indicators = calculate_indicators_for_candles(candles)
    
    # Simulate trading
    trades: List[BacktestTrade] = []
    current_trade: Optional[BacktestTrade] = None
    equity_curve = [100.0]  # Start with $100
    
    for i in range(50, len(candles)):  # Start after warm-up period
        candle = candles[i]
        price = candle.close
        
        if current_trade is None:
            # Check entry conditions
            if evaluate_conditions(entry_conditions, indicators, i, "AND"):
                current_trade = BacktestTrade(
                    entry_time=candle.timestamp,
                    entry_price=price,
                    side=side,
                )
        else:
            # Calculate current P&L
            if current_trade.side == "LONG":
                pnl_percent = ((price - current_trade.entry_price) / current_trade.entry_price) * 100
            else:  # SHORT
                pnl_percent = ((current_trade.entry_price - price) / current_trade.entry_price) * 100
            
            exit_reason = None
            
            # Check stop loss
            if pnl_percent <= -stop_loss_percent:
                exit_reason = "STOP_LOSS"
            # Check take profit
            elif pnl_percent >= take_profit_percent:
                exit_reason = "TAKE_PROFIT"
            # Check exit conditions
            elif evaluate_conditions(exit_conditions, indicators, i, "OR"):
                exit_reason = "SIGNAL"
            
            if exit_reason:
                current_trade.exit_time = candle.timestamp
                current_trade.exit_price = price
                current_trade.exit_reason = exit_reason
                current_trade.pnl_percent = pnl_percent
                current_trade.is_win = pnl_percent > 0
                trades.append(current_trade)
                
                # Update equity
                equity_curve.append(equity_curve[-1] * (1 + pnl_percent / 100))
                
                current_trade = None
    
    # Close any open trade at the end
    if current_trade:
        price = candles[-1].close
        if current_trade.side == "LONG":
            pnl_percent = ((price - current_trade.entry_price) / current_trade.entry_price) * 100
        else:
            pnl_percent = ((current_trade.entry_price - price) / current_trade.entry_price) * 100
        
        current_trade.exit_time = candles[-1].timestamp
        current_trade.exit_price = price
        current_trade.exit_reason = "END_OF_TEST"
        current_trade.pnl_percent = pnl_percent
        current_trade.is_win = pnl_percent > 0
        trades.append(current_trade)
        equity_curve.append(equity_curve[-1] * (1 + pnl_percent / 100))
    
    # Calculate metrics
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.is_win)
    losing_trades = total_trades - winning_trades
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    wins = [t.pnl_percent for t in trades if t.is_win]
    losses = [t.pnl_percent for t in trades if not t.is_win]
    
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    
    total_pnl = sum(t.pnl_percent for t in trades)
    
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit if gross_profit > 0 else 0
    
    # Calculate max drawdown
    peak = equity_curve[0]
    max_drawdown = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        drawdown = (peak - eq) / peak * 100
        max_drawdown = max(max_drawdown, drawdown)
    
    # Calculate Sharpe ratio (simplified)
    if len(trades) > 1:
        returns = [t.pnl_percent for t in trades]
        avg_return = sum(returns) / len(returns)
        std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
        sharpe_ratio = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0  # Annualized
    else:
        sharpe_ratio = 0
    
    result = BacktestResult(
        symbol=symbol,
        timeframe=timeframe,
        start_date=candles[0].timestamp if candles else datetime.now() - timedelta(days=days),
        end_date=candles[-1].timestamp if candles else datetime.now(),
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        total_pnl_percent=total_pnl,
        avg_win_percent=avg_win,
        avg_loss_percent=avg_loss,
        profit_factor=profit_factor,
        max_drawdown_percent=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        trades=trades,
    )
    
    logger.info(f"Backtest complete: {total_trades} trades, {win_rate:.1%} win rate, {total_pnl:.1f}% P&L")
    
    return result

