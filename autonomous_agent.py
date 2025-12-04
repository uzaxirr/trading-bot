"""
Autonomous Trading Agent - The Self-Operating Loop

This module implements a fully autonomous trading agent that:
1. OBSERVES - Continuously analyzes market conditions
2. THINKS - Generates strategy hypotheses using Claude
3. TESTS - Backtests strategies on historical data
4. EXECUTES - Trades autonomously based on validated strategies
5. LEARNS - Tracks performance and improves over time
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import anthropic
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from markets import MARKETS
from models import (
    Strategy, StrategyStatus, StrategyTrade,
    AgentActivity, ActivityType, Model,
    async_session
)
from backtester import backtest_strategy, fetch_historical_candles, calculate_indicators_for_candles
from stock_data import get_indicators, get_all_market_data
from trading import Account, create_position, close_position

logger = logging.getLogger(__name__)

# Minimum thresholds for strategy approval
MIN_WIN_RATE = 0.45
MIN_PROFIT_FACTOR = 1.1
MIN_SHARPE_RATIO = 0.3
MAX_DRAWDOWN = 25.0
MIN_TRADES = 2  # Lowered for testing

# Strategy generation prompt
STRATEGY_GENERATION_PROMPT = """You are an autonomous trading strategy generator. Analyze the market data and create a trading strategy.

Current Market Analysis:
{market_analysis}

Your task:
1. Identify patterns or setups that could be profitable
2. Create a specific, testable strategy with clear entry/exit rules
3. Use only these indicators: ema_9, ema_21, ema_50, rsi, macd, macd_signal

Respond with ONLY valid JSON in this exact format:
{{
    "name": "Strategy name (be creative and descriptive)",
    "description": "Brief explanation of the strategy logic",
    "symbol": "ONE symbol from: SOL, BTC, ETH, WIF, DOGE, JUP, HYPE",
    "side": "LONG or SHORT",
    "entry_conditions": [
        {{"indicator": "ema_9", "operator": "cross_above", "compare_to": "ema_21"}},
        {{"indicator": "rsi", "operator": ">", "value": 50}}
    ],
    "exit_conditions": [
        {{"indicator": "ema_9", "operator": "cross_below", "compare_to": "ema_21"}}
    ],
    "stop_loss_percent": 5.0,
    "take_profit_percent": 10.0,
    "reasoning": "Explain why this strategy might work"
}}

Valid operators: ">", "<", "above", "below", "cross_above", "cross_below"
For cross_above/cross_below, use "compare_to" with another indicator.
For >/</above/below, use "value" with a number OR "compare_to" with another indicator.

Generate ONE strategy based on the current market conditions."""


class AutonomousAgent:
    """The self-operating trading agent."""
    
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.is_running = False
        self.current_cycle = 0
        
    async def log_activity(
        self,
        session: AsyncSession,
        activity_type: ActivityType,
        title: str,
        description: str = "",
        details: Dict = None,
        strategy_id: str = None,
        symbol: str = None,
    ):
        """Log agent activity to database."""
        activity = AgentActivity(
            activity_type=activity_type,
            title=title,
            description=description,
            details=json.dumps(details or {}),
            strategy_id=strategy_id,
            symbol=symbol,
        )
        session.add(activity)
        await session.commit()
        logger.info(f"🤖 [{activity_type.value}] {title}")
    
    async def observe_markets(self, session: AsyncSession, account: Account) -> Dict[str, Any]:
        """
        OBSERVE: Gather market intelligence.
        Analyzes all available markets and returns current state.
        """
        try:
            # Get all market data at once
            raw_data = await get_all_market_data(account)
            
            market_data = {}
            for symbol, data in raw_data.items():
                if data.get("current_price"):
                    short_ema = data.get("short_term_ema", data["current_price"])
                    long_ema = data.get("long_term_ema", data["current_price"])
                    
                    market_data[symbol] = {
                        "price": data["current_price"],
                        "ema_9": short_ema,
                        "ema_21": long_ema,
                        "macd": data.get("macd", 0),
                        "macd_signal": data.get("macd_signal", 0),
                        "trend": "BULLISH" if short_ema > long_ema else "BEARISH",
                    }
        except Exception as e:
            logger.error(f"Failed to observe markets: {e}")
            market_data = {}
        
        await self.log_activity(
            session,
            ActivityType.MARKET_ANALYSIS,
            "Market Scan Complete",
            f"Analyzed {len(market_data)} markets",
            {"markets": list(market_data.keys())},
        )
        
        return market_data
    
    async def generate_strategy(self, session: AsyncSession, market_data: Dict) -> Optional[Dict]:
        """
        THINK: Use Claude to generate a strategy hypothesis.
        """
        # Format market data for Claude
        market_analysis = "## Current Market Conditions\n\n"
        for symbol, data in market_data.items():
            market_analysis += f"**{symbol}**: ${data['price']:.2f} | Trend: {data['trend']} | "
            market_analysis += f"EMA9: {data['ema_9']:.2f} | EMA21: {data['ema_21']:.2f} | "
            market_analysis += f"MACD: {data['macd']:.4f}\n"
        
        try:
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": STRATEGY_GENERATION_PROMPT.format(market_analysis=market_analysis)
                }]
            )
            
            response_text = response.content[0].text.strip()
            
            # Extract JSON from response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            strategy_data = json.loads(response_text)
            
            await self.log_activity(
                session,
                ActivityType.STRATEGY_CREATED,
                f"Strategy Hypothesis: {strategy_data.get('name', 'Unknown')}",
                strategy_data.get('reasoning', ''),
                strategy_data,
                symbol=strategy_data.get('symbol'),
            )
            
            return strategy_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse strategy JSON: {e}")
            await self.log_activity(
                session,
                ActivityType.ERROR,
                "Strategy Generation Failed",
                f"JSON parse error: {e}",
            )
            return None
        except Exception as e:
            logger.error(f"Strategy generation error: {e}")
            await self.log_activity(
                session,
                ActivityType.ERROR,
                "Strategy Generation Failed",
                str(e),
            )
            return None
    
    async def test_strategy(self, session: AsyncSession, strategy_data: Dict) -> Optional[Strategy]:
        """
        TEST: Backtest the strategy on historical data.
        Returns Strategy object if it passes validation.
        """
        symbol = strategy_data.get("symbol")
        side = strategy_data.get("side", "LONG")
        entry_conditions = strategy_data.get("entry_conditions", [])
        exit_conditions = strategy_data.get("exit_conditions", [])
        stop_loss = strategy_data.get("stop_loss_percent", 5.0)
        take_profit = strategy_data.get("take_profit_percent", 10.0)
        
        await self.log_activity(
            session,
            ActivityType.BACKTEST_STARTED,
            f"Testing: {strategy_data.get('name')}",
            f"Running 30-day backtest on {symbol}",
            {"symbol": symbol, "side": side},
            symbol=symbol,
        )
        
        # Run backtest
        result = await backtest_strategy(
            symbol=symbol,
            side=side,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            stop_loss_percent=stop_loss,
            take_profit_percent=take_profit,
            timeframe="15m",
            days=30,
        )
        
        # Check if strategy passes thresholds
        passed = (
            result.total_trades >= MIN_TRADES and
            result.win_rate >= MIN_WIN_RATE and
            result.profit_factor >= MIN_PROFIT_FACTOR and
            result.sharpe_ratio >= MIN_SHARPE_RATIO and
            result.max_drawdown_percent <= MAX_DRAWDOWN
        )
        
        await self.log_activity(
            session,
            ActivityType.BACKTEST_COMPLETED,
            f"Backtest {'PASSED ✅' if passed else 'FAILED ❌'}: {strategy_data.get('name')}",
            f"Win Rate: {result.win_rate:.1%} | Profit Factor: {result.profit_factor:.2f} | Sharpe: {result.sharpe_ratio:.2f}",
            result.to_dict(),
            symbol=symbol,
        )
        
        # Create and save strategy (even if failed, for visibility)
        strategy = Strategy(
            name=strategy_data.get("name", f"Strategy-{datetime.now().strftime('%Y%m%d%H%M')}"),
            description=strategy_data.get("description", ""),
            symbols=json.dumps([symbol]),
            side=side,
            timeframe="15m",
            entry_conditions=json.dumps(entry_conditions),
            exit_conditions=json.dumps(exit_conditions),
            stop_loss_percent=stop_loss,
            take_profit_percent=take_profit,
            status=StrategyStatus.APPROVED if passed else StrategyStatus.DRAFT,
            backtest_win_rate=result.win_rate,
            backtest_total_trades=result.total_trades,
            backtest_profit_factor=result.profit_factor,
            backtest_max_drawdown=result.max_drawdown_percent,
            backtest_sharpe_ratio=result.sharpe_ratio,
        )
        
        session.add(strategy)
        await session.commit()
        await session.refresh(strategy)
        
        if not passed:
            logger.info(f"Strategy saved as DRAFT (failed validation): {result.to_dict()}")
            return None  # Don't deploy failed strategies
        
        return strategy
    
    async def deploy_strategy(self, session: AsyncSession, strategy: Strategy):
        """
        DEPLOY: Activate a validated strategy for live trading.
        """
        strategy.status = StrategyStatus.ACTIVE
        strategy.deployed_at = datetime.utcnow()
        await session.commit()
        
        await self.log_activity(
            session,
            ActivityType.STRATEGY_DEPLOYED,
            f"Strategy Deployed: {strategy.name}",
            f"Now actively trading {json.loads(strategy.symbols)}",
            {"strategy_id": strategy.id},
            strategy_id=strategy.id,
        )
    
    async def execute_strategies(self, session: AsyncSession, account: Account):
        """
        EXECUTE: Run all active strategies and execute trades.
        """
        # Get active strategies
        result = await session.execute(
            select(Strategy).where(Strategy.status == StrategyStatus.ACTIVE)
        )
        strategies = result.scalars().all()
        
        if not strategies:
            return
        
        for strategy in strategies:
            try:
                symbols = json.loads(strategy.symbols)
                entry_conditions = json.loads(strategy.entry_conditions)
                exit_conditions = json.loads(strategy.exit_conditions)
                
                for symbol in symbols:
                    # Check for open trades
                    open_trade_result = await session.execute(
                        select(StrategyTrade).where(
                            and_(
                                StrategyTrade.strategy_id == strategy.id,
                                StrategyTrade.symbol == symbol,
                                StrategyTrade.is_open == True,
                            )
                        )
                    )
                    open_trade = open_trade_result.scalar_one_or_none()
                    
                    # Get current indicators
                    indicators = await get_indicators(symbol)
                    if not indicators:
                        continue
                    
                    # Build indicator dict for evaluation
                    ind_dict = {
                        "close": [indicators.current_price],
                        "ema_9": [indicators.short_term_ema],
                        "ema_21": [indicators.long_term_ema],
                        "ema_50": [indicators.long_term_ema],  # Approximate
                        "macd": [indicators.macd],
                        "macd_signal": [indicators.macd_signal],
                        "rsi": [50],  # Default RSI
                    }
                    
                    if open_trade:
                        # Check exit conditions
                        should_exit = self._check_exit(
                            open_trade, indicators.current_price,
                            strategy.stop_loss_percent, strategy.take_profit_percent,
                            exit_conditions, ind_dict
                        )
                        
                        if should_exit:
                            await self._close_trade(
                                session, account, strategy, open_trade,
                                indicators.current_price, should_exit
                            )
                    else:
                        # Check entry conditions
                        should_enter = self._check_entry(entry_conditions, ind_dict)
                        
                        if should_enter:
                            await self._open_trade(
                                session, account, strategy, symbol,
                                indicators.current_price
                            )
                            
            except Exception as e:
                logger.error(f"Strategy execution error: {e}")
    
    def _check_entry(self, conditions: List[Dict], indicators: Dict) -> bool:
        """Check if entry conditions are met."""
        from backtester import evaluate_conditions
        return evaluate_conditions(conditions, indicators, 0, "AND")
    
    def _check_exit(
        self, trade: StrategyTrade, current_price: float,
        stop_loss: float, take_profit: float,
        conditions: List[Dict], indicators: Dict
    ) -> Optional[str]:
        """Check if exit conditions are met. Returns exit reason or None."""
        # Calculate P&L
        if trade.side == "LONG":
            pnl_percent = ((current_price - trade.entry_price) / trade.entry_price) * 100
        else:
            pnl_percent = ((trade.entry_price - current_price) / trade.entry_price) * 100
        
        if pnl_percent <= -stop_loss:
            return "STOP_LOSS"
        if pnl_percent >= take_profit:
            return "TAKE_PROFIT"
        
        from backtester import evaluate_conditions
        if evaluate_conditions(conditions, indicators, 0, "OR"):
            return "SIGNAL"
        
        return None
    
    async def _open_trade(
        self, session: AsyncSession, account: Account,
        strategy: Strategy, symbol: str, price: float
    ):
        """Open a new trade."""
        try:
            # Calculate quantity based on position size
            quantity = strategy.position_size_usd / price
            
            # Execute trade
            result = await create_position(account, symbol, strategy.side, quantity)
            
            if result.get("success"):
                # Record trade
                trade = StrategyTrade(
                    strategy_id=strategy.id,
                    symbol=symbol,
                    side=strategy.side,
                    entry_price=price,
                    entry_quantity=quantity,
                    entry_tx=result.get("tx_signature"),
                    is_open=True,
                )
                session.add(trade)
                
                await self.log_activity(
                    session,
                    ActivityType.TRADE_EXECUTED,
                    f"Opened {strategy.side} {symbol}",
                    f"Entry: ${price:.2f} | Qty: {quantity:.6f}",
                    {"strategy": strategy.name, "tx": result.get("tx_signature")},
                    strategy_id=strategy.id,
                    symbol=symbol,
                )
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to open trade: {e}")
    
    async def _close_trade(
        self, session: AsyncSession, account: Account,
        strategy: Strategy, trade: StrategyTrade,
        price: float, reason: str
    ):
        """Close an open trade."""
        try:
            # Execute close
            result = await close_position(account, trade.symbol)
            
            if result.get("success"):
                # Calculate P&L
                if trade.side == "LONG":
                    pnl = (price - trade.entry_price) * trade.entry_quantity
                    pnl_percent = ((price - trade.entry_price) / trade.entry_price) * 100
                else:
                    pnl = (trade.entry_price - price) * trade.entry_quantity
                    pnl_percent = ((trade.entry_price - price) / trade.entry_price) * 100
                
                # Update trade
                trade.exit_price = price
                trade.exit_time = datetime.utcnow()
                trade.exit_reason = reason
                trade.exit_tx = result.get("tx_signature")
                trade.pnl = pnl
                trade.pnl_percent = pnl_percent
                trade.is_win = pnl > 0
                trade.is_open = False
                
                # Update strategy stats
                strategy.live_trades += 1
                strategy.live_pnl += pnl
                if trade.is_win:
                    strategy.live_wins += 1
                
                await self.log_activity(
                    session,
                    ActivityType.TRADE_EXECUTED,
                    f"Closed {trade.symbol} ({reason})",
                    f"Exit: ${price:.2f} | P&L: {'+' if pnl > 0 else ''}{pnl_percent:.1f}%",
                    {"strategy": strategy.name, "pnl": pnl, "reason": reason},
                    strategy_id=strategy.id,
                    symbol=trade.symbol,
                )
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to close trade: {e}")
    
    async def review_and_learn(self, session: AsyncSession):
        """
        LEARN: Review strategy performance and retire poor performers.
        """
        result = await session.execute(
            select(Strategy).where(Strategy.status == StrategyStatus.ACTIVE)
        )
        strategies = result.scalars().all()
        
        for strategy in strategies:
            # Check if strategy has enough trades to evaluate
            if strategy.live_trades < 10:
                continue
            
            live_win_rate = strategy.live_wins / strategy.live_trades if strategy.live_trades > 0 else 0
            
            # Retire poorly performing strategies
            if live_win_rate < 0.35 or strategy.live_pnl < -50:  # Below 35% win rate or lost $50+
                strategy.status = StrategyStatus.RETIRED
                
                await self.log_activity(
                    session,
                    ActivityType.STRATEGY_RETIRED,
                    f"Strategy Retired: {strategy.name}",
                    f"Poor performance: {live_win_rate:.1%} win rate, ${strategy.live_pnl:.2f} P&L",
                    {"live_trades": strategy.live_trades, "live_pnl": strategy.live_pnl},
                    strategy_id=strategy.id,
                )
                
                await session.commit()
    
    async def run_cycle(self, account: Account):
        """
        Run one complete autonomous cycle:
        OBSERVE -> THINK -> TEST -> EXECUTE -> LEARN
        """
        self.current_cycle += 1
        
        async with async_session() as session:
            try:
                logger.info(f"🔄 Starting autonomous cycle #{self.current_cycle}")
                
                # 1. OBSERVE - Analyze markets
                market_data = await self.observe_markets(session, account)
                
                if not market_data:
                    logger.warning("No market data available")
                    return
                
                # 2. THINK - Generate new strategy (every 10 cycles)
                if self.current_cycle % 10 == 1:
                    strategy_data = await self.generate_strategy(session, market_data)
                    
                    if strategy_data:
                        # 3. TEST - Backtest the strategy
                        strategy = await self.test_strategy(session, strategy_data)
                        
                        if strategy:
                            # 4. DEPLOY - Activate approved strategy
                            await self.deploy_strategy(session, strategy)
                
                # 5. EXECUTE - Run active strategies
                await self.execute_strategies(session, account)
                
                # 6. LEARN - Review and improve
                await self.review_and_learn(session)
                
                logger.info(f"✅ Cycle #{self.current_cycle} complete")
                
            except Exception as e:
                logger.error(f"Autonomous cycle error: {e}")
                await self.log_activity(
                    session,
                    ActivityType.ERROR,
                    "Cycle Error",
                    str(e),
                )
    
    async def start(self, account: Account, interval_seconds: int = 60):
        """
        Start the autonomous agent loop.
        
        Args:
            account: Trading account to use
            interval_seconds: How often to run cycles
        """
        self.is_running = True
        logger.info("🤖 Autonomous Trading Agent STARTED")
        logger.info(f"📊 Running cycles every {interval_seconds} seconds")
        
        while self.is_running:
            await self.run_cycle(account)
            await asyncio.sleep(interval_seconds)
    
    def stop(self):
        """Stop the autonomous agent."""
        self.is_running = False
        logger.info("🛑 Autonomous Trading Agent STOPPED")


# Global agent instance
autonomous_agent = AutonomousAgent()


async def start_autonomous_agent(account: Account, interval_seconds: int = 60):
    """Start the autonomous agent."""
    await autonomous_agent.start(account, interval_seconds)


def stop_autonomous_agent():
    """Stop the autonomous agent."""
    autonomous_agent.stop()

