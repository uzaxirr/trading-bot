import json
import logging

import anthropic
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from markets import MARKETS
from models import Model, Invocation, ToolCall, ToolCallType
from portfolio import get_portfolio
from positions import get_open_positions
from stock_data import get_indicators, get_oracle_price
from trading import Account, create_position, close_all_positions
from prompt import build_prompt

logger = logging.getLogger(__name__)

# Define tools for Claude
TOOLS = [
    {
        "name": "createPosition",
        "description": "Open a perpetual futures position in the given market on Drift Protocol (Solana)",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "enum": list(MARKETS.keys()),
                    "description": "The market symbol to trade (e.g., SOL, BTC, ETH)"
                },
                "side": {
                    "type": "string",
                    "enum": ["LONG", "SHORT"],
                    "description": "LONG to buy/go long, SHORT to sell/go short"
                },
                "quantity": {
                    "type": "number",
                    "description": "The quantity in base asset units (e.g., 1.5 for 1.5 SOL)"
                }
            },
            "required": ["symbol", "side", "quantity"]
        }
    },
    {
        "name": "closeAllPositions",
        "description": "Close all currently open perpetual positions",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]


async def invoke_agent(account: Account, session: AsyncSession) -> str:
    """
    Main agent invocation logic.
    
    Args:
        account: Trading account
        session: Database session
    
    Returns:
        Agent response text
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Gather indicator data for all markets
    all_indicator_data = ""
    for symbol, market in MARKETS.items():
        try:
            # Get current price
            current_price = await get_oracle_price(account, symbol)
            
            # Get short-term indicators (5m)
            intraday = await get_indicators(account, symbol, "5m")
            
            # Get long-term indicators (4h)
            long_term = await get_indicators(account, symbol, "4h")
            
            all_indicator_data += f"""
    MARKET - {symbol} (Drift {market.symbol}, max {market.leverage}x leverage)
    Current Price: ${current_price:.4f}
    
    Intraday (5m candles) (oldest → latest):
    Mid prices - [{','.join(map(str, intraday.mid_prices))}]
    EMA20 - [{','.join(map(str, intraday.ema20s))}]
    MACD - [{','.join(map(str, intraday.macd))}]

    Long Term (4h candles) (oldest → latest):
    Mid prices - [{','.join(map(str, long_term.mid_prices))}]
    EMA20 - [{','.join(map(str, long_term.ema20s))}]
    MACD - [{','.join(map(str, long_term.macd))}]

    """
        except Exception as e:
            logger.warning(f"Failed to get indicators for {symbol}: {e}")
    
    # Get portfolio and positions
    portfolio = await get_portfolio(account)
    positions = await get_open_positions(account)
    
    # Create invocation record
    invocation = Invocation(
        model_id=account.id,
        response="",
    )
    session.add(invocation)
    await session.flush()
    
    # Build the prompt
    positions_str = ", ".join(
        f"{p.symbol} {p.position:.4f} {p.sign}" 
        for p in positions
    ) if positions else "None"
    
    enriched_prompt = build_prompt(
        invocation_times=account.invocation_count,
        open_positions=positions_str,
        portfolio_value=f"${portfolio.total_collateral:.2f}",
        all_indicator_data=all_indicator_data,
        available_cash=f"${portfolio.free_collateral:.2f}",
        current_account_value=f"${portfolio.net_usd_value:.2f}",
        current_account_positions=json.dumps([
            {
                "symbol": p.symbol,
                "position": p.position,
                "sign": p.sign,
                "entry_price": p.entry_price,
                "unrealized_pnl": p.unrealized_pnl,
                "liquidation_price": p.liquidation_price,
            }
            for p in positions
        ], indent=2),
    )
    
    logger.info(f"Invoking agent for {account.name}")
    logger.debug(f"Prompt: {enriched_prompt[:500]}...")
    
    # Call Claude with tools
    messages = [{"role": "user", "content": enriched_prompt}]
    full_response = ""
    
    while True:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            tools=TOOLS,
            messages=messages,
        )
        
        # Process response
        for block in response.content:
            if block.type == "text":
                full_response += block.text
            elif block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                
                logger.info(f"Tool call: {tool_name} with input {tool_input}")
                
                # Execute tool
                tool_result = ""
                try:
                    if tool_name == "createPosition":
                        # Execute the trade as requested (no inversion like the original)
                        result = await create_position(
                            account=account,
                            symbol=tool_input["symbol"],
                            side=tool_input["side"],
                            quantity=tool_input["quantity"],
                        )
                        
                        # Record tool call
                        tool_call = ToolCall(
                            invocation_id=invocation.id,
                            tool_call_type=ToolCallType.CREATE_POSITION,
                            tool_metadata=json.dumps({
                                "symbol": tool_input["symbol"],
                                "side": tool_input["side"],
                                "quantity": tool_input["quantity"],
                                "tx_signature": result.get("tx_signature"),
                            }),
                        )
                        session.add(tool_call)
                        
                        tool_result = f"Position opened successfully: {tool_input['side']} {tool_input['quantity']} {tool_input['symbol']} at ~${result.get('price', 0):.2f}. TX: {result.get('tx_signature', 'N/A')}"
                    
                    elif tool_name == "closeAllPositions":
                        result = await close_all_positions(account)
                        
                        # Record tool call
                        tool_call = ToolCall(
                            invocation_id=invocation.id,
                            tool_call_type=ToolCallType.CLOSE_POSITION,
                            tool_metadata=json.dumps({
                                "closed_positions": result.get("closed_positions", []),
                            }),
                        )
                        session.add(tool_call)
                        
                        closed = result.get("closed_positions", [])
                        if closed:
                            tool_result = f"Closed positions in: {', '.join(closed)}"
                        else:
                            tool_result = "No positions to close"
                        logger.info(tool_result)
                
                except Exception as e:
                    tool_result = f"Error executing tool: {str(e)}"
                    logger.error(f"Tool execution error: {e}")
                
                # Add tool result to messages for continuation
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_result,
                    }]
                })
        
        # Check if we should continue
        if response.stop_reason == "end_turn":
            break
        elif response.stop_reason != "tool_use":
            break
    
    # Update invocation with response
    invocation.response = full_response.strip()
    
    # Update model invocation count
    result = await session.execute(
        select(Model).where(Model.id == account.id)
    )
    model = result.scalar_one_or_none()
    if model:
        model.invocation_count += 1
    
    await session.commit()
    
    return full_response


async def run_agent_loop(session: AsyncSession):
    """Run the agent for all models."""
    result = await session.execute(select(Model))
    models = result.scalars().all()
    
    for model in models:
        try:
            account = Account(
                private_key=model.solana_private_key,
                name=model.name,
                model_name=model.claude_model_name,
                invocation_count=model.invocation_count,
                id=model.id,
            )
            
            response = await invoke_agent(account, session)
            logger.info(f"Agent {model.name} response: {response[:200]}...")
        
        except Exception as e:
            logger.error(f"Error invoking agent for {model.name}: {e}", exc_info=True)
