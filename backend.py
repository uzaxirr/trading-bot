import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import anthropic
from fastapi import FastAPI, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from models import async_session, Invocation, PortfolioSize, Model
from markets import MARKETS

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Trading Agent API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for performance data
_performance_cache: List[dict] = []
_performance_last_updated: Optional[datetime] = None

# Cache for invocations
_invocations_cache: List[dict] = []
_invocations_last_updated: Optional[datetime] = None
_invocations_refresh_in_flight = False


class PerformanceResponse(BaseModel):
    data: List[dict]
    lastUpdated: Optional[datetime]


class InvocationsResponse(BaseModel):
    data: List[dict]
    lastUpdated: Optional[datetime]
    stale: bool = False


async def get_db():
    async with async_session() as session:
        yield session


@app.get("/performance", response_model=PerformanceResponse)
async def get_performance(session: AsyncSession = Depends(get_db)):
    global _performance_cache, _performance_last_updated
    
    # Return cached data if fresh (within 5 minutes)
    if _performance_last_updated and _performance_last_updated > datetime.utcnow() - timedelta(minutes=5):
        return PerformanceResponse(
            data=_performance_cache,
            lastUpdated=_performance_last_updated
        )
    
    # Fetch fresh data
    result = await session.execute(
        select(PortfolioSize)
        .options(selectinload(PortfolioSize.model))
        .order_by(PortfolioSize.created_at.asc())
    )
    portfolio_sizes = result.scalars().all()
    
    _performance_cache = [
        {
            "id": ps.id,
            "modelId": ps.model_id,
            "netPortfolio": ps.net_portfolio,
            "createdAt": ps.created_at.isoformat(),
            "model": {"name": ps.model.name} if ps.model else None,
        }
        for ps in portfolio_sizes
    ]
    _performance_last_updated = datetime.utcnow()
    
    return PerformanceResponse(
        data=_performance_cache,
        lastUpdated=_performance_last_updated
    )


async def refresh_invocations(session: AsyncSession, take: int):
    global _invocations_cache, _invocations_last_updated, _invocations_refresh_in_flight
    
    if _invocations_refresh_in_flight:
        return
    
    _invocations_refresh_in_flight = True
    try:
        safe_take = min(max(take, 1), 200)
        
        result = await session.execute(
            select(Invocation)
            .options(
                selectinload(Invocation.model),
                selectinload(Invocation.tool_calls)
            )
            .order_by(Invocation.created_at.desc())
            .limit(safe_take)
        )
        invocations = result.scalars().all()
        
        _invocations_cache = [
            {
                "id": inv.id,
                "modelId": inv.model_id,
                "response": inv.response,
                "createdAt": inv.created_at.isoformat(),
                "model": {"name": inv.model.name} if inv.model else None,
                "toolCalls": [
                    {
                        "toolCallType": tc.tool_call_type.value,
                        "metadata": tc.tool_metadata,
                        "createdAt": tc.created_at.isoformat(),
                    }
                    for tc in sorted(inv.tool_calls, key=lambda x: x.created_at)
                ],
            }
            for inv in invocations
        ]
        _invocations_last_updated = datetime.utcnow()
    except Exception as e:
        logger.error(f"Error refreshing invocations: {e}")
    finally:
        _invocations_refresh_in_flight = False


@app.get("/invocations", response_model=InvocationsResponse)
async def get_invocations(
    limit: int = Query(default=30, ge=1, le=200),
    session: AsyncSession = Depends(get_db)
):
    global _invocations_cache, _invocations_last_updated
    
    is_fresh = (
        _invocations_last_updated and 
        _invocations_last_updated > datetime.utcnow() - timedelta(minutes=2)
    )
    
    # If no cache, fetch data
    if not _invocations_cache:
        await refresh_invocations(session, limit)
        return InvocationsResponse(
            data=_invocations_cache[:limit],
            lastUpdated=_invocations_last_updated
        )
    
    # Return cached data
    response = InvocationsResponse(
        data=_invocations_cache[:limit],
        lastUpdated=_invocations_last_updated,
        stale=not is_fresh
    )
    
    # Trigger background refresh if stale
    if not is_fresh and not _invocations_refresh_in_flight:
        # Fire and forget refresh
        asyncio.create_task(refresh_invocations(session, limit))
    
    return response


@app.get("/health")
async def health_check():
    return {"status": "ok"}


# ============================================================================
# CHAT ENDPOINT
# ============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    response: str
    portfolio: Optional[dict] = None
    positions: Optional[List[dict]] = None


# Chat system prompt
CHAT_SYSTEM_PROMPT = """You are an AI trading assistant for the Drift Protocol on Solana. You can execute trades directly on behalf of the user.

Current Account Status:
{account_status}

Available Markets: {markets}

You have access to trading tools:
- openPosition: Open a new LONG or SHORT position
- closePosition: Close a specific position by symbol
- closeAllPositions: Close all open positions

When the user asks you to trade:
1. Confirm the trade details with them first if they're vague
2. Execute the trade using the appropriate tool
3. Report the result

Be concise, helpful, and focus on actionable insights. Use bullet points and clear formatting.
Always confirm before executing large trades or closing profitable positions."""

# Trading tools for chat
CHAT_TOOLS = [
    {
        "name": "openPosition",
        "description": "Open a new trading position. Use this when the user wants to go LONG or SHORT on a market.",
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
                    "description": "The direction of the trade"
                },
                "quantity": {
                    "type": "number",
                    "description": "The quantity to trade"
                }
            },
            "required": ["symbol", "side", "quantity"]
        }
    },
    {
        "name": "closePosition",
        "description": "Close a specific position. Use this when the user wants to close a position in a specific market.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "enum": list(MARKETS.keys()),
                    "description": "The market symbol to close (e.g., SOL, BTC, ETH)"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "closeAllPositions",
        "description": "Close all open positions. Use this when the user wants to exit all trades.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]


async def get_account_status(session: AsyncSession) -> dict:
    """Get current account status for chat context."""
    try:
        # Get the first model (trading bot)
        result = await session.execute(select(Model).limit(1))
        model = result.scalar_one_or_none()
        
        if not model:
            return {
                "status": "No trading bot configured",
                "portfolio": None,
                "positions": []
            }
        
        # Import here to avoid circular imports
        from trading import Account, DriftClientManager
        from portfolio import get_portfolio
        from positions import get_open_positions
        
        account = Account(
            private_key=model.solana_private_key,
            name=model.name,
            model_name=model.claude_model_name,
            invocation_count=model.invocation_count,
            id=model.id,
        )
        
        try:
            portfolio = await get_portfolio(account)
            positions = await get_open_positions(account)
            
            return {
                "status": "Connected",
                "portfolio": {
                    "total_collateral": f"${portfolio.total_collateral:.2f}",
                    "free_collateral": f"${portfolio.free_collateral:.2f}",
                    "net_value": f"${portfolio.net_usd_value:.2f}",
                    "leverage": f"{portfolio.leverage:.2f}x",
                    "health": f"{portfolio.health}%"
                },
                "positions": [
                    {
                        "symbol": p.symbol,
                        "side": p.sign,
                        "size": f"{abs(p.position):.6f}",
                        "entry_price": f"${p.entry_price:.2f}",
                        "unrealized_pnl": f"${p.unrealized_pnl:.2f}",
                        "liquidation_price": f"${p.liquidation_price:.2f}" if p.liquidation_price > 0 else "N/A"
                    }
                    for p in positions
                ]
            }
        except Exception as e:
            logger.warning(f"Failed to get account status: {e}")
            return {
                "status": f"Error: {str(e)}",
                "portfolio": None,
                "positions": []
            }
    except Exception as e:
        logger.error(f"Error getting account status: {e}")
        return {
            "status": "Error fetching account",
            "portfolio": None,
            "positions": []
        }


async def execute_chat_tool(tool_name: str, tool_input: dict, session: AsyncSession) -> str:
    """Execute a trading tool and return the result."""
    try:
        # Get the trading bot model
        result = await session.execute(select(Model).limit(1))
        model = result.scalar_one_or_none()
        
        if not model:
            return "Error: No trading bot configured"
        
        from trading import Account, create_position, close_position as trading_close_position, close_all_positions
        
        account = Account(
            private_key=model.solana_private_key,
            name=model.name,
            model_name=model.claude_model_name,
            invocation_count=model.invocation_count,
            id=model.id,
        )
        
        if tool_name == "openPosition":
            symbol = tool_input.get("symbol")
            side = tool_input.get("side")
            quantity = tool_input.get("quantity")
            
            logger.info(f"Chat executing: Open {side} {quantity} {symbol}")
            result = await create_position(account, symbol, side, quantity)
            
            if result.get("success"):
                tx = result.get("tx_signature", "")
                return f"✅ Successfully opened {side} position for {quantity} {symbol}. Transaction: {tx[:20]}..."
            else:
                return f"❌ Failed to open position: {result.get('error', 'Unknown error')}"
        
        elif tool_name == "closePosition":
            symbol = tool_input.get("symbol")
            
            logger.info(f"Chat executing: Close {symbol} position")
            result = await trading_close_position(account, symbol)
            
            if result.get("success"):
                tx = result.get("tx_signature", "")
                return f"✅ Successfully closed {symbol} position. Transaction: {tx[:20]}..."
            else:
                return f"❌ Failed to close position: {result.get('error', 'Unknown error')}"
        
        elif tool_name == "closeAllPositions":
            logger.info("Chat executing: Close all positions")
            result = await close_all_positions(account)
            
            if result.get("success"):
                closed = result.get("closed_positions", [])
                return f"✅ Successfully closed all positions: {', '.join(closed) if closed else 'None to close'}"
            else:
                return f"❌ Failed to close positions: {result.get('error', 'Unknown error')}"
        
        else:
            return f"Unknown tool: {tool_name}"
    
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return f"❌ Error executing trade: {str(e)}"


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, session: AsyncSession = Depends(get_db)):
    """Chat with the trading agent - supports tool use for trading."""
    try:
        # Get account status for context
        account_status = await get_account_status(session)
        
        # Format account status for system prompt
        if account_status["portfolio"]:
            status_text = f"""
Portfolio:
- Total Collateral: {account_status['portfolio']['total_collateral']}
- Free Collateral: {account_status['portfolio']['free_collateral']}
- Net Value: {account_status['portfolio']['net_value']}
- Leverage: {account_status['portfolio']['leverage']}
- Health: {account_status['portfolio']['health']}

Open Positions:"""
            if account_status["positions"]:
                for pos in account_status["positions"]:
                    status_text += f"\n- {pos['symbol']} {pos['side']}: {pos['size']} @ {pos['entry_price']} (PnL: {pos['unrealized_pnl']})"
            else:
                status_text += "\n- No open positions"
        else:
            status_text = account_status["status"]
        
        # Build system prompt
        system_prompt = CHAT_SYSTEM_PROMPT.format(
            account_status=status_text,
            markets=", ".join(MARKETS.keys())
        )
        
        # Create Anthropic client
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        # Build messages
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        full_response = ""
        tool_results = []
        
        # Call Claude with tools - loop until no more tool calls
        while True:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                system=system_prompt,
                tools=CHAT_TOOLS,
                messages=messages,
            )
            
            # Process response
            has_tool_use = False
            assistant_content = []
            
            for block in response.content:
                if block.type == "text":
                    full_response += block.text
                    assistant_content.append(block)
                elif block.type == "tool_use":
                    has_tool_use = True
                    assistant_content.append(block)
                    
                    tool_name = block.name
                    tool_input = block.input
                    tool_id = block.id
                    
                    logger.info(f"Chat tool call: {tool_name} with input {tool_input}")
                    
                    # Execute the tool
                    tool_result = await execute_chat_tool(tool_name, tool_input, session)
                    tool_results.append({
                        "tool": tool_name,
                        "input": tool_input,
                        "result": tool_result
                    })
                    
                    # Add to messages for continuation
                    messages.append({"role": "assistant", "content": assistant_content})
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": tool_result,
                        }]
                    })
                    assistant_content = []  # Reset for next iteration
            
            # Check if we should continue
            if response.stop_reason == "end_turn" or not has_tool_use:
                break
            
            # If there was tool use but no end_turn, continue the loop
            if response.stop_reason == "tool_use":
                continue
            else:
                break
        
        # Refresh account status after any trades
        if tool_results:
            account_status = await get_account_status(session)
        
        return ChatResponse(
            response=full_response.strip(),
            portfolio=account_status.get("portfolio"),
            positions=account_status.get("positions")
        )
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            response=f"Sorry, I encountered an error: {str(e)}",
            portfolio=None,
            positions=None
        )


@app.get("/status")
async def get_status(session: AsyncSession = Depends(get_db)):
    """Get current trading bot status."""
    account_status = await get_account_status(session)
    return account_status


# ============================================================================
# CLOSE POSITION ENDPOINT
# ============================================================================

class ClosePositionRequest(BaseModel):
    symbol: str


class ClosePositionResponse(BaseModel):
    success: bool
    message: str
    tx_signature: Optional[str] = None


@app.post("/close-position", response_model=ClosePositionResponse)
async def close_position(request: ClosePositionRequest, session: AsyncSession = Depends(get_db)):
    """Close a specific position."""
    try:
        # Get the trading bot model
        result = await session.execute(select(Model).limit(1))
        model = result.scalar_one_or_none()
        
        if not model:
            return ClosePositionResponse(
                success=False,
                message="No trading bot configured"
            )
        
        from trading import Account, close_position as execute_close
        
        account = Account(
            private_key=model.solana_private_key,
            name=model.name,
            model_name=model.claude_model_name,
            invocation_count=model.invocation_count,
            id=model.id,
        )
        
        # Execute the close
        result = await execute_close(account, request.symbol)
        
        if result.get("success"):
            return ClosePositionResponse(
                success=True,
                message=f"Successfully closed {request.symbol} position",
                tx_signature=result.get("tx_signature")
            )
        else:
            return ClosePositionResponse(
                success=False,
                message=result.get("error", "Failed to close position")
            )
    
    except Exception as e:
        logger.error(f"Close position error: {e}")
        return ClosePositionResponse(
            success=False,
            message=f"Error: {str(e)}"
        )


# ============================================================================
# TRADE HISTORY ENDPOINT
# ============================================================================

@app.get("/trade-history")
async def get_trade_history(
    limit: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_db)
):
    """Get trade history from tool calls."""
    try:
        from models import ToolCall, ToolCallType
        
        result = await session.execute(
            select(ToolCall)
            .options(selectinload(ToolCall.invocation))
            .order_by(ToolCall.created_at.desc())
            .limit(limit)
        )
        tool_calls = result.scalars().all()
        
        trades = []
        for tc in tool_calls:
            try:
                import json
                metadata = json.loads(tc.tool_metadata) if tc.tool_metadata else {}
                
                trade = {
                    "id": tc.id,
                    "type": tc.tool_call_type.value,
                    "symbol": metadata.get("symbol", ""),
                    "side": metadata.get("side", ""),
                    "quantity": metadata.get("quantity", 0),
                    "tx_signature": metadata.get("tx_signature", ""),
                    "created_at": tc.created_at.isoformat(),
                }
                
                # For close position trades
                if tc.tool_call_type == ToolCallType.CLOSE_POSITION:
                    trade["closed_positions"] = metadata.get("closed_positions", [])
                
                trades.append(trade)
            except Exception:
                continue
        
        return {"trades": trades}
    
    except Exception as e:
        logger.error(f"Trade history error: {e}")
        return {"trades": [], "error": str(e)}


# Serve frontend
@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")


