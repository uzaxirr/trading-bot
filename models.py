import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import List, Optional

from sqlalchemy import String, Integer, Float, Boolean, ForeignKey, DateTime, Enum, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.ext.asyncio import AsyncAttrs, create_async_engine, async_sessionmaker

from config import DATABASE_URL


class Base(AsyncAttrs, DeclarativeBase):
    pass


class ToolCallType(PyEnum):
    CREATE_POSITION = "CREATE_POSITION"
    CLOSE_POSITION = "CLOSE_POSITION"


class StrategyStatus(PyEnum):
    DRAFT = "DRAFT"           # Just created, not tested
    TESTING = "TESTING"       # Currently being backtested
    APPROVED = "APPROVED"     # Passed backtest, ready to deploy
    ACTIVE = "ACTIVE"         # Currently trading
    PAUSED = "PAUSED"         # Temporarily disabled
    RETIRED = "RETIRED"       # Permanently disabled due to poor performance


class ActivityType(PyEnum):
    MARKET_ANALYSIS = "MARKET_ANALYSIS"
    STRATEGY_CREATED = "STRATEGY_CREATED"
    BACKTEST_STARTED = "BACKTEST_STARTED"
    BACKTEST_COMPLETED = "BACKTEST_COMPLETED"
    STRATEGY_DEPLOYED = "STRATEGY_DEPLOYED"
    STRATEGY_RETIRED = "STRATEGY_RETIRED"
    TRADE_SIGNAL = "TRADE_SIGNAL"
    TRADE_EXECUTED = "TRADE_EXECUTED"
    LEARNING_UPDATE = "LEARNING_UPDATE"
    ERROR = "ERROR"


class Model(Base):
    """Trading model/agent configuration."""
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    claude_model_name: Mapped[str] = mapped_column(String(255))
    # Solana private key (base58, JSON array, or base64 encoded)
    solana_private_key: Mapped[str] = mapped_column(Text)
    invocation_count: Mapped[int] = mapped_column(Integer, default=0)

    invocations: Mapped[List["Invocation"]] = relationship(back_populates="model", cascade="all, delete-orphan")
    portfolio_sizes: Mapped[List["PortfolioSize"]] = relationship(back_populates="model", cascade="all, delete-orphan")


class Invocation(Base):
    """Record of an agent invocation."""
    __tablename__ = "invocations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id: Mapped[str] = mapped_column(String(36), ForeignKey("models.id"), index=True)
    response: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    model: Mapped["Model"] = relationship(back_populates="invocations")
    tool_calls: Mapped[List["ToolCall"]] = relationship(back_populates="invocation", cascade="all, delete-orphan")


class ToolCall(Base):
    """Record of a tool call made during an invocation."""
    __tablename__ = "tool_calls"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    invocation_id: Mapped[str] = mapped_column(String(36), ForeignKey("invocations.id"), index=True)
    tool_call_type: Mapped[ToolCallType] = mapped_column(Enum(ToolCallType))
    tool_metadata: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    invocation: Mapped["Invocation"] = relationship(back_populates="tool_calls")


class PortfolioSize(Base):
    """Historical portfolio value tracking."""
    __tablename__ = "portfolio_sizes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id: Mapped[str] = mapped_column(String(36), ForeignKey("models.id"), index=True)
    net_portfolio: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    model: Mapped["Model"] = relationship(back_populates="portfolio_sizes")


class Strategy(Base):
    """Trading strategy created by the autonomous agent."""
    __tablename__ = "strategies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    
    # Strategy configuration (JSON)
    symbols: Mapped[str] = mapped_column(Text)  # JSON array: ["SOL", "ETH"]
    side: Mapped[str] = mapped_column(String(10))  # "LONG", "SHORT", "BOTH"
    timeframe: Mapped[str] = mapped_column(String(10), default="5m")  # "1m", "5m", "15m", "1h", "4h"
    
    # Entry/Exit conditions (JSON)
    entry_conditions: Mapped[str] = mapped_column(Text)  # JSON with indicator rules
    exit_conditions: Mapped[str] = mapped_column(Text)   # JSON with exit rules
    
    # Risk management
    position_size_usd: Mapped[float] = mapped_column(Float, default=5.0)
    stop_loss_percent: Mapped[float] = mapped_column(Float, default=5.0)
    take_profit_percent: Mapped[float] = mapped_column(Float, default=10.0)
    max_positions: Mapped[int] = mapped_column(Integer, default=1)
    
    # Status
    status: Mapped[StrategyStatus] = mapped_column(Enum(StrategyStatus), default=StrategyStatus.DRAFT)
    
    # Backtest results
    backtest_win_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    backtest_total_trades: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    backtest_profit_factor: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    backtest_max_drawdown: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    backtest_sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Live performance
    live_trades: Mapped[int] = mapped_column(Integer, default=0)
    live_wins: Mapped[int] = mapped_column(Integer, default=0)
    live_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deployed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    trades: Mapped[List["StrategyTrade"]] = relationship(back_populates="strategy", cascade="all, delete-orphan")


class StrategyTrade(Base):
    """Individual trade executed by a strategy."""
    __tablename__ = "strategy_trades"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    strategy_id: Mapped[str] = mapped_column(String(36), ForeignKey("strategies.id"), index=True)
    
    symbol: Mapped[str] = mapped_column(String(20))
    side: Mapped[str] = mapped_column(String(10))  # "LONG" or "SHORT"
    
    # Entry
    entry_price: Mapped[float] = mapped_column(Float)
    entry_quantity: Mapped[float] = mapped_column(Float)
    entry_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    entry_tx: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Exit (nullable until closed)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exit_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    exit_reason: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # "TAKE_PROFIT", "STOP_LOSS", "SIGNAL", "MANUAL"
    exit_tx: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Results
    pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pnl_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_win: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    
    # Status
    is_open: Mapped[bool] = mapped_column(Boolean, default=True)
    
    strategy: Mapped["Strategy"] = relationship(back_populates="trades")


class AgentActivity(Base):
    """Log of autonomous agent activities and thoughts."""
    __tablename__ = "agent_activities"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    activity_type: Mapped[ActivityType] = mapped_column(Enum(ActivityType))
    title: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    details: Mapped[str] = mapped_column(Text, default="")  # JSON for extra data
    
    # Optional references
    strategy_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    symbol: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# Database engine and session
engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False)


async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session():
    """Get a database session."""
    async with async_session() as session:
        yield session
