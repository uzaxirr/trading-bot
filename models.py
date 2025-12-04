import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import List

from sqlalchemy import String, Integer, ForeignKey, DateTime, Enum, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.ext.asyncio import AsyncAttrs, create_async_engine, async_sessionmaker

from config import DATABASE_URL


class Base(AsyncAttrs, DeclarativeBase):
    pass


class ToolCallType(PyEnum):
    CREATE_POSITION = "CREATE_POSITION"
    CLOSE_POSITION = "CLOSE_POSITION"


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
