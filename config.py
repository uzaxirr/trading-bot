import os
from dotenv import load_dotenv

load_dotenv()

# Solana RPC Configuration
RPC_URL = os.getenv("RPC_URL", "https://api.mainnet-beta.solana.com")
DRIFT_ENV = os.getenv("DRIFT_ENV", "mainnet")  # "mainnet" or "devnet"

# Anthropic Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://localhost/trading_agent")

# Agent Configuration
INVOCATION_INTERVAL_SECONDS = 3000  # 15 seconds for testing

# Birdeye API (for historical price data)
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")
