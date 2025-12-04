# AI Trading Agent - Drift Protocol (Solana)

A Python AI trading agent that uses Claude to make trading decisions on Drift Protocol, Solana's leading perpetual futures DEX.

## Features

- **AI-Powered Trading**: Uses Claude to analyze market data and execute trades
- **Drift Protocol Integration**: Trade perpetual futures on Solana
- **Technical Indicators**: EMA20, MACD calculations for market analysis
- **Multiple Markets**: Supports SOL, BTC, ETH, WIF, JUP, DOGE, BONK, HYPE
- **REST API**: FastAPI backend for monitoring performance and invocations
- **Async Architecture**: Built with asyncio for efficient concurrent operations

## Prerequisites

- Python 3.10+
- PostgreSQL database
- Anthropic API key
- Solana wallet with funds (for trading)
- Solana RPC endpoint (Helius, QuickNode, or similar)

## Installation

1. **Create and activate virtual environment:**
   ```bash
   cd python
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp env.template .env
   # Edit .env with your actual values
   ```

4. **Create the PostgreSQL database:**
   ```bash
   createdb trading_agent
   ```

## Configuration

Edit `.env` with your credentials:

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key | Yes |
| `RPC_URL` | Solana RPC endpoint | Yes |
| `DRIFT_ENV` | `mainnet` or `devnet` | Yes |
| `DATABASE_URL` | PostgreSQL connection string | Yes |

## Database Setup

The database schema is auto-created on first run. To add a trading model:

```sql
INSERT INTO models (id, name, claude_model_name, solana_private_key)
VALUES (
    'your-uuid-here',
    'My Trading Bot',
    'claude-sonnet-4-20250514',
    'your-base58-solana-private-key'
);
```

**⚠️ Security Warning**: Store private keys securely. Consider using environment variables or a secrets manager in production.

## Running

```bash
source venv/bin/activate
python main.py
```

This will:
1. Initialize the database schema
2. Start the FastAPI server on port 3000
3. Run the trading agent every 5 minutes

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/performance` | GET | Get portfolio performance history |
| `/invocations` | GET | Get agent invocation history |

## Project Structure

```
python/
├── main.py           # Entry point - runs server and agent scheduler
├── config.py         # Configuration settings
├── models.py         # SQLAlchemy database models
├── markets.py        # Drift perp market definitions
├── indicators.py     # Technical indicators (EMA, MACD)
├── prompt.py         # AI prompt template
├── portfolio.py      # Portfolio/account data
├── positions.py      # Open positions management
├── stock_data.py     # Price and indicator data
├── trading.py        # Order execution via Drift SDK
├── agent.py          # Main agent invocation logic
├── backend.py        # FastAPI REST API
├── requirements.txt  # Python dependencies
└── env.template      # Environment variables template
```

## Available Markets

| Symbol | Drift Market | Max Leverage |
|--------|--------------|--------------|
| SOL | SOL-PERP | 20x |
| BTC | BTC-PERP | 20x |
| ETH | ETH-PERP | 20x |
| DOGE | DOGE-PERP | 10x |
| WIF | WIF-PERP | 10x |
| JUP | JUP-PERP | 10x |
| BONK | 1MBONK-PERP | 10x |
| HYPE | HYPE-PERP | 10x |

## How It Works

1. **Every 5 minutes**, the agent:
   - Fetches oracle prices from Drift Protocol
   - Gets historical price data for technical analysis
   - Calculates EMA20 and MACD indicators
   - Gets current portfolio value and open positions
   - Sends all data to Claude with trading tools
   - Claude analyzes and decides on trades
   - Executes orders via Drift SDK
   - Records invocations and tool calls

2. **Tools available to Claude**:
   - `createPosition`: Open a new perp position (LONG/SHORT)
   - `closeAllPositions`: Close all open positions

## Drift Protocol

[Drift Protocol](https://www.drift.trade/) is the leading perpetual futures DEX on Solana, offering:
- Up to 20x leverage on major assets
- Deep liquidity via JIT auctions
- Sub-second finality
- Low fees (~0.05% maker, ~0.1% taker)

## Development

### Adding New Markets

Edit `markets.py` to add new market definitions:

```python
MARKETS["NEW_TOKEN"] = Market(
    market_index=XX,  # From Drift's market list
    symbol="NEW_TOKEN-PERP",
    base_asset_symbol="NEW_TOKEN",
    leverage=10,
)
```

### Customizing the Agent

Modify `prompt.py` to change the trading strategy or add new instructions.

## Disclaimer

⚠️ **This is experimental software for educational purposes.**

- Trading cryptocurrency derivatives involves substantial risk
- Past performance does not guarantee future results
- Only trade with funds you can afford to lose
- This software makes real trades - use at your own risk

## License

MIT
