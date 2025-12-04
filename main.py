import asyncio
import logging
import os
import signal
import sys

import uvicorn

from config import INVOCATION_INTERVAL_SECONDS
from models import init_db, async_session, Model
from agent import run_agent_loop
from trading import DriftClientManager, Account
from backend import app
from sqlalchemy import select

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Flag to control graceful shutdown
shutdown_event = asyncio.Event()

# Autonomous mode: set AUTONOMOUS=1 to enable self-operating agent
AUTONOMOUS_MODE = os.getenv("AUTONOMOUS", "0") == "1"
AUTONOMOUS_INTERVAL = int(os.getenv("AUTONOMOUS_INTERVAL", "60"))  # seconds


async def agent_scheduler():
    """Run the basic agent on a schedule (non-autonomous mode)."""
    logger.info(f"Starting basic agent scheduler (interval: {INVOCATION_INTERVAL_SECONDS}s)")
    
    while not shutdown_event.is_set():
        try:
            async with async_session() as session:
                await run_agent_loop(session)
        except Exception as e:
            logger.error(f"Error in agent loop: {e}", exc_info=True)
        
        # Wait for interval or shutdown
        try:
            await asyncio.wait_for(
                shutdown_event.wait(),
                timeout=INVOCATION_INTERVAL_SECONDS
            )
        except asyncio.TimeoutError:
            pass  # Normal timeout, continue loop


async def autonomous_agent_scheduler():
    """Run the autonomous self-operating agent."""
    from autonomous_agent import AutonomousAgent
    
    logger.info("🤖 Starting AUTONOMOUS TRADING AGENT")
    logger.info(f"📊 Cycle interval: {AUTONOMOUS_INTERVAL}s")
    logger.info("=" * 60)
    
    # Get account from database
    async with async_session() as session:
        result = await session.execute(select(Model).limit(1))
        model = result.scalar_one_or_none()
        
        if not model:
            logger.error("No trading bot configured in database!")
            return
        
        account = Account(
            private_key=model.solana_private_key,
            name=model.name,
            model_name=model.claude_model_name,
            invocation_count=model.invocation_count,
            id=model.id,
        )
    
    agent = AutonomousAgent()
    
    while not shutdown_event.is_set():
        try:
            await agent.run_cycle(account)
        except Exception as e:
            logger.error(f"Error in autonomous cycle: {e}", exc_info=True)
        
        # Wait for interval or shutdown
        try:
            await asyncio.wait_for(
                shutdown_event.wait(),
                timeout=AUTONOMOUS_INTERVAL
            )
        except asyncio.TimeoutError:
            pass  # Normal timeout, continue loop
    
    agent.stop()


async def run_server():
    """Run the FastAPI server."""
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=3000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    # Run until shutdown
    await server.serve()


async def cleanup():
    """Cleanup resources on shutdown."""
    logger.info("Cleaning up resources...")
    await DriftClientManager.close_all()
    logger.info("Cleanup complete")


def handle_shutdown(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    shutdown_event.set()


async def main():
    """Main entry point."""
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    logger.info("=" * 60)
    logger.info("AI Trading Agent - Drift Protocol (Solana)")
    if AUTONOMOUS_MODE:
        logger.info("🤖 MODE: AUTONOMOUS (Self-Operating)")
    else:
        logger.info("📋 MODE: BASIC (Scheduled Invocations)")
    logger.info("=" * 60)
    
    logger.info("Initializing database...")
    await init_db()
    
    logger.info("Starting services...")
    
    try:
        # Choose which agent mode to run
        if AUTONOMOUS_MODE:
            # Run autonomous agent with API server
            await asyncio.gather(
                autonomous_agent_scheduler(),
                run_server(),
                return_exceptions=True,
            )
        else:
            # Run basic agent scheduler with API server
            await asyncio.gather(
                agent_scheduler(),
                run_server(),
                return_exceptions=True,
            )
    finally:
        await cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
