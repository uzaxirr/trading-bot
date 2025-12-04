import asyncio
import logging
import signal
import sys

import uvicorn

from config import INVOCATION_INTERVAL_SECONDS
from models import init_db, async_session
from agent import run_agent_loop
from trading import DriftClientManager
from backend import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Flag to control graceful shutdown
shutdown_event = asyncio.Event()


async def agent_scheduler():
    """Run the agent on a schedule."""
    logger.info(f"Starting agent scheduler (interval: {INVOCATION_INTERVAL_SECONDS}s)")
    
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
    logger.info("=" * 60)
    
    logger.info("Initializing database...")
    await init_db()
    
    logger.info("Starting services...")
    
    try:
        # Run both the agent scheduler and the API server concurrently
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
