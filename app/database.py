import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

# Don't raise at import time — raise when connection is actually attempted
# This allows the app to start and give better error messages
if not DATABASE_URL:
    logger.error(
        "DATABASE_URL environment variable is not set. "
        "Check Railway Variables tab or your .env file."
    )
    engine = None

else:
    # Creating the connection pool
    # pool_size: no of permanent connection to keep on.
    # max_overflow: extra connection allowed under heavy load.
    # pool_pre_cooling: test connection before use(handle docker restarts gracefully)
    engine = create_engine(
        DATABASE_URL,
        pool_size = 5,
        max_overflow = 10,
        pool_pre_ping = True,
        echo = False # Need to learn!!
    )

def get_connection():
    """Returns a Database connection from the pool"""
    if engine is None:
        raise RuntimeError(
            "Database engine is not initialized"
            "Database_URL environmet variable is missing"
        )
    return engine.connect()

def test_connection() -> bool: 
    """
    Tests whether the database is reachable
    Returns True if connected, else False.
    """
    if engine is None:
        logger.error("Cannot test connection — DATABASE_URL not set")
        return False

    try:
        with get_connection() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Database connection test: SUCCESS")
        return True
    
    except OperationalError as e:
        logger.error(f"Database connection test: Failed - {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    if success:
        print("Connected to PostgreSQL successfully")
    else:
        print("Could not connect. Is Docker running? run command: docker ps")