import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not, check you .env file")


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
    return engine.connect()

def test_connection() -> bool: 
    """
    Tests whether the database is reachable
    Returns True if connected, else False.
    """
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