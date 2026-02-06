import os
import psycopg
from psycopg_pool import AsyncConnectionPool
from core.config import Config

class PostgresPool:
    """
    Singleton Async PostgreSQL Connection Pool.
    
    Design for Concurrency:
    - LangGraph agents typically hold DB connections only during state read (start) and write (end).
    - During LLM inference (which takes most of the time), the connection is released.
    - Therefore, a pool size of 50-100 can often support 1000+ concurrent users, 
      provided the database server itself has enough max_connections.
    """
    _pool: AsyncConnectionPool = None

    @classmethod
    def get_or_create_pool(cls) -> AsyncConnectionPool:
        if cls._pool is None:
            # Optimal configuration for high concurrency (1000 users)
            # Assuming Postgres 'max_connections' is set to >= 200 on the server.
            # If running multiple workers, divide this number by the number of workers.
            cls._pool = AsyncConnectionPool(
                conninfo=Config.CHECKPOINT_DB_URI,
                max_size=80,  # Increased to support higher concurrency (Requires Postgres max_connections > 80)
                open=False,   # Don't open immediately to avoid event loop issues at import time
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                }
            )
        return cls._pool

    @classmethod
    async def open_pool(cls):
        """Explicitly open the pool. Call this on app startup."""
        pool = cls.get_or_create_pool()
        await pool.open()
        print(f"PostgreSQL Connection Pool Opened (max_size={pool.max_size}).")

    @classmethod
    async def close_pool(cls):
        """Close the pool. Call this on app shutdown."""
        if cls._pool:
            await cls._pool.close()
            print("PostgreSQL Connection Pool Closed.")
