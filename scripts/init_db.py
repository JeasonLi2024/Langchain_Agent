
import asyncio
import sys
import os

print("Script started...")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from core.db import PostgresPool

async def main():
    print("Connecting to PostgreSQL to initialize tables...")
    
    # Ensure pool is created
    pool = PostgresPool.get_or_create_pool()
    
    # Use context manager to ensure connection is opened/ready
    async with pool: 
        checkpointer = AsyncPostgresSaver(pool)
        print("Running checkpointer.setup()...")
        await checkpointer.setup()
        print("Successfully initialized checkpoint tables.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Failed to initialize tables: {e}")
        sys.exit(1)
