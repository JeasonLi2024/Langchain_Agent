
import os
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from starlette.types import Message
import json
from contextlib import asynccontextmanager
from core.db import PostgresPool

# python server.py
# API 文档 : http://localhost:8000/docs
# 学生智能体调试 : http://localhost:8000/student/playground
# 发布者智能体调试 : http://localhost:8000/publisher/playground

# 0. Load Config (Prioritize Local Env)
# Must load before setup_django to avoid parent .env pollution
from core.config import Config

# 1. Setup Django
# This is crucial because the agents rely on Django models and settings
from core.django_setup import setup_django
setup_django()

# 2. Import Agents
# Student/Main Agent
from graph.main_agent import master_app as student_agent
# Publisher Agent
from graph.publisher_main_agent import publisher_main_app as publisher_agent
# QA Agent
from graph.qa_agent import qa_app as qa_agent

# 3. Define Lifespan for Connection Pool Management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Open Database Pool
    await PostgresPool.open_pool()
    
    # CRITICAL: Swap placeholder checkpointer with real AsyncPostgresSaver
    # This must be done inside the loop
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    
    pool = PostgresPool.get_or_create_pool()
    real_checkpointer = AsyncPostgresSaver(pool)
    
    # 1. Swap for Student Agent
    # Note: CompiledGraph.checkpointer is the attribute.
    student_agent.checkpointer = real_checkpointer
    
    # 2. Swap for Publisher Agent
    publisher_agent.checkpointer = real_checkpointer
    
    print("Swapped In-Memory Checkpointers with AsyncPostgresSaver.")
    
    yield
    # Shutdown: Close Database Pool
    await PostgresPool.close_pool()

# 4. Create FastAPI app
app = FastAPI(
    title="LangChain Agent Server",
    version="1.0",
    description="API server for Student and Publisher agents",
    lifespan=lifespan,
)

# 5. Add CORS
# Allow all origins for development/production (configure as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def per_req_config_modifier(config, request):
    if "configurable" not in config:
        config["configurable"] = {}
    
    # Priority 1: Check Headers (Best Practice - Lightweight & Standard)
    # Support 'X-Thread-ID' or 'Thread-ID'
    header_thread_id = request.headers.get("x-thread-id") or request.headers.get("thread-id")
    if header_thread_id:
        config["configurable"]["thread_id"] = header_thread_id
        return config

    # Priority 2: Check Query Parameters (Convenient for testing)
    query_thread_id = request.query_params.get("thread_id")
    if query_thread_id:
        config["configurable"]["thread_id"] = query_thread_id
        return config

    # Priority 3: Fallback to Middleware-injected state (For JSON body support)
    if "thread_id" not in config["configurable"]:
        # Try to get from state injected by middleware
        if hasattr(request.state, "thread_id"):
            config["configurable"]["thread_id"] = request.state.thread_id
            
    return config

# --- Custom History Management Endpoints ---

@app.get("/student/history/{thread_id}")
async def get_chat_history(thread_id: str):
    """
    Get chat history for a specific thread.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        # Fetch state from the graph
        state = await student_agent.aget_state(config)
        if not state.values:
            return {"messages": []}
            
        messages = state.values.get("messages", [])
        # Serialize messages
        history = []
        for msg in messages:
            history.append({
                "type": msg.type,
                "content": msg.content,
                # Add other fields if needed
            })
        return {"messages": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/student/history/{thread_id}")
async def delete_chat_history(thread_id: str):
    """
    Delete chat history for a specific thread (Hard Delete).
    """
    pool = PostgresPool.get_or_create_pool()
    try:
        async with pool.connection() as conn:
            # Delete from checkpoints and writes
            # Note: Table names depend on AsyncPostgresSaver default setup.
            # Usually 'checkpoints', 'checkpoint_blobs', 'checkpoint_writes'
            # We filter by thread_id
            
            # Using transactions to ensure atomicity
            async with conn.transaction():
                await conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
                await conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
                await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
                
        return {"status": "success", "message": f"History for thread {thread_id} deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete history: {str(e)}")

# 5. Add Routes
# Student Agent Endpoint
add_routes(
    app,
    student_agent,
    path="/student",
    playground_type="default",
    per_req_config_modifier=per_req_config_modifier,
)

# Publisher Agent Endpoint
add_routes(
    app,
    publisher_agent,
    path="/publisher",
    playground_type="default",
    per_req_config_modifier=per_req_config_modifier,
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
