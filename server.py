import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes


# python server.py
# API 文档 : http://localhost:8000/docs
# 学生智能体调试 : http://localhost:8000/student/playground
# 发布者智能体调试 : http://localhost:8000/publisher/playground

# 1. Setup Django
# This is crucial because the agents rely on Django models and settings
from core.django_setup import setup_django
setup_django()

# 2. Import Agents
# Student/Main Agent
from graph.main_agent import master_app as student_agent
# Publisher Agent
from graph.publisher_main_agent import publisher_main_app as publisher_agent

# 3. Create FastAPI app
app = FastAPI(
    title="LangChain Agent Server",
    version="1.0",
    description="API server for Student and Publisher agents",
)

# 4. Add CORS
# Allow all origins for development/production (configure as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Add Routes
# Student Agent Endpoint
add_routes(
    app,
    student_agent,
    path="/student",
    playground_type="default",
)

# Publisher Agent Endpoint
add_routes(
    app,
    publisher_agent,
    path="/publisher",
    playground_type="default",
)

if __name__ == "__main__":
    # Get host and port from environment variables or default to 0.0.0.0:8000
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 50018))
    
    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
