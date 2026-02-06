import os
from typing import List, Optional
from dotenv import load_dotenv

# Load env variables BEFORE importing other modules that might rely on them or set defaults
# 1. Try explicit ENV_FILE
env_file = os.getenv("ENV_FILE")
if env_file and os.path.exists(env_file):
    load_dotenv(env_file)
else:
    # 2. Try .env in langchain-v2.0 root
    current_dir = os.path.dirname(os.path.abspath(__file__)) # core/
    root_dir = os.path.dirname(current_dir) # langchain-v2.0/
    default_env = os.path.join(root_dir, ".env")
    if os.path.exists(default_env):
        load_dotenv(default_env)
    else:
        # 3. Fallback to standard load (current working dir)
        load_dotenv()

from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings.dashscope import DashScopeEmbeddings, embed_with_retry
from langchain_milvus import Milvus
import pymysql
import logging

# Suppress Milvus async error logs
logging.getLogger("pymilvus").setLevel(logging.CRITICAL)

# Custom Embeddings Wrapper to support dimension parameter
class CustomDashScopeEmbeddings(DashScopeEmbeddings):
    dimension: Optional[int] = None
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        kwargs = {
            "input": texts,
            "text_type": "document",
            "model": self.model
        }
        if self.dimension:
            kwargs["dimension"] = self.dimension
            
        embeddings = embed_with_retry(self, **kwargs)
        embedding_list = [item["embedding"] for item in embeddings]
        return embedding_list

    def embed_query(self, text: str) -> List[float]:
        kwargs = {
            "input": text,
            "text_type": "query",
            "model": self.model
        }
        if self.dimension:
            kwargs["dimension"] = self.dimension
            
        embedding = embed_with_retry(self, **kwargs)[0]["embedding"]
        return embedding

class Config:
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', 3306))
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_NAME = os.getenv('DB_NAME', 'zhihui_db')
    DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
    MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
    MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 5))

    # PostgreSQL Checkpoints
    # Password 'BuptZH@2025' must be URL-encoded (@ -> %40)
    CHECKPOINT_DB_URI = os.getenv("CHECKPOINT_DB_URI", "postgresql://ai_agent:BuptZH%402025@localhost:5432/langgraph_checkpoints")

    # Model Configurations
    LLM_MODEL_UTILITY = os.getenv("LLM_MODEL_UTILITY", "qwen-turbo")
    LLM_MODEL_REASONING = os.getenv("LLM_MODEL_REASONING", "qwq-32b-preview")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")

    @classmethod
    def get_db_connection(cls):
        return pymysql.connect(
            host=cls.DB_HOST,
            port=cls.DB_PORT,
            user=cls.DB_USER,
            password=cls.DB_PASSWORD,
            database=cls.DB_NAME
        )

    @classmethod
    def get_embeddings(cls):
        if not cls.DASHSCOPE_API_KEY:
             raise ValueError("DASHSCOPE_API_KEY not found in environment variables.")
        
        dimension = None
        if "v4" in cls.EMBEDDING_MODEL:
            dimension = 1536

        return CustomDashScopeEmbeddings(
            model=cls.EMBEDDING_MODEL,
            dashscope_api_key=cls.DASHSCOPE_API_KEY,
            dimension=dimension
        )

    @classmethod
    def get_milvus_store(cls, collection_name):
        return Milvus(
            embedding_function=cls.get_embeddings(),
            connection_args={"host": cls.MILVUS_HOST, "port": cls.MILVUS_PORT},
            collection_name=collection_name
        )

    @classmethod
    def get_utility_llm(cls):
        if not cls.DASHSCOPE_API_KEY:
             raise ValueError("DASHSCOPE_API_KEY not found in environment variables.")
        # Enable streaming=True to ensure on_chat_model_stream events are emitted
        return ChatTongyi(model=cls.LLM_MODEL_UTILITY, api_key=cls.DASHSCOPE_API_KEY, streaming=True)

    @classmethod
    def get_reasoning_llm(cls):
        if not cls.DASHSCOPE_API_KEY:
             raise ValueError("DASHSCOPE_API_KEY not found in environment variables.")
        return ChatTongyi(model=cls.LLM_MODEL_REASONING, api_key=cls.DASHSCOPE_API_KEY, streaming=True)
