import os
from urllib.parse import urlparse
from dotenv import load_dotenv

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_CURRENT_DIR)

# Load env variables BEFORE importing other modules that might rely on them or set defaults
# 1. Try explicit ENV_FILE
env_file = os.getenv("ENV_FILE")
if env_file and os.path.exists(env_file):
    load_dotenv(env_file)
else:
    # 2. Try .env in langchain-v2.0 root
    default_env = os.path.join(_ROOT_DIR, ".env")
    if os.path.exists(default_env):
        load_dotenv(default_env)
    else:
        # 3. Fallback to standard load (current working dir)
        load_dotenv()

from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI
import pymysql
import logging
from core.gateway_embeddings import GatewayEmbeddings

# Suppress Milvus async error logs
logging.getLogger("pymilvus").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

class Config:
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', 3306))
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_NAME = os.getenv('DB_NAME', 'zhihui_db')
    LLM_GATEWAY_API_KEY = (
        os.getenv("BUPT_API_KEY")
        or os.getenv("LLM_GATEWAY_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
    )
    ARK_API_KEY = os.getenv('ARK_API_KEY')
    MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
    MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 5))

    # PostgreSQL Checkpoints
    # Password 'BuptZH@2025' must be URL-encoded (@ -> %40)
    CHECKPOINT_DB_URI = os.getenv("CHECKPOINT_DB_URI", "postgresql://ai_agent:BuptZH%402025@localhost:5432/langgraph_checkpoints")

    # Model Configurations
    LLM_MODEL_UTILITY = os.getenv("LLM_MODEL_UTILITY", "deepseek-medium")
    LLM_MODEL_REASONING = os.getenv("LLM_MODEL_REASONING", "qwen-latest")
    LLM_MODEL_EXTRACTION = os.getenv("LLM_MODEL_EXTRACTION", "qwen-medium")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-large-zh-v1.5")
    TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", EMBEDDING_MODEL)
    TEXT_EMBEDDING_ENDPOINT = os.getenv("TEXT_EMBEDDING_ENDPOINT", "")
    TEXT_EMBEDDING_DIM = int(os.getenv("TEXT_EMBEDDING_DIM", "1024"))
    TEXT_EMBEDDING_BATCH_SIZE = int(os.getenv("TEXT_EMBEDDING_BATCH_SIZE", "32"))
    TEXT_EMBEDDING_NORMALIZE = os.getenv("TEXT_EMBEDDING_NORMALIZE", "true").lower() in {"1", "true", "yes", "on"}
    TEXT_EMBEDDING_QUERY_INSTRUCTION = os.getenv(
        "TEXT_EMBEDDING_QUERY_INSTRUCTION",
        "Given a web search query, retrieve relevant passages that answer the query",
    )
    LLM_CHAT_BASE_URL = os.getenv(
        "LLM_CHAT_BASE_URL",
        "https://llm-gw.bupt.edu.cn/v1/chat/completions",
    )
    LLM_EMBEDDING_BASE_URL = os.getenv(
        "LLM_EMBEDDING_BASE_URL",
        "https://llm-gw.bupt.edu.cn/v1/embeddings",
    )
    OPENAI_COMPAT_BASE_URL = os.getenv(
        "OPENAI_COMPAT_BASE_URL",
        "https://llm-gw.bupt.edu.cn/v1",
    )

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
    def get_text_embedding_model(cls) -> str:
        return cls.TEXT_EMBEDDING_MODEL or cls.EMBEDDING_MODEL or "bge-large-zh-v1.5"

    @classmethod
    def get_text_embedding_target(cls) -> str:
        return cls.TEXT_EMBEDDING_ENDPOINT or cls.get_text_embedding_model()

    @staticmethod
    def _normalize_openai_base_url(url: str) -> str:
        raw = (url or "").strip()
        if not raw:
            raise ValueError("OpenAI-compatible gateway base URL is not configured.")

        parsed = urlparse(raw)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid gateway URL: {raw}")

        path = parsed.path.rstrip("/")
        for suffix in ("/chat/completions", "/embeddings"):
            if path.endswith(suffix):
                path = path[: -len(suffix)]
                break

        normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
        return normalized.rstrip("/")

    @classmethod
    def get_chat_base_url(cls) -> str:
        return cls._normalize_openai_base_url(
            cls.LLM_CHAT_BASE_URL or cls.OPENAI_COMPAT_BASE_URL
        )

    @classmethod
    def get_embedding_base_url(cls) -> str:
        return cls._normalize_openai_base_url(
            cls.LLM_EMBEDDING_BASE_URL or cls.OPENAI_COMPAT_BASE_URL
        )

    @classmethod
    def get_text_embedding_dimension(cls) -> int:
        allowed_dims = {2048, 1024, 512, 256}
        if cls.TEXT_EMBEDDING_DIM not in allowed_dims:
            raise ValueError(
                f"TEXT_EMBEDDING_DIM must be one of {sorted(allowed_dims)}, got {cls.TEXT_EMBEDDING_DIM}"
            )
        return cls.TEXT_EMBEDDING_DIM

    @classmethod
    def get_embeddings(cls):
        if not cls.LLM_GATEWAY_API_KEY:
             raise ValueError("LLM gateway API key not found in environment variables.")

        return GatewayEmbeddings(
            api_key=cls.LLM_GATEWAY_API_KEY,
            base_url=cls.get_embedding_base_url(),
            model=cls.get_text_embedding_target(),
            dimension=cls.get_text_embedding_dimension(),
            normalize=cls.TEXT_EMBEDDING_NORMALIZE,
            batch_size=cls.TEXT_EMBEDDING_BATCH_SIZE,
            query_instruction=cls.TEXT_EMBEDDING_QUERY_INSTRUCTION,
        )

    @classmethod
    def get_milvus_store(cls, collection_name):
        # Determine text field based on collection
        if collection_name in ["student_interests", "student_skills"]:
            text_field = "text"
        else:
            text_field = "content"
            
        return Milvus(
            embedding_function=cls.get_embeddings(),
            connection_args={"host": cls.MILVUS_HOST, "port": cls.MILVUS_PORT},
            collection_name=collection_name,
            text_field=text_field
        )

    @classmethod
    def _build_chat_model(cls, model_name: str):
        if not cls.LLM_GATEWAY_API_KEY:
             raise ValueError("LLM gateway API key not found in environment variables.")
        return ChatOpenAI(
            base_url=cls.get_chat_base_url(),
            api_key=cls.LLM_GATEWAY_API_KEY,
            model=model_name,
            streaming=True,
        )

    @classmethod
    def get_utility_llm(cls):
        return cls._build_chat_model(cls.LLM_MODEL_UTILITY)

    @classmethod
    def get_reasoning_llm(cls):
        return cls._build_chat_model(cls.LLM_MODEL_REASONING)

    @classmethod
    def get_extraction_llm(cls, temperature: float = 0.1):
        if not cls.LLM_GATEWAY_API_KEY:
             raise ValueError("LLM gateway API key not found in environment variables.")
        return ChatOpenAI(
            base_url=cls.get_chat_base_url(),
            api_key=cls.LLM_GATEWAY_API_KEY,
            model=cls.LLM_MODEL_EXTRACTION,
            streaming=True,
            temperature=temperature,
        )
