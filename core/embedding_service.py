import os
import asyncio
import logging
import json
from typing import List, Optional, Any
from django.conf import settings

logger = logging.getLogger(__name__)

# Constants
COLLECTION_EMBEDDINGS = 'project_embeddings'
COLLECTION_RAW_DOCS = 'project_raw_docs'
DEFAULT_EMBEDDING_MODEL = "bge-large-zh-v1.5"
DEFAULT_EMBEDDING_DIM = 1024


def _configured_embedding_model() -> str:
    try:
        from core.config import Config
        return Config.get_text_embedding_model()
    except Exception:
        return DEFAULT_EMBEDDING_MODEL


def _configured_embedding_dim() -> int:
    try:
        from core.config import Config
        return Config.get_text_embedding_dimension()
    except Exception:
        return DEFAULT_EMBEDDING_DIM


def _embedding_cache_namespace() -> str:
    return f"{_configured_embedding_model()}:{_configured_embedding_dim()}"

# Milvus Config
MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')
MILVUS_ALIAS = 'default'

class EmbeddingService:
    """
    Core Embedding Service for LangChain Agent.
    Standalone version that attempts to use Django cache if available, but falls back gracefully.
    Uses the embedding provider configured in core.config.
    """
    
    @staticmethod
    def get_configured_embeddings():
        from core.config import Config
        return Config.get_embeddings()
    
    @staticmethod
    def get_embeddings(texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """
        Batch get embeddings with cache support.
        """
        if not texts:
            return []
            
        # Try to use Django cache
        cache = None
        try:
            from django.core.cache import cache as django_cache
            cache = django_cache
        except (ImportError, Exception):
            logger.debug("Django cache not available in EmbeddingService")

        if use_cache and cache:
            try:
                cached_embeddings = []
                uncached_texts = []
                uncached_indices = []
                embedding_dim = _configured_embedding_dim()
                cache_ns = _embedding_cache_namespace()
                
                for i, text in enumerate(texts):
                    cache_key = f"embedding:{cache_ns}:{hash(text)}"
                    cached_val = cache.get(cache_key)
                    if cached_val:
                        if len(cached_val) == embedding_dim:
                            cached_embeddings.append((i, cached_val))
                        else:
                            # Cache invalid (wrong dim), treat as uncached
                            uncached_texts.append(text)
                            uncached_indices.append(i)
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
                
                if not uncached_texts:
                    result = [None] * len(texts)
                    for i, embedding in cached_embeddings:
                        result[i] = embedding
                    return result
                
                new_embeddings = EmbeddingService._fetch_embeddings(uncached_texts)
                
                for i, text in enumerate(uncached_texts):
                    if i < len(new_embeddings):
                        cache_key = f"embedding:{cache_ns}:{hash(text)}"
                        cache.set(cache_key, new_embeddings[i], timeout=3600)
                
                result = [None] * len(texts)
                for i, embedding in cached_embeddings:
                    result[i] = embedding
                for i, idx in enumerate(uncached_indices):
                    if i < len(new_embeddings):
                        result[idx] = new_embeddings[i]
                return result
            except Exception as e:
                logger.warning(f"Cache operation failed: {e}. Proceeding without cache.")
                return EmbeddingService._fetch_embeddings(texts)
        else:
            return EmbeddingService._fetch_embeddings(texts)

    @staticmethod
    def _fetch_embeddings(texts: List[str]) -> List[List[float]]:
        try:
            embedder = EmbeddingService.get_configured_embeddings()
            return embedder.embed_documents(texts)
        except Exception as e:
            logger.error(f"[Embedding ERROR] {e}")
            zero = [0.0] * _configured_embedding_dim()
            return [zero[:] for _ in texts]
    
    @staticmethod
    def get_single_embedding(text: str, use_cache: bool = True) -> List[float]:
        embeddings = EmbeddingService.get_embeddings([text], use_cache=use_cache)
        return embeddings[0] if embeddings else [0.0] * _configured_embedding_dim()

    @staticmethod
    async def aget_embeddings(texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """
        Async Batch get embeddings with cache support.
        """
        if not texts:
            return []
            
        # Try to use Django cache (sync access is acceptable for memory cache, 
        # for redis it blocks but it's fast. Ideally use sync_to_async or async cache client)
        cache = None
        try:
            from django.core.cache import cache as django_cache
            cache = django_cache
        except (ImportError, Exception):
            pass

        if use_cache and cache:
            try:
                cached_embeddings = []
                uncached_texts = []
                uncached_indices = []
                embedding_dim = _configured_embedding_dim()
                cache_ns = _embedding_cache_namespace()
                
                for i, text in enumerate(texts):
                    cache_key = f"embedding:{cache_ns}:{hash(text)}"
                    cached_val = cache.get(cache_key)
                    if cached_val:
                        if len(cached_val) == embedding_dim:
                            cached_embeddings.append((i, cached_val))
                        else:
                            uncached_texts.append(text)
                            uncached_indices.append(i)
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
                
                if not uncached_texts:
                    result = [None] * len(texts)
                    for i, embedding in cached_embeddings:
                        result[i] = embedding
                    return result
                
                new_embeddings = await EmbeddingService._afetch_embeddings(uncached_texts)
                
                for i, text in enumerate(uncached_texts):
                    if i < len(new_embeddings):
                        cache_key = f"embedding:{cache_ns}:{hash(text)}"
                        cache.set(cache_key, new_embeddings[i], timeout=3600)
                
                result = [None] * len(texts)
                for i, embedding in cached_embeddings:
                    result[i] = embedding
                for i, idx in enumerate(uncached_indices):
                    if i < len(new_embeddings):
                        result[idx] = new_embeddings[i]
                return result
            except Exception as e:
                logger.warning(f"Async Cache operation failed: {e}. Proceeding without cache.")
                return await EmbeddingService._afetch_embeddings(texts)
        else:
            return await EmbeddingService._afetch_embeddings(texts)

    @staticmethod
    async def _afetch_embeddings(texts: List[str]) -> List[List[float]]:
        try:
            embedder = EmbeddingService.get_configured_embeddings()
            return await asyncio.to_thread(embedder.embed_documents, texts)
        except Exception as e:
            logger.error(f"[Async Embedding ERROR] {e}")
            zero = [0.0] * _configured_embedding_dim()
            return [zero[:] for _ in texts]


def generate_embedding(text: str) -> List[float]:
    """Wrapper for compatibility."""
    return EmbeddingService.get_single_embedding(text)

# --- Milvus Helpers ---
from pymilvus import connections, Collection, utility, DataType, FieldSchema, CollectionSchema

def ensure_milvus_connection():
    try:
        connections.connect(alias=MILVUS_ALIAS, host=MILVUS_HOST, port=MILVUS_PORT)
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")

def get_or_create_collection(collection_name: str, dim: Optional[int] = None) -> Collection:
    ensure_milvus_connection()
    if dim is None:
        dim = _configured_embedding_dim()
    
    if utility.has_collection(collection_name):
        return Collection(collection_name)
    
    # Schema Definition
    if collection_name == COLLECTION_EMBEDDINGS:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="project_id", dtype=DataType.INT64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535)
        ]
    elif collection_name == COLLECTION_RAW_DOCS:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="project_id", dtype=DataType.INT64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="chunk_index", dtype=DataType.INT64)
        ]
    else:
        raise ValueError(f"Unknown collection: {collection_name}")
        
    schema = CollectionSchema(fields, f"{collection_name} schema")
    collection = Collection(collection_name, schema)
    
    # Create Index
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="vector", index_params=index_params)
    collection.load()
    return collection
