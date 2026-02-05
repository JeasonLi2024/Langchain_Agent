import os
import logging
import json
from typing import List, Optional, Any
from langchain_community.embeddings.dashscope import DashScopeEmbeddings, embed_with_retry
from django.conf import settings

logger = logging.getLogger(__name__)

# Constants
COLLECTION_EMBEDDINGS = 'project_embeddings'
COLLECTION_RAW_DOCS = 'project_raw_docs'
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_EMBEDDING_DIM = 1536

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

# Milvus Config
MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')
MILVUS_ALIAS = 'default'

class EmbeddingService:
    """
    Core Embedding Service for LangChain Agent (DashScope text-embedding-v4)
    Standalone version that attempts to use Django cache if available, but falls back gracefully.
    Uses langchain_community.embeddings.DashScopeEmbeddings for standards compliance.
    """
    
    @staticmethod
    def get_dashscope_embeddings():
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            # Try to get from Config if available
            try:
                from core.config import Config
                api_key = getattr(Config, "DASHSCOPE_API_KEY", None)
            except ImportError:
                pass
                
        if not api_key:
             # Last resort: try Django settings if configured
            try:
                from django.conf import settings
                api_key = getattr(settings, "DASHSCOPE_API_KEY", None)
            except ImportError:
                pass

        if not api_key:
             raise ValueError("DASHSCOPE_API_KEY not found in environment or config")
             
        # Explicitly set dimension if possible, or rely on model default (v4 is 1536)
        # To be safe against 1024 defaults (e.g. if v3 is used under hood), we might need extra params.
        # But DashScopeEmbeddings wrapper is simple. 
        model_name = DEFAULT_EMBEDDING_MODEL
        try:
            from core.config import Config
            model_name = getattr(Config, "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        except ImportError:
            pass

        # Use CustomDashScopeEmbeddings to support dimension
        # Only set dimension for v4 or if explicitly required
        dimension = None
        if "v4" in model_name:
            dimension = 1536
            
        return CustomDashScopeEmbeddings(
            model=model_name,
            dashscope_api_key=api_key,
            dimension=dimension
        )
    
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
                
                for i, text in enumerate(texts):
                    cache_key = f"embedding:v4:{hash(text)}"
                    cached_val = cache.get(cache_key)
                    if cached_val:
                        # Validate dimension to prevent 1024 vs 1536 mismatch from old cache
                        if len(cached_val) == DEFAULT_EMBEDDING_DIM:
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
                        cache_key = f"embedding:v4:{hash(text)}"
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
            # Check if we should fallback to OpenAI client to force dimension if LangChain wrapper fails
            # For now, let's try to trust the model name.
            # But if 1024 persists, we must switch. 
            # Given the user error, let's be robust: 
            # If LangChain wrapper returns 1024, pad it? No, that's bad.
            # Use OpenAI client to be safe as in project/services.py
            
            from openai import OpenAI
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                try:
                    from core.config import Config
                    api_key = getattr(Config, "DASHSCOPE_API_KEY", None)
                except: pass
            
            if api_key:
                client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
                batch_size = 10
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    try:
                        model_name = DEFAULT_EMBEDDING_MODEL
                        try:
                            from core.config import Config
                            model_name = getattr(Config, "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
                        except: pass

                        response = client.embeddings.create(
                            model=model_name,
                            input=batch,
                            dimensions=DEFAULT_EMBEDDING_DIM # Enforce 1536
                        )
                        batch_embeddings = [None] * len(batch)
                        for item in response.data:
                             if item.index < len(batch):
                                 batch_embeddings[item.index] = item.embedding
                        all_embeddings.extend(batch_embeddings)
                    except Exception as batch_error:
                         logger.error(f"[Embedding Batch ERROR] {batch_error}")
                         all_embeddings.extend([None] * len(batch))
                
                final_embeddings = []
                for emb in all_embeddings:
                    if emb is None:
                        final_embeddings.append([0.0] * DEFAULT_EMBEDDING_DIM)
                    else:
                        final_embeddings.append(emb)
                return final_embeddings
            
            else:
                # Fallback to LangChain if no key found (unlikely)
                embedder = EmbeddingService.get_dashscope_embeddings()
                return embedder.embed_documents(texts)

        except Exception as e:
            logger.error(f"[Embedding ERROR] {e}")
            return [[0.0] * DEFAULT_EMBEDDING_DIM for _ in texts]
    
    @staticmethod
    def get_single_embedding(text: str, use_cache: bool = True) -> List[float]:
        embeddings = EmbeddingService.get_embeddings([text], use_cache=use_cache)
        return embeddings[0] if embeddings else [0.0] * DEFAULT_EMBEDDING_DIM

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
                
                for i, text in enumerate(texts):
                    cache_key = f"embedding:v4:{hash(text)}"
                    # Sync cache access
                    cached_val = cache.get(cache_key)
                    if cached_val:
                        if len(cached_val) == DEFAULT_EMBEDDING_DIM:
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
                        cache_key = f"embedding:v4:{hash(text)}"
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
            from openai import AsyncOpenAI
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                try:
                    from core.config import Config
                    api_key = getattr(Config, "DASHSCOPE_API_KEY", None)
                except: pass
            
            if api_key:
                client = AsyncOpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
                batch_size = 10
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    try:
                        model_name = DEFAULT_EMBEDDING_MODEL
                        try:
                            from core.config import Config
                            model_name = getattr(Config, "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
                        except: pass

                        response = await client.embeddings.create(
                            model=model_name,
                            input=batch,
                            dimensions=DEFAULT_EMBEDDING_DIM
                        )
                        batch_embeddings = [None] * len(batch)
                        for item in response.data:
                             if item.index < len(batch):
                                 batch_embeddings[item.index] = item.embedding
                        all_embeddings.extend(batch_embeddings)
                    except Exception as batch_error:
                         logger.error(f"[Async Embedding Batch ERROR] {batch_error}")
                         all_embeddings.extend([None] * len(batch))
                
                final_embeddings = []
                for emb in all_embeddings:
                    if emb is None:
                        final_embeddings.append([0.0] * DEFAULT_EMBEDDING_DIM)
                    else:
                        final_embeddings.append(emb)
                return final_embeddings
            
            else:
                # Fallback to sync run_in_executor
                import asyncio
                embedder = EmbeddingService.get_dashscope_embeddings()
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, embedder.embed_documents, texts)

        except Exception as e:
            logger.error(f"[Async Embedding ERROR] {e}")
            return [[0.0] * DEFAULT_EMBEDDING_DIM for _ in texts]


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

def get_or_create_collection(collection_name: str, dim: int = 1536) -> Collection:
    ensure_milvus_connection()
    
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
