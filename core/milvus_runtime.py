"""Single-client Milvus runtime for long-running batch jobs."""
from __future__ import annotations

from typing import Any, Iterable

from pymilvus import DataType, MilvusClient

from core.config import Config


TAG_COLLECTIONS = ("student_interests", "student_skills")
PROJECT_COLLECTIONS = ("project_raw_docs", "project_embeddings")


class MilvusRuntime:
    """Own exactly one MilvusClient and reuse it for all batch operations."""

    def __init__(self, rpc_timeout: float = 30.0) -> None:
        self.rpc_timeout = rpc_timeout
        kwargs: dict[str, Any] = {
            "dedicated": True,
            "grpc_options": {
                "grpc.keepalive_time_ms": 600_000,
                "grpc.keepalive_timeout_ms": 20_000,
                "grpc.keepalive_permit_without_calls": False,
            },
        }
        if Config.MILVUS_LITE_URI:
            kwargs["uri"] = Config.MILVUS_LITE_URI
        else:
            kwargs["uri"] = f"http://{Config.MILVUS_HOST}:{Config.MILVUS_PORT}"
            if Config.MILVUS_DB_NAME:
                kwargs["db_name"] = Config.MILVUS_DB_NAME
        self.client = MilvusClient(**kwargs)
        self._loaded: set[str] = set()

    def close(self) -> None:
        self.client.close()

    def _ensure_project_collection(self, collection_name: str, dimension: int) -> None:
        if self.client.has_collection(collection_name, timeout=self.rpc_timeout):
            return
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="project_id", datatype=DataType.INT64)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        if collection_name == "project_raw_docs":
            schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
            timeout=60,
        )

    def initialize(self, require_tags: bool, require_projects: bool) -> None:
        required: list[str] = []
        if require_tags:
            required.extend(TAG_COLLECTIONS)
        if require_projects:
            dimension = Config.get_text_embedding_dimension()
            for name in PROJECT_COLLECTIONS:
                self._ensure_project_collection(name, dimension)
            required.extend(PROJECT_COLLECTIONS)
        missing = [name for name in required if not self.client.has_collection(name, timeout=self.rpc_timeout)]
        if missing:
            raise RuntimeError(f"Milvus collections missing: {missing}")
        for name in required:
            self.load(name)

    def load(self, collection_name: str) -> None:
        if collection_name in self._loaded:
            return
        self.client.load_collection(collection_name, timeout=60)
        self._loaded.add(collection_name)

    def search(self, collection_name: str, vector: list[float], output_fields: Iterable[str], limit: int = 5) -> list[dict[str, Any]]:
        self.load(collection_name)
        result = self.client.search(
            collection_name=collection_name,
            data=[vector],
            anns_field="vector",
            limit=limit,
            output_fields=list(output_fields),
            # Do not override metric_type: use the collection index metric (tags are L2; project vectors are COSINE).
            search_params={"params": {}},
            timeout=self.rpc_timeout,
        )
        return list(result[0]) if result else []

    def replace_project_vectors(
        self,
        project_id: int,
        chunks: list[str],
        chunk_vectors: list[list[float]],
        semantic_text: str,
        semantic_vector: list[float],
    ) -> tuple[int, bool]:
        """Idempotently replace both project collections for one project id."""
        for name in PROJECT_COLLECTIONS:
            self.load(name)
            self.client.delete(
                collection_name=name,
                filter=f"project_id == {project_id}",
                timeout=self.rpc_timeout,
            )
        raw_rows = [
            {
                "project_id": project_id,
                "vector": vector,
                "content": chunk[:65535],
                "chunk_index": index,
            }
            for index, (chunk, vector) in enumerate(zip(chunks, chunk_vectors))
            if vector and any(vector)
        ]
        if raw_rows:
            self.client.insert(
                collection_name="project_raw_docs",
                data=raw_rows,
                timeout=60,
            )
        semantic_ok = bool(semantic_vector and any(semantic_vector))
        if semantic_ok:
            self.client.insert(
                collection_name="project_embeddings",
                data=[{
                    "project_id": project_id,
                    "vector": semantic_vector,
                    "content": semantic_text[:65535],
                }],
                timeout=60,
            )
        self.client.flush("project_raw_docs", timeout=60)
        self.client.flush("project_embeddings", timeout=60)
        return len(raw_rows), semantic_ok

    def replace_semantic_vector(
        self,
        project_id: int,
        semantic_text: str,
        semantic_vector: list[float],
    ) -> bool:
        """Replace only the semantic vector, preserving parsed chunk vectors."""
        self.load("project_embeddings")
        self.client.delete(
            collection_name="project_embeddings",
            filter=f"project_id == {project_id}",
            timeout=self.rpc_timeout,
        )
        semantic_ok = bool(semantic_vector and any(semantic_vector))
        if semantic_ok:
            self.client.insert(
                collection_name="project_embeddings",
                data=[{
                    "project_id": project_id,
                    "vector": semantic_vector,
                    "content": semantic_text[:65535],
                }],
                timeout=60,
            )
        self.client.flush("project_embeddings", timeout=60)
        return semantic_ok
