import asyncio
from typing import List, Optional

import numpy as np
from langchain_core.embeddings import Embeddings
from volcenginesdkarkruntime import Ark


DEFAULT_QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


class ArkTextEmbeddings(Embeddings):
    """Volcengine Ark text embeddings adapter for LangChain."""

    def __init__(
        self,
        api_key: str,
        model: str,
        dimension: int = 1024,
        normalize: bool = True,
        batch_size: int = 32,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.normalize = normalize
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.use_multimodal_api = "vision" in model.lower()

    def _build_client(self) -> Ark:
        return Ark(api_key=self.api_key)

    def _prepare_inputs(self, inputs: List[str], is_query: bool) -> List[str]:
        if not is_query:
            return inputs
        return [
            f"Instruct: {self.query_instruction}\nQuery: {text}"
            for text in inputs
        ]

    def _prepare_multimodal_inputs(self, inputs: List[str], is_query: bool) -> List[dict]:
        prepared_texts = self._prepare_inputs(inputs, is_query=is_query)
        return [{"text": text, "type": "text"} for text in prepared_texts]

    def _post_process(self, vectors: List[List[float]]) -> List[List[float]]:
        if not vectors:
            return []

        array = np.asarray(vectors, dtype=np.float32)
        if self.dimension:
            array = array[:, : self.dimension]

        if self.normalize:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            array = array / norms

        return array.astype(np.float32).tolist()

    def _embed(self, inputs: List[str], is_query: bool) -> List[List[float]]:
        if not inputs:
            return []

        client = self._build_client()
        all_vectors: List[List[float]] = []

        if self.use_multimodal_api:
            prepared_inputs = self._prepare_multimodal_inputs(inputs, is_query=is_query)
            # The multimodal embeddings API expects one multimodal sample per request,
            # not a batch of independent text items.
            for batch_item in prepared_inputs:
                response = client.multimodal_embeddings.create(
                    model=self.model,
                    input=[batch_item],
                    encoding_format="float",
                )
                data = response.data
                if hasattr(data, "embedding"):
                    all_vectors.append(data.embedding)
                else:
                    ordered = sorted(data, key=lambda item: item.index)
                    all_vectors.extend([item.embedding for item in ordered])
        else:
            prepared_inputs = self._prepare_inputs(inputs, is_query=is_query)
            for start in range(0, len(prepared_inputs), self.batch_size):
                batch = prepared_inputs[start : start + self.batch_size]
                response = client.embeddings.create(
                    model=self.model,
                    input=batch,
                    encoding_format="float",
                )
                ordered = sorted(response.data, key=lambda item: item.index)
                all_vectors.extend([item.embedding for item in ordered])

        return self._post_process(all_vectors)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts, is_query=False)

    def embed_query(self, text: str) -> List[float]:
        vectors = self._embed([text], is_query=True)
        return vectors[0] if vectors else []

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.embed_query, text)
