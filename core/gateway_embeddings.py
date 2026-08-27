import asyncio
from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings
from openai import BadRequestError, OpenAI


DEFAULT_QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


class GatewayEmbeddings(Embeddings):
    """OpenAI-compatible embeddings adapter for the campus LLM gateway."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        dimension: int = 1024,
        normalize: bool = True,
        batch_size: int = 32,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.dimension = dimension
        self.normalize = normalize
        self.batch_size = batch_size
        self.query_instruction = query_instruction

    def _build_client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _prepare_inputs(self, inputs: List[str], is_query: bool) -> List[str]:
        if not is_query:
            return inputs
        return [
            f"Instruct: {self.query_instruction}\nQuery: {text}"
            for text in inputs
        ]

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

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "maximum context length" in message or "input_tokens" in message

    @staticmethod
    def _split_text(text: str) -> List[str]:
        text = (text or "").strip()
        if len(text) <= 1:
            return [text]

        midpoint = len(text) // 2
        candidates = [
            text.rfind("\n", 0, midpoint),
            text.rfind("。", 0, midpoint),
            text.rfind("，", 0, midpoint),
            text.rfind(" ", 0, midpoint),
        ]
        split_at = max(candidates)
        if split_at <= 0:
            split_at = midpoint

        left = text[:split_at].strip()
        right = text[split_at:].strip()
        return [part for part in (left, right) if part]

    def _request_embeddings(self, client: OpenAI, inputs: List[str]) -> List[List[float]]:
        response = client.embeddings.create(
            model=self.model,
            input=inputs,
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    def _embed_single(self, client: OpenAI, text: str) -> List[float]:
        try:
            return self._request_embeddings(client, [text])[0]
        except BadRequestError as exc:
            if not self._is_context_length_error(exc):
                raise

            parts = self._split_text(text)
            if len(parts) <= 1:
                raise

            vectors = [self._embed_single(client, part) for part in parts]
            weights = np.asarray([max(len(part), 1) for part in parts], dtype=np.float32)
            matrix = np.asarray(vectors, dtype=np.float32)
            averaged = np.average(matrix, axis=0, weights=weights)
            return averaged.astype(np.float32).tolist()

    def _embed(self, inputs: List[str], is_query: bool) -> List[List[float]]:
        if not inputs:
            return []

        client = self._build_client()
        prepared_inputs = self._prepare_inputs(inputs, is_query=is_query)
        all_vectors: List[List[float]] = []

        for start in range(0, len(prepared_inputs), self.batch_size):
            batch = prepared_inputs[start : start + self.batch_size]
            try:
                all_vectors.extend(self._request_embeddings(client, batch))
            except BadRequestError as exc:
                if not self._is_context_length_error(exc):
                    raise
                all_vectors.extend([self._embed_single(client, text) for text in batch])

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
