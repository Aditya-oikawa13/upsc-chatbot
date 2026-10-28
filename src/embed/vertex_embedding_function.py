"""Chroma-compatible embedding function backed by Vertex AI's text-embedding-005.

Uses the `google-genai` SDK against the Vertex AI backend, authenticated via
Application Default Credentials (no API key involved). This is the only
network dependency in the storage/retrieval path — see PLAN.md §5/§5a for
why this is on Vertex AI rather than a local model, and for the (negligible)
expected cost.

Documents and queries are embedded with different task types
(RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY), which Vertex's embedding models are
tuned to expect — Chroma calls `embed_query` automatically for `.query()`
calls and `__call__` for everything else (e.g. `.add()`).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api.types import Space
from google import genai
from google.genai import types

from .. import config


class VertexEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        project: str = config.GCP_PROJECT,
        location: str = config.GCP_LOCATION,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.project = project
        self.location = location
        self.batch_size = batch_size
        self._client = genai.Client(vertexai=True, project=project, location=location)

    def _embed(self, texts: List[str], task_type: str, max_retries: int = 5) -> Embeddings:
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            last_error: Exception | None = None
            for attempt in range(max_retries):
                try:
                    resp = self._client.models.embed_content(
                        model=self.model_name,
                        contents=batch,
                        config=types.EmbedContentConfig(task_type=task_type),
                    )
                    all_embeddings.extend(e.values for e in resp.embeddings)
                    break
                except Exception as e:
                    # Transient network errors (connection reset, broken
                    # pipe) have been observed against this API from this
                    # environment — retry with backoff before giving up.
                    last_error = e
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
            else:
                raise RuntimeError(
                    f"Embedding batch {i}:{i + len(batch)} failed after {max_retries} attempts"
                ) from last_error
        return all_embeddings  # type: ignore[return-value]

    def __call__(self, input: Documents) -> Embeddings:
        """Used when adding/upserting documents into the collection."""
        return self._embed(list(input), task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, input: Documents) -> Embeddings:
        """Used automatically by Collection.query()."""
        return self._embed(list(input), task_type="RETRIEVAL_QUERY")

    @staticmethod
    def name() -> str:
        return "vertex_text_embedding"

    def default_space(self) -> Space:
        return "cosine"

    def supported_spaces(self) -> List[Space]:
        return ["cosine", "l2", "ip"]

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "project": self.project,
            "location": self.location,
            "batch_size": self.batch_size,
        }

    @staticmethod
    def build_from_config(config_dict: Dict[str, Any]) -> "VertexEmbeddingFunction":
        return VertexEmbeddingFunction(**config_dict)
