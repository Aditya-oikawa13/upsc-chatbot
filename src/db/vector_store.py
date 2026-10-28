"""Local, persistent Chroma vector store for question embeddings.

Vectors live on disk under DATA_DIR/chroma (see config.py) and survive
process restarts. The embedding function is Vertex AI's text-embedding-005
(see embed/vertex_embedding_function.py) — Chroma calls it automatically on
`.add()` / `.query()`, so callers of this module never touch embeddings
directly.

Usage:
    python -m src.db.vector_store   # smoke-tests the collection
"""
from __future__ import annotations

import chromadb

from .. import config
from ..embed.vertex_embedding_function import VertexEmbeddingFunction


def get_collection() -> chromadb.api.models.Collection.Collection:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        embedding_function=VertexEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


if __name__ == "__main__":
    collection = get_collection()
    print(f"Collection '{collection.name}' ready, {collection.count()} records.")
