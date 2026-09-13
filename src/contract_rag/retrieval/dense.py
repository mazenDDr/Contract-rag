"""Dense retrieval: chunk embeddings in local Qdrant, one collection per (chunking strategy, model)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qdrant_client import QdrantClient, models

from contract_rag.retrieval.embed import Encoder
from contract_rag.schemas import Chunk, RetrievedChunk


def build_collection(
    client: QdrantClient, name: str, chunks: Sequence[Chunk], vectors: np.ndarray, batch_size: int = 256
) -> None:
    """(Re)create the collection with cosine distance and a payload index on doc_id."""
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name, vectors_config=models.VectorParams(size=int(vectors.shape[1]), distance=models.Distance.COSINE)
    )
    client.create_payload_index(name, field_name="doc_id", field_schema=models.PayloadSchemaType.KEYWORD)
    for start in range(0, len(chunks), batch_size):
        client.upsert(
            name,
            points=[
                models.PointStruct(
                    id=start + offset,
                    vector=vectors[start + offset].tolist(),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "section_path": chunk.section_path,
                        "page_start": chunk.page_start,
                    },
                )
                for offset, chunk in enumerate(chunks[start : start + batch_size])
            ],
        )


class DenseRetriever:
    """Implements schemas.Retriever: nearest chunks by cosine similarity, optionally within one contract."""

    def __init__(self, client: QdrantClient, collection: str, encoder: Encoder, query_prefix: str = ""):
        self.client = client
        self.collection = collection
        self.encoder = encoder
        self.query_prefix = query_prefix

    def retrieve(self, query: str, k: int, doc_id: str | None = None) -> list[RetrievedChunk]:
        vector = self.encoder.encode([self.query_prefix + query], batch_size=1)[0]
        query_filter = (
            models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
            if doc_id is not None
            else None
        )
        points = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            limit=k,
            query_filter=query_filter,
            with_payload=["chunk_id"],
        ).points
        return [
            RetrievedChunk(chunk_id=p.payload["chunk_id"], score=float(p.score), rank=rank, stage="dense")
            for rank, p in enumerate(points, start=1)
        ]
