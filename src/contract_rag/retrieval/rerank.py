"""Cross-encoder reranking: the model reads the query and each candidate together, which is more accurate
than comparing two independently computed vectors, and too slow to run on anything but a short list."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from contract_rag.schemas import RetrievedChunk


class PairScorer(Protocol):
    def predict(self, sentences: list[tuple[str, str]], **kwargs: Any) -> Sequence[float]: ...


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 512,
        scorer: PairScorer | None = None,
    ):
        if scorer is None:
            import torch
            from sentence_transformers import CrossEncoder

            if device == "auto":
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            scorer = CrossEncoder(model_name, device=device, max_length=max_length)
        self.name = model_name
        self.scorer = scorer
        self.batch_size = batch_size

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], texts: Mapping[str, str]
    ) -> list[RetrievedChunk]:
        """All candidates, reordered by cross-encoder score (ties keep their incoming order). The full list is
        returned so failure analysis can see how far a relevant chunk was pushed down."""
        if not candidates:
            return []
        pairs = [(query, texts[c.chunk_id]) for c in candidates]
        scores = [
            float(s) for s in self.scorer.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        ]
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i].rank))
        return [
            RetrievedChunk(chunk_id=candidates[i].chunk_id, score=scores[i], rank=rank, stage="rerank")
            for rank, i in enumerate(order, start=1)
        ]
