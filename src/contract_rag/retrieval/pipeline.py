"""Retrieval pipeline: sparse and/or dense candidates -> optional fusion -> optional cross-encoder rerank.

Every intermediate ranked list and its latency is recorded in RetrievalResult, so failure analysis can see
where a relevant chunk was lost ("BM25 had it at rank 3; the reranker pushed it to 14").
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path

from qdrant_client import QdrantClient

from contract_rag.retrieval.bm25 import BM25Retriever
from contract_rag.retrieval.build import load_bm25_retriever
from contract_rag.retrieval.config import IndexConfig, collection_name, load_chunks
from contract_rag.retrieval.dense import DenseRetriever
from contract_rag.retrieval.embed import Encoder, SentenceTransformerEncoder
from contract_rag.retrieval.fusion import reciprocal_rank_fusion, weighted_fusion
from contract_rag.retrieval.rerank import CrossEncoderReranker
from contract_rag.schemas import RetrievalConfig, RetrievalResult, RetrievedChunk, Retriever


def scoped_query(question: str, contract_name: str | None, doc_filter: bool) -> str:
    """Within one contract, the contract's name only matches its title page, preamble and signature blocks,
    which then outrank the actual clause. When retrieval is scoped to that contract, drop the name."""
    if doc_filter and contract_name and contract_name in question:
        return question.replace(contract_name, "the agreement")
    return question


class RetrievalPipeline:
    """Implements schemas.Retriever (via `retrieve`); `run` returns the full RetrievalResult."""

    def __init__(
        self,
        config: RetrievalConfig,
        *,
        bm25: Retriever | None = None,
        dense: Retriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        chunk_text: Mapping[str, str] | None = None,
    ):
        if not config.sparse and not config.dense_model:
            raise ValueError("a pipeline needs BM25, a dense model, or both")
        if config.sparse and bm25 is None:
            raise ValueError("config asks for BM25 but no BM25 retriever was given")
        if config.dense_model and dense is None:
            raise ValueError("config asks for a dense model but no dense retriever was given")
        if config.sparse and config.dense_model and config.fusion == "none":
            raise ValueError("combining BM25 and dense retrieval needs a fusion method")
        if config.reranker and (reranker is None or chunk_text is None):
            raise ValueError("config asks for a reranker but no reranker or chunk text was given")
        self.config = config
        self.config_id = config.config_id()
        self.bm25, self.dense, self.reranker = bm25, dense, reranker
        self.chunk_text = chunk_text

    def run(self, query: str, qid: str = "", doc_id: str | None = None) -> RetrievalResult:
        cfg = self.config
        scope = doc_id if cfg.doc_filter else None
        stages: dict[str, list[RetrievedChunk]] = {}
        latency: dict[str, float] = {}
        started = time.perf_counter()

        def timed(stage: str, step: Callable[[], list[RetrievedChunk]]) -> list[RetrievedChunk]:
            start = time.perf_counter()
            stages[stage] = step()
            latency[stage] = (time.perf_counter() - start) * 1000
            return stages[stage]

        candidates: list[RetrievedChunk] = []
        if cfg.sparse:
            candidates = timed("bm25", lambda: self.bm25.retrieve(query, cfg.k_candidates, scope))
        if cfg.dense_model:
            candidates = timed("dense", lambda: self.dense.retrieve(query, cfg.k_candidates, scope))
        if cfg.sparse and cfg.dense_model:
            if cfg.fusion == "rrf":
                fuse = lambda: reciprocal_rank_fusion([stages["bm25"], stages["dense"]], cfg.rrf_k)  # noqa: E731
            else:
                fuse = lambda: weighted_fusion(stages["bm25"], stages["dense"], cfg.fusion_alpha)  # noqa: E731
            candidates = timed("fusion", lambda: fuse()[: cfg.k_candidates])
        if self.reranker is not None and cfg.reranker:
            candidates = timed("rerank", lambda: self.reranker.rerank(query, candidates, self.chunk_text))
        final = [c.model_copy(update={"rank": i}) for i, c in enumerate(candidates[: cfg.k_final], start=1)]
        latency["total"] = (time.perf_counter() - started) * 1000
        return RetrievalResult(
            qid=qid, config_id=self.config_id, final=final, stages=stages, latency_ms=latency
        )

    def retrieve(self, query: str, k: int, doc_id: str | None = None) -> list[RetrievedChunk]:
        return self.run(query, doc_id=doc_id).final[:k]


class RetrievalResources:
    """Lazily loaded indexes and models, shared across pipelines. Qdrant's local mode allows one client per
    path, so every dense retriever shares a single client."""

    def __init__(
        self,
        index_config: IndexConfig,
        repo_root: Path,
        encoder_factory: Callable[[str, str], Encoder] = SentenceTransformerEncoder,
        reranker_factory: Callable[..., CrossEncoderReranker] = CrossEncoderReranker,
    ):
        self.config = index_config
        self.root = repo_root
        self.encoder_factory = encoder_factory
        self.reranker_factory = reranker_factory
        self._client: QdrantClient | None = None
        self._texts: dict[str, dict[str, str]] = {}
        self._bm25: dict[str, BM25Retriever] = {}
        self._encoders: dict[str, Encoder] = {}
        self._rerankers: dict[str, CrossEncoderReranker] = {}

    def chunk_text(self, strategy: str) -> dict[str, str]:
        if strategy not in self._texts:
            chunks = load_chunks(self.root / self.config.chunks_dir / f"{strategy}.jsonl")
            self._texts[strategy] = {c.chunk_id: c.text for c in chunks}
        return self._texts[strategy]

    def bm25(self, strategy: str) -> BM25Retriever:
        if strategy not in self._bm25:
            self._bm25[strategy] = load_bm25_retriever(self.root / self.config.index_dir, strategy)
        return self._bm25[strategy]

    def dense(self, strategy: str, model_key: str) -> DenseRetriever:
        model = self.config.dense.models[model_key]
        if self._client is None:
            self._client = QdrantClient(path=str(self.root / self.config.index_dir / "qdrant"))
        if model_key not in self._encoders:
            self._encoders[model_key] = self.encoder_factory(model.name, self.config.dense.device)
        return DenseRetriever(
            self._client, collection_name(strategy, model_key), self._encoders[model_key], model.query_prefix
        )

    def reranker(self, key: str) -> CrossEncoderReranker:
        if key not in self._rerankers:
            cfg = self.config.rerankers[key]
            self._rerankers[key] = self.reranker_factory(
                cfg.name,
                device=self.config.dense.device,
                batch_size=cfg.batch_size,
                max_length=cfg.max_length,
            )
        return self._rerankers[key]

    def pipeline(self, config: RetrievalConfig) -> RetrievalPipeline:
        return RetrievalPipeline(
            config,
            bm25=self.bm25(config.chunking) if config.sparse else None,
            dense=self.dense(config.chunking, config.dense_model) if config.dense_model else None,
            reranker=self.reranker(config.reranker) if config.reranker else None,
            chunk_text=self.chunk_text(config.chunking) if config.reranker else None,
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
