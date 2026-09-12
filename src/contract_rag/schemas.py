"""Shared data contracts between pipeline stages.

Every stage reads and writes these models, so change existing fields deliberately.
Adding new optional fields is always safe.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

ChunkStrategy = Literal["fixed", "sentence_window", "section"]
BlockType = Literal["heading", "paragraph", "list_item", "table", "other"]
Stage = Literal["bm25", "dense", "fusion", "rerank"]
Split = Literal["dev", "test"]
QuestionType = Literal["cuad_derived", "multi_span", "numeric", "unanswerable", "cross_doc"]


# ---------- ingestion ----------


class Document(BaseModel):
    doc_id: str  # stable slug derived from the CUAD filename
    title: str
    source_path: str  # relative to repo root
    num_pages: int
    contract_type: str | None = None  # CUAD folder / agreement type
    full_text: str  # normalized text of the parsed PDF; all char offsets refer to this
    parser: str = "pymupdf"


class Block(BaseModel):
    """Layout-aware parser output, before chunking."""

    doc_id: str
    block_id: str  # f"{doc_id}::b{index:05d}"
    block_type: BlockType
    text: str  # tables are rendered as markdown
    page: int  # 1-indexed
    section_path: list[str] = Field(default_factory=list)  # e.g. ["12. TERMINATION", "12.2 For Convenience"]
    char_start: int
    char_end: int


class Chunk(BaseModel):
    chunk_id: str  # f"{doc_id}::{strategy}::{index:05d}" — what the LLM cites
    doc_id: str
    strategy: ChunkStrategy
    text: str  # the text that is embedded / indexed
    context_text: str | None = None  # expanded text passed to the LLM (sentence_window); falls back to text
    section_path: list[str] = Field(default_factory=list)
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    token_count: int
    has_table: bool = False


# ---------- evaluation set ----------


class EvalQuestion(BaseModel):
    qid: str  # e.g. "q0001"
    question: str
    doc_id: str | None  # None => corpus-wide question
    qtype: QuestionType
    category: str  # CUAD category name, or "custom"
    reference_answer: str  # "" for unanswerable
    evidence_spans: list[str] = Field(default_factory=list)  # verbatim gold evidence text (not offsets)
    answerable: bool = True
    split: Split
    source: Literal["cuad", "handwritten"] = "cuad"
    notes: str = ""


# ---------- retrieval ----------


class RetrievedChunk(BaseModel):
    chunk_id: str
    score: float
    rank: int  # 1-indexed within its stage
    stage: Stage


class RetrievalConfig(BaseModel):
    chunking: ChunkStrategy
    sparse: bool = True
    dense_model: str | None = None  # "bge-large" | "e5-large" | "oai-3-large" | None
    fusion: Literal["none", "rrf", "weighted"] = "none"
    fusion_alpha: float = 0.5  # weight on dense for "weighted"
    rrf_k: int = 60
    reranker: str | None = None  # "bge-reranker-v2-m3" | "cohere" | None
    k_candidates: int = 50  # per-retriever depth before fusion/rerank
    k_final: int = 8  # chunks handed to the generator
    doc_filter: bool = True  # scope retrieval to the question's doc_id when present

    def config_id(self) -> str:
        parts = [
            self.chunking,
            "bm25" if self.sparse else "nobm25",
            self.dense_model or "nodense",
            self.fusion if self.fusion != "weighted" else f"w{self.fusion_alpha:g}",
            self.reranker or "norerank",
            f"k{self.k_final}",
            "doc" if self.doc_filter else "corpus",
        ]
        return "__".join(parts)


class RetrievalResult(BaseModel):
    qid: str
    config_id: str
    final: list[RetrievedChunk]  # what goes to the generator
    # every intermediate ranked list (bm25/dense/fusion/rerank), kept for failure analysis
    stages: dict[str, list[RetrievedChunk]] = Field(default_factory=dict)
    latency_ms: dict[str, float] = Field(default_factory=dict)  # per stage + "total"


# ---------- generation ----------


class GenerationResult(BaseModel):
    qid: str
    config_id: str
    model: str
    answer: str
    cited_chunk_ids: list[str] = Field(default_factory=list)
    abstained: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


# ---------- scores ----------


class EvalScores(BaseModel):
    qid: str
    config_id: str
    recall_at: dict[int, float] = Field(default_factory=dict)  # {1:…, 3:…, 5:…, 10:…, 20:…}
    mrr: float | None = None
    ndcg_at_10: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    faithfulness: float | None = None
    citation_validity: float | None = None
    answer_relevance: float | None = None
    answer_correctness: float | None = None
    abstention_correct: bool | None = None
    judge_model: str | None = None
    judge_rationale: str | None = None


# ---------- component protocols ----------


@runtime_checkable
class Chunker(Protocol):
    strategy: ChunkStrategy

    def chunk(self, doc: Document, blocks: list[Block]) -> list[Chunk]: ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, k: int, doc_id: str | None = None) -> list[RetrievedChunk]: ...


@runtime_checkable
class Generator(Protocol):
    model: str

    def generate(self, question: str, chunks: list[Chunk], qid: str, config_id: str) -> GenerationResult: ...
