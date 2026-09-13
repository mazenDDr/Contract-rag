"""Configuration for building and loading retrieval indexes."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from contract_rag.schemas import Chunk

StrategyName = Literal["fixed", "sentence_window", "section"]


class BM25Config(BaseModel):
    k1: float = 1.5
    b: float = 0.75
    stemmer: str | None = "english"  # PyStemmer language; None disables stemming
    stopwords: str | None = "en"


class DenseModelConfig(BaseModel):
    name: str  # sentence-transformers model id
    query_prefix: str = ""  # e5 needs "query: "; bge uses an instruction for queries only
    passage_prefix: str = ""  # e5 needs "passage: "


class DenseConfig(BaseModel):
    device: str = "auto"  # "auto" picks Apple's MPS when available, else CPU
    batch_size: int = Field(default=64, gt=0)
    models: dict[str, DenseModelConfig] = Field(default_factory=dict)


class IndexConfig(BaseModel):
    chunks_dir: Path = Path("data/processed/chunks")
    index_dir: Path = Path("indexes")
    strategies: list[StrategyName] = Field(default_factory=lambda: ["fixed", "sentence_window", "section"])
    bm25: BM25Config = Field(default_factory=BM25Config)
    dense: DenseConfig = Field(default_factory=DenseConfig)
    smoke_queries: list[str] = Field(default_factory=list)


def collection_name(strategy: str, model_key: str) -> str:
    """One Qdrant collection per (chunking strategy, embedding model)."""
    return f"{strategy}__{model_key}"


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
