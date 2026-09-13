"""Question answering over one contract: the project's best setup behind a small service.

Retrieval uses fixed 512-token chunks and BM25 inside the chosen contract, the setup that won the
retrieval ablation. The local generator answers from the top 8 excerpts, and every citation comes back
with its page, section and text, so a client can show exactly where an answer came from.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from contract_rag.generation.generator import GeneratorConfig, OllamaGenerator
from contract_rag.retrieval.config import IndexConfig, load_chunks
from contract_rag.retrieval.pipeline import RetrievalResources
from contract_rag.schemas import Chunk, Document, RetrievalConfig

_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _best_setup() -> RetrievalConfig:
    return RetrievalConfig(chunking="fixed", sparse=True, k_final=8, doc_filter=True)


class ServiceConfig(BaseModel):
    retrieval_config: Path = Path("configs/retrieval.yaml")
    documents_path: Path = Path("data/processed/documents.jsonl")
    retrieval: RetrievalConfig = Field(default_factory=_best_setup)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    max_question_chars: int = 500
    host: str = "127.0.0.1"
    port: int = 8000
    log_path: Path | None = None  # also append one JSON line per request here


class Contract(BaseModel):
    doc_id: str
    title: str  # readable, e.g. "Consulting Agreement · Adurobiotech, Inc · 2020"
    source: str  # the CUAD file name it came from
    pages: int


def display_title(source: str) -> str:
    """Turn a CUAD file name ('ADUROBIOTECH,INC_06_02_2020-EX-10.7-CONSULTING AGREEMENT') into a readable
    title: the agreement type (after the last exhibit code), the company (before the first '_'), the year."""
    if "_" not in source:
        return source
    company = source.split("_", 1)[0].replace(",", ", ").replace(",  ", ", ").strip()
    kind = re.split(r"EX-[0-9A-Za-z.()]+[-_ ]", source)[-1].strip(" -_").replace("_", " ") or source
    mostly_upper = lambda s: sum(c.isupper() for c in s) > sum(c.islower() for c in s)  # noqa: E731
    if mostly_upper(kind):
        kind = kind.title()
    if mostly_upper(company):
        company = company.title()
    year = re.search(r"(?<!\d)((?:19|20)\d{2})(?:\d{4})?(?!\d)", source)  # 2020, or 2019 in 20190815
    return " · ".join(p for p in (kind, company, year.group(1) if year else "") if p)


class Citation(BaseModel):
    n: int  # the number the answer uses, [n]
    chunk_id: str
    page_start: int
    page_end: int
    section: str
    text: str  # the excerpt the model was shown


class AskResponse(BaseModel):
    request_id: str
    doc_id: str
    question: str
    answer: str
    abstained: bool  # the model said the excerpts don't cover the question
    citations: list[Citation]
    invalid_citations: int  # citation markers that point at no excerpt
    truncated: bool  # the model hit its output limit
    model: str
    retrieval: str  # the retrieval setup's config id
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float  # 0 for a local model; kept so a hosted model can report its cost
    latency_ms: dict[str, float]  # per retrieval stage, "retrieval", "generation" and "total"


class UnknownContract(KeyError):
    pass


def cited_numbers(answer: str, n_excerpts: int) -> list[int]:
    """Excerpt numbers the answer cites, in order of first appearance, keeping only real excerpts."""
    seen: list[int] = []
    for group in _CITE.findall(answer):
        for n in (int(x) for x in group.split(",")):
            if 1 <= n <= n_excerpts and n not in seen:
                seen.append(n)
    return seen


def load_config(path: Path) -> ServiceConfig:
    return ServiceConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


class QAService:
    def __init__(
        self,
        config: ServiceConfig,
        repo_root: Path,
        resources_factory: Callable[[IndexConfig, Path], Any] = RetrievalResources,
        generator_factory: Callable[[GeneratorConfig], Any] = OllamaGenerator,
    ):
        self.config = config
        index_cfg = IndexConfig.model_validate(
            yaml.safe_load((repo_root / config.retrieval_config).read_text(encoding="utf-8"))
        )
        lines = (repo_root / config.documents_path).read_text(encoding="utf-8").splitlines()
        docs = [Document.model_validate_json(x) for x in lines if x.strip()]
        self._contracts = {
            d.doc_id: Contract(
                doc_id=d.doc_id, title=display_title(d.title), source=d.title, pages=d.num_pages
            )
            for d in docs
        }
        chunk_file = repo_root / index_cfg.chunks_dir / f"{config.retrieval.chunking}.jsonl"
        self._chunks: dict[str, Chunk] = {c.chunk_id: c for c in load_chunks(chunk_file)}
        self._resources = resources_factory(index_cfg, repo_root)
        self._pipeline = self._resources.pipeline(config.retrieval)
        self._generator = generator_factory(config.generator)
        self.retrieval_id = config.retrieval.config_id()
        self.model = config.generator.model
        self._lock = threading.Lock()  # one laptop GPU: answer one question at a time

    def contracts(self) -> list[Contract]:
        return sorted(self._contracts.values(), key=lambda c: c.title.lower())

    def ask(self, doc_id: str, question: str, request_id: str) -> AskResponse:
        if doc_id not in self._contracts:
            raise UnknownContract(doc_id)
        started = time.perf_counter()
        with self._lock:
            result = self._pipeline.run(question, qid=request_id, doc_id=doc_id)
            excerpts = [self._chunks[r.chunk_id] for r in result.final]
            gen = self._generator.generate(question, excerpts, request_id, self.retrieval_id)
        citations = [
            Citation(
                n=n,
                chunk_id=excerpts[n - 1].chunk_id,
                page_start=excerpts[n - 1].page_start,
                page_end=excerpts[n - 1].page_end,
                section=" > ".join(excerpts[n - 1].section_path),
                text=excerpts[n - 1].context_text or excerpts[n - 1].text,
            )
            for n in cited_numbers(gen.answer, len(excerpts))
        ]
        latency = {f"retrieval_{k}": round(v, 1) for k, v in result.latency_ms.items() if k != "total"}
        latency |= {
            "retrieval": round(result.latency_ms.get("total", 0.0), 1),
            "generation": round(gen.latency_ms, 1),
            "total": round((time.perf_counter() - started) * 1000, 1),
        }
        return AskResponse(
            request_id=request_id,
            doc_id=doc_id,
            question=question,
            answer=gen.answer,
            abstained=gen.abstained,
            citations=citations,
            invalid_citations=gen.invalid_citations,
            truncated=gen.truncated,
            model=gen.model,
            retrieval=self.retrieval_id,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            cost_usd=gen.cost_usd,
            latency_ms=latency,
        )

    def close(self) -> None:
        self._resources.close()
