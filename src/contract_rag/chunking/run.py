"""Run configured chunking strategies over parsed documents."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from contract_rag.chunking.fixed import FixedChunker
from contract_rag.chunking.section import SectionChunker
from contract_rag.chunking.sentence_window import SentenceWindowChunker
from contract_rag.schemas import Block, Chunk, Chunker, Document

StrategyName = Literal["fixed", "sentence_window", "section"]


class FixedConfig(BaseModel):
    max_tokens: Annotated[int, Field(gt=0)] = 512
    overlap_tokens: Annotated[int, Field(ge=0)] = 64

    @model_validator(mode="after")
    def validate_overlap(self) -> FixedConfig:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        return self


class SentenceWindowConfig(BaseModel):
    window_size: Annotated[int, Field(ge=0)] = 3
    min_unit_tokens: Annotated[int, Field(gt=0)] = 5


class SectionConfig(BaseModel):
    max_tokens: Annotated[int, Field(gt=0)] = 512


class ChunkingRunConfig(BaseModel):
    documents_path: Path = Path("data/processed/documents.jsonl")
    blocks_path: Path = Path("data/processed/blocks.jsonl")
    output_dir: Path = Path("data/processed/chunks")
    stats_path: Path = Path("data/processed/chunk_stats.json")
    encoding_name: str = "cl100k_base"
    fixed: FixedConfig = Field(default_factory=FixedConfig)
    sentence_window: SentenceWindowConfig = Field(default_factory=SentenceWindowConfig)
    section: SectionConfig = Field(default_factory=SectionConfig)


def _load_inputs(config: ChunkingRunConfig, repo_root: Path) -> tuple[list[Document], dict[str, list[Block]]]:
    documents = [
        Document.model_validate_json(line)
        for line in (repo_root / config.documents_path).read_text(encoding="utf-8").splitlines()
        if line
    ]
    blocks_by_doc: dict[str, list[Block]] = defaultdict(list)
    for line in (repo_root / config.blocks_path).read_text(encoding="utf-8").splitlines():
        if line:
            block = Block.model_validate_json(line)
            blocks_by_doc[block.doc_id].append(block)
    return sorted(documents, key=lambda document: document.doc_id), blocks_by_doc


def _chunker(name: StrategyName, config: ChunkingRunConfig) -> Chunker:
    if name == "fixed":
        return FixedChunker(
            max_tokens=config.fixed.max_tokens,
            overlap_tokens=config.fixed.overlap_tokens,
            encoding_name=config.encoding_name,
        )
    if name == "sentence_window":
        return SentenceWindowChunker(
            window_size=config.sentence_window.window_size,
            min_unit_tokens=config.sentence_window.min_unit_tokens,
            encoding_name=config.encoding_name,
        )
    return SectionChunker(max_tokens=config.section.max_tokens, encoding_name=config.encoding_name)


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, chunks: list[Chunk]) -> None:
    _write_atomic(path, "".join(json.dumps(chunk.model_dump(), sort_keys=True) + "\n" for chunk in chunks))


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def chunk_stats(chunks: list[Chunk]) -> dict[str, object]:
    token_counts = [chunk.token_count for chunk in chunks]
    table_count = sum(chunk.has_table for chunk in chunks)
    return {
        "chunk_count": len(chunks),
        "token_percentiles": {
            "min": min(token_counts, default=0),
            "p25": _percentile(token_counts, 0.25),
            "p50": _percentile(token_counts, 0.50),
            "p75": _percentile(token_counts, 0.75),
            "p90": _percentile(token_counts, 0.90),
            "p95": _percentile(token_counts, 0.95),
            "p99": _percentile(token_counts, 0.99),
            "max": max(token_counts, default=0),
        },
        "table_chunks": table_count,
        "table_chunk_share": table_count / max(1, len(chunks)),
    }


def run_chunking(
    config: ChunkingRunConfig, repo_root: Path, strategies: list[StrategyName] | None = None
) -> dict[str, dict[str, object]]:
    selected = list(dict.fromkeys(strategies or ["fixed", "sentence_window", "section"]))
    documents, blocks_by_doc = _load_inputs(config, repo_root)
    report: dict[str, dict[str, object]] = {}
    output_dir = repo_root / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in selected:
        chunker = _chunker(name, config)
        chunks = [
            chunk
            for document in documents
            for chunk in chunker.chunk(document, blocks_by_doc[document.doc_id])
        ]
        _write_jsonl(output_dir / f"{name}.jsonl", chunks)
        report[name] = chunk_stats(chunks)
        print(f"{name}: wrote {len(chunks)} chunks")

    stats_path = repo_root / config.stats_path
    _write_atomic(stats_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/chunking.yaml"))
    parser.add_argument("--strategy", action="append", choices=("fixed", "sentence_window", "section"))
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    config = ChunkingRunConfig.model_validate(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    report = run_chunking(config, repo_root, args.strategy)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
