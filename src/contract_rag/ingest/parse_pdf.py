"""Layout-aware CUAD PDF parser built on PyMuPDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
import yaml

from contract_rag.ingest.download import CuadRecord
from contract_rag.ingest.normalize import normalize_inline, normalize_lines
from contract_rag.schemas import Block, BlockType, Document

_CLAUSE_PREFIX = re.compile(
    r"^(?P<label>(?:section\s+)?\d+(?:\.\d+)*\.?|article\s+(?:[-–—]\s*)?(?:[ivxlcdm]+|\d+))"
    r"\s+(?P<body>.+)$",
    re.IGNORECASE,
)
_LIST_PREFIX = re.compile(r"^(?:[-•▪‣◦]\s+|\([a-zivxlcdm]+\)\s+|[a-z]\)\s+)", re.IGNORECASE)
_PAGE_NUMBER = re.compile(r"^(?:page\s*)?[-–—]?\s*\d+\s*[-–—]?$", re.IGNORECASE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.;:])\s+")
_DEFINED_TERM = re.compile(r"^(?P<title>[“\"][^”\"]{1,80}[”\"])(?P<body>\s+means\b.*)$", re.IGNORECASE)
_UPPERCASE_TITLE = re.compile(r"^(?P<title>[A-Z][A-Z0-9 &/()-]{2,80})(?P<body>\s+.*)$")
_PARSER_CACHE_VERSION = 2


@dataclass(frozen=True)
class LayoutBlock:
    page: int
    bbox: tuple[float, float, float, float]
    text: str
    max_font_size: float
    bold_ratio: float
    is_table: bool = False


@dataclass(frozen=True)
class ParsedUnit:
    page: int
    text: str
    block_type: BlockType
    heading_level: int | None = None


def _intersects(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> bool:
    x_overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    y_overlap = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    first_area = max(1.0, (first[2] - first[0]) * (first[3] - first[1]))
    return (x_overlap * y_overlap) / first_area >= 0.5


def table_to_markdown(rows: Iterable[Iterable[str | None]]) -> str:
    """Render extracted table cells as deterministic Markdown."""
    materialized = [[normalize_inline(cell or "").replace("|", r"\|") for cell in row] for row in rows]
    materialized = [row for row in materialized if any(row)]
    if not materialized:
        return ""
    width = max(len(row) for row in materialized)
    padded = [row + [""] * (width - len(row)) for row in materialized]
    header = padded[0]
    if not any(header):
        header = [f"Column {index}" for index in range(1, width + 1)]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in padded[1:])
    return "\n".join(lines)


def _line_text(line: dict[str, Any]) -> str:
    return "".join(str(span.get("text", "")) for span in line.get("spans", []))


def _extract_text_blocks(page: pymupdf.Page) -> list[LayoutBlock]:
    flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES
    page_dict = page.get_text("dict", flags=flags, sort=True)
    blocks: list[LayoutBlock] = []
    for raw in page_dict.get("blocks", []):
        lines = raw.get("lines", [])
        text = normalize_lines([_line_text(line) for line in lines])
        if not text:
            continue
        spans = [span for line in lines for span in line.get("spans", []) if span.get("text", "").strip()]
        weighted_chars = sum(len(str(span["text"]).strip()) for span in spans)
        bold_chars = sum(
            len(str(span["text"]).strip())
            for span in spans
            if "bold" in str(span.get("font", "")).casefold() or int(span.get("flags", 0)) & 16
        )
        blocks.append(
            LayoutBlock(
                page=page.number + 1,
                bbox=tuple(float(value) for value in raw["bbox"]),
                text=text,
                max_font_size=max((float(span.get("size", 0.0)) for span in spans), default=0.0),
                bold_ratio=bold_chars / max(1, weighted_chars),
            )
        )
    return blocks


def _extract_tables(page: pymupdf.Page) -> list[LayoutBlock]:
    tables: list[LayoutBlock] = []
    for table in page.find_tables().tables:
        markdown = table_to_markdown(table.extract())
        if not markdown:
            continue
        tables.append(
            LayoutBlock(
                page=page.number + 1,
                bbox=tuple(float(value) for value in table.bbox),
                text=markdown,
                max_font_size=0.0,
                bold_ratio=0.0,
                is_table=True,
            )
        )
    return tables


def _repetition_key(text: str) -> str:
    normalized = normalize_inline(text).casefold()
    return re.sub(r"\d+", "#", normalized)


def _repeated_margin_keys(
    pages: list[list[LayoutBlock]], page_heights: list[float], margin_ratio: float, min_pages: int
) -> set[str]:
    occurrences: dict[str, set[int]] = defaultdict(set)
    for page_number, (blocks, height) in enumerate(zip(pages, page_heights, strict=True), start=1):
        for block in blocks:
            in_margin = block.bbox[1] <= height * margin_ratio or block.bbox[3] >= height * (1 - margin_ratio)
            if in_margin and len(block.text) <= 200:
                occurrences[_repetition_key(block.text)].add(page_number)
    threshold = max(min_pages, math.ceil(len(pages) * 0.5))
    return {key for key, seen_pages in occurrences.items() if len(seen_pages) >= threshold}


def _body_font_size(blocks: list[LayoutBlock]) -> float:
    sizes = [
        round(block.max_font_size, 1) for block in blocks if not block.is_table and len(block.text) >= 40
    ]
    return statistics.median(sizes) if sizes else 10.0


def _numbered_heading(text: str) -> tuple[str, str | None, int] | None:
    match = _CLAUSE_PREFIX.match(text)
    if not match:
        return None
    label = match.group("label")
    body = match.group("body").strip()
    level = 1 if label.casefold().startswith("article") else label.rstrip(".").count(".") + 1

    title_candidate = body.rstrip(".:").strip()
    letters = [character for character in title_candidate if character.isalpha()]
    mostly_upper = bool(letters) and sum(character.isupper() for character in letters) / len(letters) >= 0.8
    title_like = title_candidate.istitle() or mostly_upper or body.endswith(":")
    if len(body) <= 100 and len(body.split()) <= 12 and title_like:
        return text, None, level

    first_part = _SENTENCE_BOUNDARY.split(body, maxsplit=1)
    first_title = first_part[0].rstrip(".:").strip()
    if (
        len(first_part) == 2
        and len(first_part[0]) <= 100
        and len(first_part[0].split()) <= 12
        and (first_title.istitle() or first_part[0].endswith(":") or first_title.isupper())
    ):
        return f"{label} {first_part[0]}", first_part[1], level
    defined_term = _DEFINED_TERM.match(body)
    if defined_term:
        return f"{label} {defined_term.group('title')}", defined_term.group("body").strip(), level
    uppercase_title = _UPPERCASE_TITLE.match(body)
    if uppercase_title:
        return (
            f"{label} {uppercase_title.group('title').strip()}",
            uppercase_title.group("body").strip(),
            level,
        )
    return label, body, level


def _looks_like_visual_heading(block: LayoutBlock, body_size: float) -> bool:
    text = block.text
    if len(text) > 180 or len(text.split()) > 24:
        return False
    letters = [character for character in text if character.isalpha()]
    is_upper = bool(letters) and sum(character.isupper() for character in letters) / len(letters) >= 0.85
    emphasized = block.bold_ratio >= 0.45 or block.max_font_size >= body_size * 1.15
    terminal_sentence = text.endswith(("?", "!")) or (text.endswith(".") and len(text.split()) > 12)
    return not terminal_sentence and (emphasized or (is_upper and len(text.split()) <= 16))


def _classify(block: LayoutBlock, body_size: float) -> list[ParsedUnit]:
    if block.is_table:
        return [ParsedUnit(page=block.page, text=block.text, block_type="table")]

    numbered = _numbered_heading(block.text)
    if numbered:
        heading, remainder, level = numbered
        units = [ParsedUnit(page=block.page, text=heading, block_type="heading", heading_level=level)]
        if remainder:
            units.append(ParsedUnit(page=block.page, text=remainder, block_type="paragraph"))
        return units
    if _LIST_PREFIX.match(block.text):
        return [ParsedUnit(page=block.page, text=block.text, block_type="list_item")]
    if _looks_like_visual_heading(block, body_size):
        return [ParsedUnit(page=block.page, text=block.text, block_type="heading", heading_level=1)]
    return [ParsedUnit(page=block.page, text=block.text, block_type="paragraph")]


def _with_offsets(doc_id: str, units: list[ParsedUnit]) -> tuple[str, list[Block]]:
    full_text = ""
    blocks: list[Block] = []
    section_stack: list[str] = []
    for index, unit in enumerate(units):
        if unit.block_type == "heading":
            level = unit.heading_level or 1
            section_stack = section_stack[: level - 1]
            section_stack.append(unit.text)
        if full_text:
            full_text += "\n\n"
        char_start = len(full_text)
        full_text += unit.text
        blocks.append(
            Block(
                doc_id=doc_id,
                block_id=f"{doc_id}::b{index:05d}",
                block_type=unit.block_type,
                text=unit.text,
                page=unit.page,
                section_path=list(section_stack),
                char_start=char_start,
                char_end=len(full_text),
            )
        )
    return full_text, blocks


def parse_pdf(
    record: CuadRecord,
    repo_root: Path,
    *,
    detect_tables: bool = True,
    remove_repeated_margins: bool = True,
    margin_ratio: float = 0.1,
    min_repeated_pages: int = 2,
    min_text_chars: int = 20,
) -> tuple[Document, list[Block]]:
    """Parse one PDF into a canonical document and offset-bearing blocks."""
    source = repo_root / record.pdf_path
    with pymupdf.open(source) as pdf:
        page_heights = [float(page.rect.height) for page in pdf]
        pages: list[list[LayoutBlock]] = []
        for page in pdf:
            text_blocks = _extract_text_blocks(page)
            tables = _extract_tables(page) if detect_tables else []
            text_blocks = [
                block
                for block in text_blocks
                if not any(_intersects(block.bbox, table.bbox) for table in tables)
            ]
            pages.append(sorted([*text_blocks, *tables], key=lambda block: (block.bbox[1], block.bbox[0])))

        repeated = (
            _repeated_margin_keys(pages, page_heights, margin_ratio, min_repeated_pages)
            if remove_repeated_margins and len(pages) >= min_repeated_pages
            else set()
        )
        layout_blocks = [
            block
            for page_blocks in pages
            for block in page_blocks
            if _repetition_key(block.text) not in repeated and not _PAGE_NUMBER.fullmatch(block.text)
        ]
        body_size = _body_font_size(layout_blocks)
        units = [unit for block in layout_blocks for unit in _classify(block, body_size) if unit.text]
        full_text, blocks = _with_offsets(record.doc_id, units)
        if len(full_text.strip()) < min_text_chars:
            raise ValueError(f"Extracted only {len(full_text.strip())} text characters")
        document = Document(
            doc_id=record.doc_id,
            title=record.title,
            source_path=record.pdf_path,
            num_pages=pdf.page_count,
            contract_type=record.contract_type,
            full_text=full_text,
            parser="pymupdf",
        )
    return document, blocks


def _load_manifest(path: Path) -> list[CuadRecord]:
    return [CuadRecord(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _cache_paths(cache_dir: Path, doc_id: str) -> tuple[Path, Path]:
    return cache_dir / f"{doc_id}.document.json", cache_dir / f"{doc_id}.blocks.jsonl"


def _write_cached(cache_dir: Path, document: Document, blocks: list[Block]) -> None:
    document_path, blocks_path = _cache_paths(cache_dir, document.doc_id)
    _write_atomic(document_path, json.dumps(document.model_dump(), sort_keys=True) + "\n")
    _write_atomic(
        blocks_path,
        "".join(json.dumps(block.model_dump(), sort_keys=True) + "\n" for block in blocks),
    )


def _read_cached(cache_dir: Path, doc_id: str) -> tuple[Document, list[Block]]:
    document_path, blocks_path = _cache_paths(cache_dir, doc_id)
    document = Document.model_validate_json(document_path.read_text(encoding="utf-8"))
    blocks = [
        Block.model_validate_json(line) for line in blocks_path.read_text(encoding="utf-8").splitlines()
    ]
    return document, blocks


def parse_manifest(config: dict[str, Any], repo_root: Path, *, force: bool = False) -> dict[str, Any]:
    """Parse a manifest with per-document atomic caching and assemble JSONL outputs."""
    started = time.perf_counter()
    records = _load_manifest(repo_root / config["manifest_path"])
    failures: list[dict[str, str]] = []

    parser_options = {
        "detect_tables": bool(config.get("detect_tables", True)),
        "remove_repeated_margins": bool(config.get("remove_repeated_margins", True)),
        "margin_ratio": float(config.get("margin_ratio", 0.1)),
        "min_repeated_pages": int(config.get("min_repeated_pages", 2)),
        "min_text_chars": int(config.get("min_text_chars", 20)),
    }
    cache_payload = json.dumps(
        {"version": _PARSER_CACHE_VERSION, **parser_options}, sort_keys=True, separators=(",", ":")
    )
    cache_key = hashlib.sha256(cache_payload.encode()).hexdigest()[:12]
    cache_dir = repo_root / config["cache_dir"] / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    for position, record in enumerate(records, start=1):
        document_path, blocks_path = _cache_paths(cache_dir, record.doc_id)
        if force or not (document_path.is_file() and blocks_path.is_file()):
            try:
                document, blocks = parse_pdf(record, repo_root, **parser_options)
                _write_cached(cache_dir, document, blocks)
            except Exception as exc:  # keep long corpus jobs moving and report every failed document
                failures.append({"doc_id": record.doc_id, "source_path": record.pdf_path, "error": repr(exc)})
        if position % 10 == 0 or position == len(records):
            print(f"Parsed or resumed {position}/{len(records)} documents")

    documents: list[Document] = []
    all_blocks: list[Block] = []
    failed_ids = {failure["doc_id"] for failure in failures}
    for record in records:
        if record.doc_id in failed_ids:
            continue
        try:
            document, blocks = _read_cached(cache_dir, record.doc_id)
        except (OSError, ValueError) as exc:
            failures.append({"doc_id": record.doc_id, "source_path": record.pdf_path, "error": repr(exc)})
            continue
        documents.append(document)
        all_blocks.extend(blocks)

    _write_atomic(
        repo_root / config["documents_path"],
        "".join(json.dumps(document.model_dump(), sort_keys=True) + "\n" for document in documents),
    )
    _write_atomic(
        repo_root / config["blocks_path"],
        "".join(json.dumps(block.model_dump(), sort_keys=True) + "\n" for block in all_blocks),
    )
    _write_atomic(
        repo_root / config["failures_path"],
        "".join(json.dumps(failure, sort_keys=True) + "\n" for failure in failures),
    )

    block_types = Counter(block.block_type for block in all_blocks)
    report = {
        "requested_documents": len(records),
        "parsed_documents": len(documents),
        "failed_documents": len(failures),
        "failure_rate": len(failures) / max(1, len(records)),
        "pages": sum(document.num_pages for document in documents),
        "blocks": len(all_blocks),
        "block_types": dict(sorted(block_types.items())),
        "table_documents": len({block.doc_id for block in all_blocks if block.block_type == "table"}),
        "cache_key": cache_key,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    _write_atomic(repo_root / config["report_path"], json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/parse_pdf.yaml"))
    parser.add_argument("--manifest", type=Path, help="Override the configured input manifest")
    parser.add_argument("--force", action="store_true", help="Reparse documents even when cached")
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.manifest:
        config = {**config, "manifest_path": args.manifest.as_posix()}
    report = parse_manifest(config, repo_root, force=args.force)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
