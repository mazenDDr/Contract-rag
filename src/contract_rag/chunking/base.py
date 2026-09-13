"""Shared offset and token helpers for chunking strategies."""

from __future__ import annotations

from collections.abc import Sequence

import tiktoken

from contract_rag.schemas import Block, Chunk, ChunkStrategy, Document


class BaseChunker:
    """Base implementation for exact-offset chunk construction."""

    strategy: ChunkStrategy

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding_name = encoding_name
        self.encoding = tiktoken.get_encoding(encoding_name)

    def token_count(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))

    def token_windows(
        self, text: str, start: int, end: int, max_tokens: int, overlap_tokens: int = 0
    ) -> list[tuple[int, int]]:
        """Find exact character slices bounded by token count."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= overlap_tokens < max_tokens:
            raise ValueError("overlap_tokens must be non-negative and smaller than max_tokens")
        if not 0 <= start <= end <= len(text):
            raise ValueError("Invalid character range")
        if start == end:
            return []

        windows: list[tuple[int, int]] = []
        cursor = start
        while cursor < end:
            window_end = self._largest_end(text, cursor, end, max_tokens)
            windows.append((cursor, window_end))
            if window_end == end:
                break
            next_cursor = self._suffix_start(text, cursor, window_end, overlap_tokens)
            if next_cursor <= cursor:
                raise RuntimeError("Token overlap prevented chunking progress")
            cursor = next_cursor
        return windows

    def _largest_end(self, text: str, start: int, limit: int, max_tokens: int) -> int:
        initial_width = max(32, max_tokens * 8)
        high = min(limit, start + initial_width)
        while high < limit and self.token_count(text[start:high]) <= max_tokens:
            width = high - start
            high = min(limit, start + width * 2)
        if self.token_count(text[start:high]) <= max_tokens:
            return high

        low = start + 1
        while low < high:
            middle = (low + high + 1) // 2
            if self.token_count(text[start:middle]) <= max_tokens:
                low = middle
            else:
                high = middle - 1
        if self.token_count(text[start:low]) > max_tokens:
            raise ValueError("A single character exceeds the configured token limit")
        return low

    def _suffix_start(self, text: str, start: int, end: int, overlap_tokens: int) -> int:
        if overlap_tokens == 0:
            return end
        low = start + 1
        high = end
        while low < high:
            middle = (low + high) // 2
            if self.token_count(text[middle:end]) <= overlap_tokens:
                high = middle
            else:
                low = middle + 1
        candidates = range(max(start + 1, low - 32), min(end, low + 32) + 1)
        counts = [(candidate, self.token_count(text[candidate:end])) for candidate in candidates]
        exact = [candidate for candidate, count in counts if count == overlap_tokens]
        if exact:
            return min(exact, key=lambda candidate: abs(candidate - low))
        valid = [(candidate, count) for candidate, count in counts if count < overlap_tokens]
        return max(valid, key=lambda item: item[1])[0] if valid else low

    def make_chunk(
        self,
        doc: Document,
        blocks: Sequence[Block],
        index: int,
        start: int,
        end: int,
        *,
        context_text: str | None = None,
        section_path: list[str] | None = None,
    ) -> Chunk:
        if not 0 <= start < end <= len(doc.full_text):
            raise ValueError(f"Invalid chunk offsets for {doc.doc_id}: {start}:{end}")
        overlapping = [block for block in blocks if block.char_start < end and block.char_end > start]
        if not overlapping:
            raise ValueError(f"Chunk {start}:{end} overlaps no blocks in {doc.doc_id}")
        text = doc.full_text[start:end]
        path = section_path if section_path is not None else common_section_path(overlapping)
        return Chunk(
            chunk_id=f"{doc.doc_id}::{self.strategy}::{index:05d}",
            doc_id=doc.doc_id,
            strategy=self.strategy,
            text=text,
            context_text=context_text,
            section_path=path,
            page_start=min(block.page for block in overlapping),
            page_end=max(block.page for block in overlapping),
            char_start=start,
            char_end=end,
            token_count=self.token_count(text),
            has_table=any(block.block_type == "table" for block in overlapping),
        )


def common_section_path(blocks: Sequence[Block]) -> list[str]:
    """Return the deepest section prefix shared by all overlapping blocks."""
    if not blocks:
        return []
    prefix = list(blocks[0].section_path)
    for block in blocks[1:]:
        shared = 0
        for left, right in zip(prefix, block.section_path, strict=False):
            if left != right:
                break
            shared += 1
        prefix = prefix[:shared]
    return prefix


def validate_inputs(doc: Document, blocks: Sequence[Block]) -> list[Block]:
    """Validate parser invariants and return blocks in canonical order."""
    ordered = sorted(blocks, key=lambda block: (block.char_start, block.char_end, block.block_id))
    for block in ordered:
        if block.doc_id != doc.doc_id:
            raise ValueError(f"Block {block.block_id} does not belong to {doc.doc_id}")
        if doc.full_text[block.char_start : block.char_end] != block.text:
            raise ValueError(f"Block {block.block_id} violates the parser offset invariant")
    return ordered
