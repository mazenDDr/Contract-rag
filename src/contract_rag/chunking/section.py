"""Section-aware chunking that never merges across detected headings."""

from __future__ import annotations

from typing import Literal

from contract_rag.chunking.base import BaseChunker, validate_inputs
from contract_rag.schemas import Block, Chunk, Document


class SectionChunker(BaseChunker):
    strategy: Literal["section"] = "section"

    def __init__(self, max_tokens: int = 512, encoding_name: str = "cl100k_base") -> None:
        super().__init__(encoding_name)
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens

    @staticmethod
    def _sections(blocks: list[Block]) -> list[list[Block]]:
        sections: list[list[Block]] = []
        current: list[Block] = []
        for block in blocks:
            if block.block_type == "heading" and any(
                current_block.block_type != "heading" for current_block in current
            ):
                sections.append(current)
                current = []
            current.append(block)
        if current:
            sections.append(current)
        return sections

    def _subdivide(self, doc: Document, section: list[Block]) -> list[tuple[int, int]]:
        chunks: list[tuple[int, int]] = []
        pending: list[Block] = []

        def flush_pending() -> None:
            if pending:
                chunks.append((pending[0].char_start, pending[-1].char_end))
                pending.clear()

        for block in section:
            if self.token_count(block.text) > self.max_tokens:
                start = pending[0].char_start if pending else block.char_start
                pending.clear()
                chunks.extend(self.token_windows(doc.full_text, start, block.char_end, self.max_tokens))
                continue
            candidate_start = pending[0].char_start if pending else block.char_start
            if (
                pending
                and self.token_count(doc.full_text[candidate_start : block.char_end]) > self.max_tokens
            ):
                if all(pending_block.block_type == "heading" for pending_block in pending):
                    pending.clear()
                    chunks.extend(
                        self.token_windows(doc.full_text, candidate_start, block.char_end, self.max_tokens)
                    )
                    continue
                flush_pending()
            pending.append(block)
        flush_pending()
        return chunks

    def chunk(self, doc: Document, blocks: list[Block]) -> list[Chunk]:
        ordered = validate_inputs(doc, blocks)
        spans = [span for section in self._sections(ordered) for span in self._subdivide(doc, section)]
        return [self.make_chunk(doc, ordered, index, start, end) for index, (start, end) in enumerate(spans)]
