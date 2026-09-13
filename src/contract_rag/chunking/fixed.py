"""Fixed-token chunking with overlap."""

from __future__ import annotations

from typing import Literal

from contract_rag.chunking.base import BaseChunker, validate_inputs
from contract_rag.schemas import Block, Chunk, Document


class FixedChunker(BaseChunker):
    strategy: Literal["fixed"] = "fixed"

    def __init__(
        self, max_tokens: int = 512, overlap_tokens: int = 64, encoding_name: str = "cl100k_base"
    ) -> None:
        super().__init__(encoding_name)
        if not 0 <= overlap_tokens < max_tokens:
            raise ValueError("overlap_tokens must be non-negative and smaller than max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, doc: Document, blocks: list[Block]) -> list[Chunk]:
        ordered = validate_inputs(doc, blocks)
        windows = self.token_windows(
            doc.full_text, 0, len(doc.full_text), self.max_tokens, self.overlap_tokens
        )
        return [
            self.make_chunk(doc, ordered, index, start, end) for index, (start, end) in enumerate(windows)
        ]
