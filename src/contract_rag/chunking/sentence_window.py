"""One-sentence chunks with expanded neighboring-sentence context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

from contract_rag.chunking.base import BaseChunker, validate_inputs
from contract_rag.schemas import Block, Chunk, Document

_ABBREVIATIONS = {
    "art",
    "corp",
    "dr",
    "e.g",
    "i.e",
    "inc",
    "llc",
    "ltd",
    "mr",
    "mrs",
    "no",
    "sec",
    "u.s",
    "vs",
}


@dataclass(frozen=True)
class SentenceUnit:
    start: int
    end: int
    block: Block


def _sentence_tokenizer() -> PunktSentenceTokenizer:
    parameters = PunktParameters()
    parameters.abbrev_types = set(_ABBREVIATIONS)
    return PunktSentenceTokenizer(parameters)


class SentenceWindowChunker(BaseChunker):
    strategy: Literal["sentence_window"] = "sentence_window"

    def __init__(
        self,
        window_size: int = 3,
        min_unit_tokens: int = 5,
        encoding_name: str = "cl100k_base",
    ) -> None:
        super().__init__(encoding_name)
        if window_size < 0:
            raise ValueError("window_size must be non-negative")
        if min_unit_tokens < 1:
            raise ValueError("min_unit_tokens must be positive")
        self.window_size = window_size
        self.min_unit_tokens = min_unit_tokens
        self.tokenizer = _sentence_tokenizer()

    def _raw_units(self, blocks: list[Block]) -> list[SentenceUnit]:
        units: list[SentenceUnit] = []
        for block in blocks:
            if block.block_type in {"heading", "table", "other"}:
                units.append(SentenceUnit(block.char_start, block.char_end, block))
                continue
            spans = list(self.tokenizer.span_tokenize(block.text)) or [(0, len(block.text))]
            for local_start, local_end in spans:
                while local_start < local_end and block.text[local_start].isspace():
                    local_start += 1
                while local_end > local_start and block.text[local_end - 1].isspace():
                    local_end -= 1
                if local_start < local_end:
                    units.append(
                        SentenceUnit(block.char_start + local_start, block.char_start + local_end, block)
                    )
        return units

    def _units(self, blocks: list[Block]) -> list[SentenceUnit]:
        """Attach headings and very short sentence fragments to the following unit."""
        units: list[SentenceUnit] = []
        pending: list[SentenceUnit] = []
        for unit in self._raw_units(blocks):
            text = unit.block.text[unit.start - unit.block.char_start : unit.end - unit.block.char_start]
            is_prefix = unit.block.block_type == "heading" or (
                unit.block.block_type not in {"table", "other"}
                and self.token_count(text) < self.min_unit_tokens
            )
            if is_prefix:
                pending.append(unit)
                continue
            if pending:
                unit = SentenceUnit(pending[0].start, unit.end, unit.block)
                pending.clear()
            units.append(unit)

        if pending:
            if units:
                previous = units[-1]
                units[-1] = SentenceUnit(previous.start, pending[-1].end, previous.block)
            else:
                units.append(SentenceUnit(pending[0].start, pending[-1].end, pending[-1].block))
        return units

    def chunk(self, doc: Document, blocks: list[Block]) -> list[Chunk]:
        ordered = validate_inputs(doc, blocks)
        units = self._units(ordered)
        chunks: list[Chunk] = []
        for index, unit in enumerate(units):
            context_start = units[max(0, index - self.window_size)].start
            context_end = units[min(len(units) - 1, index + self.window_size)].end
            chunks.append(
                self.make_chunk(
                    doc,
                    ordered,
                    index,
                    unit.start,
                    unit.end,
                    context_text=doc.full_text[context_start:context_end],
                    section_path=list(unit.block.section_path),
                )
            )
        return chunks
