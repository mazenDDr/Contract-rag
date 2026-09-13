"""Prompt that forces answers grounded in numbered context excerpts, so faithfulness is checkable.

Chunks are shown to the model as [1]..[k] instead of their full chunk_ids: small models copy short
numbers far more reliably than ids like "cuad_0042::section::00031". Alias n always means the n-th
chunk passed in, i.e. RetrievalResult.final[n-1], and `parse_citations` maps aliases back to chunk_ids.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from contract_rag.schemas import Chunk

ABSTAIN_ANSWER = "The provided contract excerpts do not contain the answer."

SYSTEM_PROMPT = """You are a careful legal analyst answering questions about a contract.

Rules:
1. Use ONLY the numbered context excerpts. Never use outside knowledge.
2. End every sentence of your answer with the excerpt number(s) that support it, like [2] or [1][3].
3. Use the contract's exact wording for key terms (dates, amounts, notice periods, governing law).
4. If the excerpts do not contain the answer, set "abstained" to true and do not guess.
5. Answer only what was asked. Do not add facts from excerpts that are unrelated to the question.
6. Be concise: at most 4 sentences.

Respond as JSON: {"answer": "...", "abstained": false}"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "abstained": {"type": "boolean"}},
    "required": ["answer", "abstained"],
}

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def format_context(chunks: Sequence[Chunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        where = " > ".join(c.section_path) or "no section"
        pages = f"p.{c.page_start}" if c.page_start == c.page_end else f"pp.{c.page_start}-{c.page_end}"
        parts.append(f"[{i}] ({where}, {pages})\n{(c.context_text or c.text).strip()}")
    return "\n\n".join(parts)


def build_messages(question: str, chunks: Sequence[Chunk]) -> list[dict[str, str]]:
    user = f"Context excerpts:\n\n{format_context(chunks)}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_citations(answer: str, chunks: Sequence[Chunk]) -> tuple[list[str], int]:
    """Map [n] / [n, m] markers back to chunk_ids in first-cited order. Returns (chunk_ids, n_invalid)."""
    cited: list[str] = []
    invalid = 0
    for match in _CITATION.finditer(answer):
        for num in match.group(1).split(","):
            n = int(num)
            if 1 <= n <= len(chunks):
                if chunks[n - 1].chunk_id not in cited:
                    cited.append(chunks[n - 1].chunk_id)
            else:
                invalid += 1
    return cited, invalid
