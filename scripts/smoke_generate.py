"""Smoke-test the local generator on a hand-made contract excerpt set.

Usage: python scripts/smoke_generate.py [--model qwen3.5:4b] [--think]
Checks the three behaviors we care about: grounded answer, correct citations, abstention.
"""

from __future__ import annotations

import argparse

from contract_rag.generation.generator import GeneratorConfig, OllamaGenerator
from contract_rag.schemas import Chunk

EXCERPTS = [
    (
        "15. GOVERNING LAW",
        "This Agreement shall be governed by and construed in accordance with the laws of "
        "the State of Delaware, without regard to its conflict of laws principles.",
    ),
    (
        "12. TERMINATION",
        "12.2 Either party may terminate this Agreement for convenience upon ninety (90) days' "
        "prior written notice to the other party.",
    ),
    (
        "12. TERMINATION",
        "12.3 Upon termination, Distributor shall return all Confidential Information within "
        "thirty (30) days.",
    ),
    (
        "9. LIMITATION OF LIABILITY",
        "In no event shall either party's aggregate liability exceed the fees paid "
        "by Distributor in the twelve (12) months preceding the claim.",
    ),
]
QUESTIONS = [
    ("Which state's law governs this agreement?", "answerable"),
    (
        "Can the agreement be terminated for convenience, and what must happen after termination?",
        "multi-span",
    ),
    ("What is the royalty rate payable by the Distributor?", "unanswerable"),
]


def build_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"smoke::section::{i:05d}",
            doc_id="smoke",
            strategy="section",
            text=text,
            section_path=[section],
            page_start=i + 1,
            page_end=i + 1,
            char_start=0,
            char_end=len(text),
            token_count=0,
        )
        for i, (section, text) in enumerate(EXCERPTS)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--think", action="store_true", help="enable the model's hidden reasoning")
    args = parser.parse_args()

    gen = OllamaGenerator(GeneratorConfig(model=args.model, think=args.think))
    chunks = build_chunks()
    for i, (question, kind) in enumerate(QUESTIONS, start=1):
        r = gen.generate(question, chunks, qid=f"smoke{i}", config_id="smoke")
        print(f"\n[{kind}] {question}")
        print(f"  answer:    {r.answer}")
        print(f"  cited:     {r.cited_chunk_ids}  invalid={r.invalid_citations}  abstained={r.abstained}")
        print(
            f"  tokens:    {r.prompt_tokens} in / {r.completion_tokens} out   latency: {r.latency_ms:.0f} ms"
        )


if __name__ == "__main__":
    main()
