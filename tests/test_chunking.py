import json
from pathlib import Path

from contract_rag.chunking import FixedChunker, SectionChunker, SentenceWindowChunker
from contract_rag.ingest.download import CuadRecord
from contract_rag.ingest.parse_pdf import parse_pdf
from contract_rag.schemas import Block, Chunker, Document

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "cuad_mini"


def _fixture() -> list[tuple[Document, list[Block]]]:
    records = [
        CuadRecord(**json.loads(line))
        for line in (FIXTURE_ROOT / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return [parse_pdf(record, REPO_ROOT, detect_tables=False) for record in records]


def test_all_chunkers_preserve_offsets_and_stable_ids() -> None:
    chunkers: list[Chunker] = [
        FixedChunker(),
        SentenceWindowChunker(),
        SectionChunker(),
    ]
    for document, blocks in _fixture():
        for chunker in chunkers:
            chunks = chunker.chunk(document, blocks)
            assert chunks
            assert [chunk.chunk_id for chunk in chunks] == [
                f"{document.doc_id}::{chunker.strategy}::{index:05d}" for index in range(len(chunks))
            ]
            for chunk in chunks:
                assert document.full_text[chunk.char_start : chunk.char_end] == chunk.text
                assert 1 <= chunk.page_start <= chunk.page_end <= document.num_pages


def test_fixed_chunks_obey_token_limit_and_overlap() -> None:
    text = " ".join(f"term{index}" for index in range(100))
    document = Document(doc_id="fixed", title="Fixed", source_path="fixed.pdf", num_pages=1, full_text=text)
    block = Block(
        doc_id="fixed",
        block_id="fixed::b00000",
        block_type="paragraph",
        text=text,
        page=1,
        char_start=0,
        char_end=len(text),
    )
    chunker = FixedChunker(max_tokens=20, overlap_tokens=5)
    chunks = chunker.chunk(document, [block])

    assert len(chunks) > 1
    assert all(chunk.token_count <= 20 for chunk in chunks)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        overlap = document.full_text[current.char_start : previous.char_end]
        assert chunker.token_count(overlap) == 5


def test_sentence_window_uses_one_sentence_with_neighbor_context() -> None:
    text = "One sentence. Two sentences. Three sentences. Four sentences. Five sentences."
    document = Document(
        doc_id="sentences", title="Sentences", source_path="sentences.pdf", num_pages=1, full_text=text
    )
    block = Block(
        doc_id="sentences",
        block_id="sentences::b00000",
        block_type="paragraph",
        text=text,
        page=1,
        section_path=["Terms"],
        char_start=0,
        char_end=len(text),
    )
    chunks = SentenceWindowChunker(window_size=1).chunk(document, [block])

    assert [chunk.text for chunk in chunks] == [
        "One sentence.",
        "Two sentences.",
        "Three sentences.",
        "Four sentences.",
        "Five sentences.",
    ]
    assert chunks[2].context_text == "Two sentences. Three sentences. Four sentences."
    assert all(chunk.section_path == ["Terms"] for chunk in chunks)


def test_all_chunkers_propagate_table_metadata() -> None:
    text = "| Fee | Amount |\n| --- | --- |\n| Base | $10 |"
    document = Document(doc_id="table", title="Table", source_path="table.pdf", num_pages=2, full_text=text)
    block = Block(
        doc_id="table",
        block_id="table::b00000",
        block_type="table",
        text=text,
        page=2,
        section_path=["Fees"],
        char_start=0,
        char_end=len(text),
    )

    for chunker in (FixedChunker(), SentenceWindowChunker(), SectionChunker()):
        chunks = chunker.chunk(document, [block])

        assert all(chunk.has_table for chunk in chunks)
        assert all((chunk.page_start, chunk.page_end) == (2, 2) for chunk in chunks)


def test_section_chunker_keeps_short_sections_separate() -> None:
    parts = ["1. First", "First body.", "2. Second", "Second body."]
    full_text = "\n\n".join(parts)
    blocks = []
    cursor = 0
    for index, text in enumerate(parts):
        start = full_text.index(text, cursor)
        end = start + len(text)
        heading = index % 2 == 0
        path = [text] if heading else [parts[index - 1]]
        blocks.append(
            Block(
                doc_id="sections",
                block_id=f"sections::b{index:05d}",
                block_type="heading" if heading else "paragraph",
                text=text,
                page=1,
                section_path=path,
                char_start=start,
                char_end=end,
            )
        )
        cursor = end
    document = Document(
        doc_id="sections", title="Sections", source_path="sections.pdf", num_pages=1, full_text=full_text
    )

    chunks = SectionChunker(max_tokens=100).chunk(document, blocks)

    assert [chunk.text for chunk in chunks] == [
        "1. First\n\nFirst body.",
        "2. Second\n\nSecond body.",
    ]
    assert [chunk.section_path for chunk in chunks] == [["1. First"], ["2. Second"]]


def test_section_chunker_subdivides_long_section_without_crossing_next_heading() -> None:
    first_heading = "1. Long Section"
    long_body = " ".join(f"obligation{index}" for index in range(80))
    second_heading = "2. Short Section"
    second_body = "Brief obligation."
    parts = [first_heading, long_body, second_heading, second_body]
    full_text = "\n\n".join(parts)
    blocks = []
    cursor = 0
    for index, text in enumerate(parts):
        start = full_text.index(text, cursor)
        end = start + len(text)
        heading = index in {0, 2}
        section_heading = first_heading if index < 2 else second_heading
        blocks.append(
            Block(
                doc_id="long-section",
                block_id=f"long-section::b{index:05d}",
                block_type="heading" if heading else "paragraph",
                text=text,
                page=1,
                section_path=[section_heading],
                char_start=start,
                char_end=end,
            )
        )
        cursor = end
    document = Document(
        doc_id="long-section",
        title="Long Section",
        source_path="long-section.pdf",
        num_pages=1,
        full_text=full_text,
    )

    chunks = SectionChunker(max_tokens=20).chunk(document, blocks)

    assert len(chunks) > 3
    assert all(chunk.token_count <= 20 for chunk in chunks)
    assert all(not (first_heading in chunk.text and second_heading in chunk.text) for chunk in chunks)
    assert chunks[-1].text == f"{second_heading}\n\n{second_body}"
