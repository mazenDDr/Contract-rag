import json
from pathlib import Path

import pymupdf

from contract_rag.ingest.download import CuadRecord
from contract_rag.ingest.normalize import whitespace_equivalent
from contract_rag.ingest.parse_pdf import LayoutBlock, _classify, parse_pdf, table_to_markdown

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "cuad_mini"


def _fixture_records() -> list[CuadRecord]:
    return [
        CuadRecord(**json.loads(line))
        for line in (FIXTURE_ROOT / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_parse_fixture_preserves_exact_block_offsets() -> None:
    for record in _fixture_records():
        document, blocks = parse_pdf(record, REPO_ROOT)

        assert document.doc_id == record.doc_id
        assert document.num_pages > 0
        assert blocks
        assert any(block.block_type == "heading" for block in blocks)
        for index, block in enumerate(blocks):
            assert block.block_id == f"{record.doc_id}::b{index:05d}"
            extracted = document.full_text[block.char_start : block.char_end]
            assert whitespace_equivalent(extracted, block.text)


def test_parse_pdf_removes_repeated_headers_and_page_numbers(tmp_path: Path) -> None:
    source = tmp_path / "repeated.pdf"
    pdf = pymupdf.open()
    for page_number in range(1, 4):
        page = pdf.new_page()
        page.insert_text((72, 30), "CONFIDENTIAL CONTRACT")
        page.insert_text((72, 120), f"{page_number}. Section {page_number}", fontsize=14)
        page.insert_text((72, 160), f"Body text unique to page {page_number}.")
        page.insert_text((280, 770), f"Page {page_number}")
    pdf.save(source)
    pdf.close()
    record = CuadRecord(
        doc_id="repeated",
        title="Repeated",
        contract_type="Test",
        part="Part_I",
        pdf_path=source.relative_to(tmp_path).as_posix(),
        txt_path="unused.txt",
        pdf_sha256="0" * 64,
        pdf_size_bytes=source.stat().st_size,
    )

    document, blocks = parse_pdf(record, tmp_path, detect_tables=False)

    assert "CONFIDENTIAL CONTRACT" not in document.full_text
    assert "Page 1" not in document.full_text
    assert len(blocks) == 6


def test_table_to_markdown_normalizes_cells_and_escapes_pipes() -> None:
    markdown = table_to_markdown([["Fee", "Amount"], ["Base | recurring", "$ 10"]])

    assert markdown == "| Fee | Amount |\n| --- | --- |\n| Base \\| recurring | $ 10 |"


def _layout(text: str) -> LayoutBlock:
    return LayoutBlock(
        page=1,
        bbox=(72.0, 72.0, 400.0, 90.0),
        text=text,
        max_font_size=14.0,
        bold_ratio=1.0,
    )


def test_heading_title_case_allows_minor_words() -> None:
    units = _classify(_layout("2. Compensation and Expenses."), body_size=10.0)

    assert [(unit.block_type, unit.text) for unit in units] == [("heading", "2. Compensation and Expenses.")]


def test_large_level_one_numbers_are_not_headings() -> None:
    for text in ("1820 Gateway Drive", "2020 Annual Fee", "75 Dollars"):
        units = _classify(_layout(text), body_size=10.0)

        assert [unit.block_type for unit in units] == ["paragraph"]


def test_table_of_contents_line_is_other_not_heading() -> None:
    units = _classify(_layout("2. Compensation and Expenses........ 4"), body_size=10.0)

    assert [(unit.block_type, unit.text) for unit in units] == [
        ("other", "2. Compensation and Expenses........ 4")
    ]
