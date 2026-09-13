import json
from collections import Counter
from io import BytesIO
from pathlib import Path

import fitz
import pytest

from contract_rag.ingest.download import CuadRecord, download_archive, stratified_subset


def _records() -> list[CuadRecord]:
    records = []
    for contract_type, count in (("A", 8), ("B", 4), ("C", 3)):
        for index in range(count):
            records.append(
                CuadRecord(
                    doc_id=f"{contract_type.lower()}-{index}",
                    title=f"Contract {contract_type}-{index}",
                    contract_type=contract_type,
                    part="Part_I",
                    pdf_path=f"{contract_type}-{index}.pdf",
                    txt_path=f"{contract_type}-{index}.txt",
                    pdf_sha256="0" * 64,
                    pdf_size_bytes=100 + index,
                )
            )
    return records


def test_stratified_subset_is_deterministic_and_covers_strata() -> None:
    first = stratified_subset(_records(), size=7, seed=42)
    second = stratified_subset(_records(), size=7, seed=42)

    assert [record.doc_id for record in first] == [record.doc_id for record in second]
    assert len(first) == 7
    assert set(Counter(record.contract_type for record in first)) == {"A", "B", "C"}


def test_cuad_mini_fixture_has_three_readable_contracts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fixture_root = repo_root / "tests" / "fixtures" / "cuad_mini"
    records = [json.loads(line) for line in (fixture_root / "manifest.jsonl").read_text().splitlines()]

    assert len(records) == 3
    assert len({record["contract_type"] for record in records}) == 3
    for record in records:
        pdf_path = repo_root / record["pdf_path"]
        txt_path = repo_root / record["txt_path"]
        assert txt_path.read_text(encoding="utf-8", errors="replace").strip()
        with fitz.open(pdf_path) as document:
            assert document.page_count > 0


def test_download_archive_deletes_corrupt_partial(tmp_path: Path, mocker) -> None:
    response = BytesIO(b"not the expected archive")
    response.status = 200
    mocker.patch("contract_rag.ingest.download.urllib.request.urlopen", return_value=response)
    destination = tmp_path / "CUAD_v1.zip"

    with pytest.raises(ValueError, match="checksum mismatch"):
        download_archive("https://example.test/CUAD_v1.zip", destination, expected_md5="0" * 32)

    assert not destination.with_suffix(".zip.part").exists()
