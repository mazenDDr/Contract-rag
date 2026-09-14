import json
import logging

import pytest
from fastapi.testclient import TestClient

from contract_rag.api.app import create_app
from contract_rag.api.service import QAService, ServiceConfig, cited_numbers, display_title
from contract_rag.schemas import Chunk, Document, GenerationResult, RetrievalResult, RetrievedChunk

CLAUSES = [
    "This Agreement is governed by the laws of the State of Delaware.",
    "Either party may terminate this Agreement on ninety (90) days written notice.",
    "The licensee shall pay royalties every quarter.",
]


class FakePipeline:
    def __init__(self):
        self.calls = []

    def run(self, query, qid="", doc_id=None):
        self.calls.append((query, doc_id))
        final = [
            RetrievedChunk(chunk_id=f"d1::fixed::0000{i}", score=1.0, rank=i + 1, stage="bm25")
            for i in range(3)
        ]
        return RetrievalResult(qid=qid, config_id="x", final=final, latency_ms={"bm25": 0.4, "total": 0.5})


class FakeResources:
    def __init__(self, index_cfg, root):
        self.pipe = FakePipeline()
        self.closed = False

    def pipeline(self, config):
        return self.pipe

    def close(self):
        self.closed = True


class FakeGenerator:
    fail = False

    def __init__(self, cfg):
        pass

    def generate(self, question, chunks, qid, config_id):
        if FakeGenerator.fail:
            raise ConnectionError("ollama is not running")
        return GenerationResult(
            qid=qid,
            config_id=config_id,
            model="fake-4b",
            answer="Either party may terminate on ninety (90) days written notice [2], [7].",
            cited_chunk_ids=[chunks[1].chunk_id],
            invalid_citations=1,
            prompt_tokens=120,
            completion_tokens=20,
            latency_ms=35.0,
        )


@pytest.fixture
def service(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "chunks").mkdir()
    doc = Document(
        doc_id="d1",
        title="Master Services Agreement",
        source_path="x",
        num_pages=5,
        full_text=" ".join(CLAUSES),
    )
    (tmp_path / "data/documents.jsonl").write_text(doc.model_dump_json() + "\n")
    chunks, pos = [], 0
    for i, clause in enumerate(CLAUSES):
        chunks.append(
            Chunk(
                chunk_id=f"d1::fixed::0000{i}",
                doc_id="d1",
                strategy="fixed",
                text=clause,
                section_path=["4. Term and Termination"] if i == 1 else [],
                page_start=i + 1,
                page_end=i + 1,
                char_start=pos,
                char_end=pos + len(clause),
                token_count=0,
            )
        )
        pos += len(clause) + 1
    (tmp_path / "chunks/fixed.jsonl").write_text("".join(c.model_dump_json() + "\n" for c in chunks))
    (tmp_path / "retrieval.yaml").write_text("chunks_dir: chunks\nindex_dir: indexes\n")
    cfg = ServiceConfig(
        retrieval_config="retrieval.yaml", documents_path="data/documents.jsonl", max_question_chars=80
    )
    FakeGenerator.fail = False
    return QAService(cfg, tmp_path, resources_factory=FakeResources, generator_factory=FakeGenerator)


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(json.loads(record.getMessage()))


@pytest.fixture
def client_and_log(service):
    logger = logging.getLogger("test.api")
    handler = ListHandler()
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    return TestClient(create_app(service, api_key="s3cret", logger=logger)), handler


KEY = {"X-API-Key": "s3cret"}


def test_health_is_open_and_the_rest_needs_the_key(client_and_log):
    client, _ = client_and_log
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["contracts"] == 1 and health["auth"] is True
    assert client.get("/contracts").status_code == 401
    assert client.get("/contracts", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/contracts", headers=KEY).json() == [
        {
            "doc_id": "d1",
            "title": "Master Services Agreement",
            "source": "Master Services Agreement",
            "pages": 5,
        }
    ]


def test_ask_returns_the_answer_with_its_cited_excerpts(client_and_log, service):
    client, log = client_and_log
    response = client.post(
        "/ask",
        json={"doc_id": "d1", "question": "  Can either side end it early?  "},
        headers=KEY | {"X-Request-ID": "abc"},
    )
    assert response.status_code == 200 and response.headers["X-Request-ID"] == "abc"
    data = response.json()
    assert data["answer"].startswith("Either party") and data["abstained"] is False
    # [7] points at no excerpt (only 3 were given), so only [2] comes back, with the excerpt the model read
    assert [c["n"] for c in data["citations"]] == [2] and data["invalid_citations"] == 1
    cite = data["citations"][0]
    assert (
        cite["chunk_id"] == "d1::fixed::00001" and cite["page_start"] == 2 and "ninety (90)" in cite["text"]
    )
    assert cite["section"] == "4. Term and Termination"
    assert set(data["latency_ms"]) >= {"retrieval_bm25", "retrieval", "generation", "total"}
    assert data["prompt_tokens"] == 120 and data["cost_usd"] == 0.0
    assert service._resources.pipe.calls == [("Can either side end it early?", "d1")]  # scoped, trimmed
    line = log.lines[-1]
    assert line["path"] == "/ask" and line["status"] == 200 and line["request_id"] == "abc"
    assert line["doc_id"] == "d1" and line["citations"] == 1 and line["generation_ms"] == 35.0


def test_ask_errors_are_specific(client_and_log):
    client, log = client_and_log
    assert (
        client.post(
            "/ask", json={"doc_id": "nope", "question": "Which law governs?"}, headers=KEY
        ).status_code
        == 404
    )
    assert client.post("/ask", json={"doc_id": "d1", "question": "x" * 81}, headers=KEY).status_code == 422
    assert client.post("/ask", json={"doc_id": "d1"}, headers=KEY).status_code == 422
    FakeGenerator.fail = True
    failed = client.post("/ask", json={"doc_id": "d1", "question": "Which law governs?"}, headers=KEY)
    assert failed.status_code == 502 and "ConnectionError" in failed.json()["detail"]
    assert log.lines[-1]["status"] == 502


def test_auth_can_be_turned_off_for_local_development(service):
    client = TestClient(create_app(service, api_key=None))
    assert client.get("/contracts").status_code == 200 and client.get("/health").json()["auth"] is False


def test_display_title_makes_cuad_file_names_readable():
    assert display_title("ADUROBIOTECH,INC_06_02_2020-EX-10.7-CONSULTING AGREEMENT") == (
        "Consulting Agreement · Adurobiotech, Inc · 2020"
    )
    assert (
        display_title(
            "AlliedEsportsEntertainmentInc_20190815_8-K_EX-10.19_11788293_EX-10.19_Content License Agreement"
        )
        == "Content License Agreement · AlliedEsportsEntertainmentInc · 2019"
    )
    assert (
        display_title("ALCOSTORESINC_12_14_2005-EX-10.26-AGENCY AGREEMENT")
        == "Agency Agreement · Alcostoresinc · 2005"
    )
    assert display_title("Master Services Agreement") == "Master Services Agreement"


def test_cited_numbers_keeps_real_excerpts_in_order():
    assert cited_numbers("A [3]. B [1, 3]. C [9].", 8) == [3, 1]
    assert cited_numbers("No citations.", 8) == []
