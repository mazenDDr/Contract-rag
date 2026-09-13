import json
from types import SimpleNamespace

from contract_rag.eval.judge import OllamaJudge
from contract_rag.eval.run_matrix import MatrixConfig, choose_configs, parse_config_key, run
from contract_rag.schemas import (
    Chunk,
    Document,
    EvalQuestion,
    GenerationResult,
    RetrievalResult,
    RetrievedChunk,
)

CLAUSES = [
    "This Agreement is governed by the laws of the State of Delaware.",
    "Either party may terminate this Agreement on ninety days notice.",
    "The licensee shall pay royalties every quarter.",
]
BM25 = "section__bm25__nodense__none__norerank__k8__doc"
HYBRID = "section__bm25__e5-base__rrf__minilm__k8__doc"
WEIGHTED = "fixed__bm25__bge-base__w0.4__norerank__k8__corpus"


def test_parse_config_key_inverts_the_ablation_ids():
    for key in (BM25, HYBRID, WEIGHTED, "fixed__bm25__nodense__none__norerank__k8__doc__rawq"):
        assert parse_config_key(key)[0].config_id() == key.removesuffix("__rawq")
    assert parse_config_key(WEIGHTED)[0].fusion_alpha == 0.4
    assert parse_config_key(WEIGHTED)[1] == "raw"  # corpus scope keeps the question as written
    assert parse_config_key(BM25)[1] == "scoped"
    assert parse_config_key(BM25 + "__rawq")[1] == "raw"
    summary = {"selected_on_dev": [HYBRID], "configs": {HYBRID: {}, BM25: {}}}
    assert choose_configs(summary, True, []) == [HYBRID, BM25]


class FakePipeline:
    def run(self, query, qid="", doc_id=None):
        final = [
            RetrievedChunk(chunk_id=f"d1::section::0000{i}", score=1.0, rank=i + 1, stage="bm25")
            for i in (0, 1)
        ]
        return RetrievalResult(qid=qid, config_id="x", final=final, latency_ms={"total": 1.0})


class FakeResources:
    """Every configuration retrieves the same chunks, so the second one must reuse the first answer."""

    def __init__(self, index_cfg, root):
        pass

    def pipeline(self, rc):
        return FakePipeline()

    def close(self):
        pass


class FakeGenerator:
    calls = 0

    def __init__(self, cfg):
        pass

    def generate(self, question, chunks, qid, config_id):
        FakeGenerator.calls += 1
        return GenerationResult(
            qid=qid,
            config_id=config_id,
            model="fake",
            answer="Delaware law governs [1].",
            cited_chunk_ids=[chunks[0].chunk_id],
        )


class FakeJudgeClient:
    def chat(self, **kwargs):
        user = kwargs["messages"][1]["content"]
        if "Statement 1" in user:
            n = user.count("Statement ")
            content = {
                "verdicts": [
                    {"id": i, "supported": True, "on_topic": True, "reason": ""} for i in range(1, n + 1)
                ]
            }
        else:
            content = {"correctness": "correct", "reason": ""}
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(content)))


def _write_corpus(root):
    (root / "data").mkdir()
    (root / "chunks").mkdir()
    doc = Document(doc_id="d1", title="t", source_path="x", num_pages=1, full_text="\n\n".join(CLAUSES))
    pos, chunks = 0, []
    for i, clause in enumerate(CLAUSES):
        chunks.append(
            Chunk(
                chunk_id=f"d1::section::0000{i}",
                doc_id="d1",
                strategy="section",
                text=clause,
                page_start=1,
                page_end=1,
                char_start=pos,
                char_end=pos + len(clause),
                token_count=0,
            )
        )
        pos += len(clause) + 2
    question = EvalQuestion(
        qid="q1",
        question="Under the Deal, which law governs?",
        doc_id="d1",
        qtype="cuad_derived",
        category="Governing Law",
        reference_answer="Delaware",
        evidence_spans=[CLAUSES[0]],
        split="test",
        contract_name="the Deal",
    )
    (root / "data/documents.jsonl").write_text(doc.model_dump_json() + "\n")
    (root / "data/questions.jsonl").write_text(question.model_dump_json() + "\n")
    (root / "chunks/section.jsonl").write_text("".join(c.model_dump_json() + "\n" for c in chunks))
    (root / "retrieval.yaml").write_text("chunks_dir: chunks\nindex_dir: indexes\n")
    configs = {HYBRID: {"description": "hybrid"}, BM25: {"description": "bm25"}}
    (root / "summary.json").write_text(json.dumps({"selected_on_dev": [HYBRID], "configs": configs}))


def test_run_reuses_identical_contexts_resumes_and_writes_a_report(tmp_path):
    _write_corpus(tmp_path)
    cfg = MatrixConfig(
        ablation_summary="summary.json",
        questions_path="data/questions.jsonl",
        documents_path="data/documents.jsonl",
        retrieval_config="retrieval.yaml",
        report_path="docs/answer_quality.md",
        n_boot=20,
    )

    def go():
        return run(
            cfg,
            tmp_path,
            tmp_path / "runs/m1",
            resources_factory=FakeResources,
            generator_factory=FakeGenerator,
            judge_factory=lambda c: OllamaJudge(c, client=FakeJudgeClient()),
            log=lambda _: None,
        )

    FakeGenerator.calls = 0
    report = go()
    assert FakeGenerator.calls == 1 and report["meta"]["generations_reused"] == 1
    assert set(report["configs"]) == {HYBRID, BM25}
    hybrid = report["configs"][HYBRID]
    assert hybrid["answer_correctness"]["mean"] == 1.0 and hybrid["faithfulness"]["mean"] == 1.0
    assert hybrid["context_recall"]["mean"] == 1.0
    assert (tmp_path / "docs/answer_quality.md").read_text().startswith("# Answer quality")
    scores = tmp_path / "runs/m1/scores.jsonl"
    assert len(scores.read_text().splitlines()) == 2

    go()  # resumed: nothing is generated or scored twice
    assert FakeGenerator.calls == 1
    assert len(scores.read_text().splitlines()) == 2
