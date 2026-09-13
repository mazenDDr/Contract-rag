import json
import re
import zlib

import numpy as np
import yaml

from contract_rag.eval.ablation import AblationConfig, CachingScorer, build_grid, config_key, describe, run
from contract_rag.retrieval.build import build_all
from contract_rag.retrieval.config import DenseModelConfig, IndexConfig, RerankerConfig
from contract_rag.schemas import Chunk, Document, EvalQuestion, RetrievalConfig

CLAUSES = [
    "This Agreement is governed by the laws of the State of Delaware.",
    "Either party may terminate this Agreement for convenience on ninety days notice.",
    "The licensee shall pay royalties to the licensor every quarter.",
    "Each party shall maintain general liability insurance of one million dollars.",
]


class FakeEncoder:
    def __init__(self, name="fake", device="cpu"):
        self.name = name

    def encode(self, texts, batch_size):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                out[row, zlib.crc32(word.encode()) % 64] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class OverlapScorer:
    """Fake cross-encoder: number of shared lowercase words; counts pairs it is asked to score."""

    def __init__(self):
        self.pairs = 0

    def predict(self, sentences, **kwargs):
        self.pairs += len(sentences)
        return [len(set(q.lower().split()) & set(t.lower().split())) for q, t in sentences]


def _write_corpus(root):
    docs, chunks, questions = [], [], []
    for d in range(4):
        doc_id = f"doc{d}"
        text = "\n\n".join(CLAUSES)
        docs.append(Document(doc_id=doc_id, title=doc_id, source_path="x", num_pages=1, full_text=text))
        pos = 0
        for i, clause in enumerate(CLAUSES):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::section::{i:05d}",
                    doc_id=doc_id,
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
        name = f"the Supply Agreement between Acme {d} and Beta"
        questions.append(
            EvalQuestion(
                qid=f"q{d:04d}",
                question=f"Under {name}, which state's laws govern it?",
                doc_id=doc_id,
                qtype="cuad_derived",
                category="Governing Law",
                reference_answer="Delaware",
                evidence_spans=[CLAUSES[0]],
                split="dev" if d < 2 else "test",
                contract_name=name,
            )
        )
    (root / "data").mkdir()
    (root / "data/documents.jsonl").write_text("".join(x.model_dump_json() + "\n" for x in docs))
    (root / "data/questions.jsonl").write_text("".join(x.model_dump_json() + "\n" for x in questions))
    (root / "chunks").mkdir()
    (root / "chunks/section.jsonl").write_text("".join(c.model_dump_json() + "\n" for c in chunks))


def test_grid_enumerates_every_combination_plus_raw_query_controls():
    cfg = AblationConfig(
        chunkings=["section"], dense_models=["m1", "m2"], rerankers=["rr"], scopes=["doc", "corpus"]
    )
    items = build_grid(cfg, {("section", "m1"): 0.4, ("section", "m2"): 0.6})
    # (1 bm25 + 2 dense + 2x2 hybrid) x (none + 1 reranker) x 2 scopes + 3 raw-query controls
    assert len(items) == 7 * 2 * 2 + 3
    assert len({config_key(rc, mode) for rc, mode in items}) == len(items)
    assert all(mode == "raw" for rc, mode in items if not rc.doc_filter)
    weighted = [rc for rc, _ in items if rc.fusion == "weighted" and rc.dense_model == "m2"]
    assert {rc.fusion_alpha for rc in weighted} == {0.6}
    rc = RetrievalConfig(chunking="section", sparse=True, dense_model="m1", fusion="rrf", reranker="rr")
    assert describe(rc) == "section · BM25+m1 (RRF) · rr · contract"


def test_caching_scorer_scores_each_pair_once():
    inner = OverlapScorer()
    scorer = CachingScorer(inner, "rerank:x", {"rerank:x": []} | {})
    pairs = [("q a", "a b"), ("q a", "c d"), ("q a", "a b")]
    assert scorer.predict(pairs) == [1, 0, 1] and inner.pairs == 2
    scorer.predict(pairs)
    assert inner.pairs == 2 and scorer.pairs_scored == 2


def test_run_end_to_end_writes_scores_report_and_resumes(tmp_path):
    _write_corpus(tmp_path)
    index_cfg = IndexConfig(
        chunks_dir="chunks",
        index_dir="indexes",
        strategies=["section"],
        dense={"models": {"fake": DenseModelConfig(name="fake")}},
        rerankers={"overlap": RerankerConfig(name="overlap")},
    )
    build_all(index_cfg, tmp_path, encoder_factory=FakeEncoder, log=lambda _: None)
    (tmp_path / "retrieval.yaml").write_text(yaml.safe_dump(json.loads(index_cfg.model_dump_json())))
    cfg = AblationConfig(
        questions_path="data/questions.jsonl",
        documents_path="data/documents.jsonl",
        retrieval_config="retrieval.yaml",
        runs_dir="runs",
        report_path="docs/ablations.md",
        chunkings=["section"],
        dense_models=["fake"],
        rerankers=["overlap"],
        alpha_grid=[0.3, 0.7],
        scopes=["doc"],
        n_boot=50,
        select_top=3,
    )
    scorer = OverlapScorer()
    kwargs = {
        "encoder_factory": FakeEncoder,
        "scorer_factory": lambda name, device, max_length: scorer,
        "log": lambda _: None,
    }
    summary = run(cfg, tmp_path, run_dir=tmp_path / "runs" / "r1", **kwargs)

    n_configs = (1 + 1 + 2) * 2 + 3
    assert summary["meta"]["configs"] == n_configs and len(summary["configs"]) == n_configs
    lines = (tmp_path / "runs/r1/scores.jsonl").read_text().splitlines()
    assert len(lines) == n_configs * 4
    assert len(summary["selected_on_dev"]) == 3
    assert (tmp_path / "docs/ablations.md").read_text().startswith("# Retrieval ablations")
    assert any(c["comparison"].startswith("section: drop contract name") for c in summary["comparisons"])
    assert summary["meta"]["reranker_pairs_scored"]["overlap"] == scorer.pairs  # no pair scored twice

    before = scorer.pairs
    run(cfg, tmp_path, run_dir=tmp_path / "runs" / "r1", **kwargs)  # resume: nothing left to do
    assert len((tmp_path / "runs/r1/scores.jsonl").read_text().splitlines()) == len(lines)
    assert scorer.pairs == before  # finished configs are skipped; alpha tuning uses no reranker
