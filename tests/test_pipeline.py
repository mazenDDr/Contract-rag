import re
import zlib

import numpy as np
import pytest

from contract_rag.retrieval.build import build_all
from contract_rag.retrieval.config import DenseModelConfig, IndexConfig, RerankerConfig
from contract_rag.retrieval.fusion import reciprocal_rank_fusion, weighted_fusion
from contract_rag.retrieval.pipeline import RetrievalPipeline, RetrievalResources, scoped_query
from contract_rag.retrieval.rerank import CrossEncoderReranker
from contract_rag.schemas import Chunk, RetrievalConfig, RetrievedChunk, Retriever


def ranked(ids, stage="bm25", scores=None):
    scores = scores or [1.0 / (i + 1) for i in range(len(ids))]
    return [
        RetrievedChunk(chunk_id=c, score=s, rank=i, stage=stage)
        for i, (c, s) in enumerate(zip(ids, scores, strict=True), 1)
    ]


def test_rrf_small_k_trusts_first_places_and_large_k_rewards_agreement():
    bm25 = ranked(["c1", "c3", "c2", "c4", "c6"])
    dense = ranked(["c4", "c3", "c0", "c2", "c7"], stage="dense")
    assert [r.chunk_id for r in reciprocal_rank_fusion([bm25, dense], k=1)][:2] == ["c4", "c3"]
    fused = reciprocal_rank_fusion([bm25, dense], k=60)
    assert [r.chunk_id for r in fused][:2] == ["c3", "c4"]  # #2 in both beats #1 in one
    assert [r.rank for r in fused] == list(range(1, 8)) and all(
        r.stage == "fusion" for r in fused
    )  # 7 unique
    assert fused[0].score == pytest.approx(2 / 62)


def test_weighted_fusion_rescales_and_mixes():
    bm25 = ranked(["a", "b"], scores=[12.0, 3.0])
    dense = ranked(["b", "c"], stage="dense", scores=[0.9, 0.2])
    assert [r.chunk_id for r in weighted_fusion(bm25, dense, alpha=0.0)][0] == "a"
    assert [r.chunk_id for r in weighted_fusion(bm25, dense, alpha=1.0)][0] == "b"
    mixed = {r.chunk_id: r.score for r in weighted_fusion(bm25, dense, alpha=0.5)}
    assert mixed == pytest.approx({"a": 0.5, "b": 0.5, "c": 0.0})
    with pytest.raises(ValueError):
        weighted_fusion(bm25, dense, alpha=1.5)


class FixedRetriever:
    def __init__(self, ids, stage):
        self.ids, self.stage, self.calls = ids, stage, []

    def retrieve(self, query, k, doc_id=None):
        self.calls.append(doc_id)
        return ranked(self.ids[:k], stage=self.stage)


class LengthScorer:
    """Fake cross-encoder: prefers longer texts."""

    def predict(self, sentences, **kwargs):
        return [len(text) for _, text in sentences]


TEXTS = {"c1": "x", "c2": "xxxx", "c3": "xx", "c4": "xxx", "c0": "xxxxx"}


def test_pipeline_records_every_stage_and_reranks_the_full_candidate_list():
    config = RetrievalConfig(
        chunking="section",
        sparse=True,
        dense_model="e5-base",
        fusion="rrf",
        reranker="fake",
        k_candidates=4,
        k_final=2,
    )
    bm25, dense = (
        FixedRetriever(["c1", "c3", "c2", "c4"], "bm25"),
        FixedRetriever(["c4", "c3", "c0"], "dense"),
    )
    reranker = CrossEncoderReranker("fake", scorer=LengthScorer())
    pipe = RetrievalPipeline(config, bm25=bm25, dense=dense, reranker=reranker, chunk_text=TEXTS)
    result = pipe.run("q", qid="q1", doc_id="d1")

    assert set(result.stages) == {"bm25", "dense", "fusion", "rerank"}
    assert len(result.stages["fusion"]) == 4 and len(result.stages["rerank"]) == 4  # full list kept
    # fused top 4 = c3, c4, c1, then c0 (c0 and c2 tie at 1/63; the tie goes to the lower chunk id);
    # the fake reranker prefers longer texts, so c0 (5 chars) then c4 (3 chars)
    assert [c.chunk_id for c in result.stages["fusion"]] == ["c3", "c4", "c1", "c0"]
    assert [c.chunk_id for c in result.final] == ["c0", "c4"] and [c.rank for c in result.final] == [1, 2]
    assert set(result.latency_ms) == {"bm25", "dense", "fusion", "rerank", "total"}
    assert result.config_id == config.config_id() and bm25.calls == ["d1"]
    assert isinstance(pipe, Retriever) and pipe.retrieve("q", k=1, doc_id="d1")[0].chunk_id == "c0"


def test_single_retriever_without_rerank_and_corpus_scope():
    config = RetrievalConfig(chunking="fixed", sparse=True, k_final=2, doc_filter=False)
    bm25 = FixedRetriever(["c1", "c3", "c2"], "bm25")
    result = RetrievalPipeline(config, bm25=bm25).run("q", doc_id="d1")
    assert set(result.stages) == {"bm25"} and [c.chunk_id for c in result.final] == ["c1", "c3"]
    assert bm25.calls == [None]  # doc_filter=False searches the whole corpus


def test_pipeline_rejects_inconsistent_configs():
    bm25, dense = FixedRetriever([], "bm25"), FixedRetriever([], "dense")
    with pytest.raises(ValueError):
        RetrievalPipeline(RetrievalConfig(chunking="fixed", sparse=False))
    with pytest.raises(ValueError):
        RetrievalPipeline(
            RetrievalConfig(chunking="fixed", dense_model="e5-base", fusion="none"), bm25=bm25, dense=dense
        )
    with pytest.raises(ValueError):
        RetrievalPipeline(RetrievalConfig(chunking="fixed", reranker="minilm"), bm25=bm25)


def test_scoped_query_drops_the_contract_name_only_when_scoped():
    name = "the Supply Agreement between Acme, Inc. and Beta LLC"
    question = f"Under {name}, which state's law governs the agreement?"
    assert (
        scoped_query(question, name, doc_filter=True)
        == "Under the agreement, which state's law governs the agreement?"
    )
    assert scoped_query(question, name, doc_filter=False) == question  # corpus-wide needs the name
    assert scoped_query(question, None, doc_filter=True) == question


class FakeEncoder:
    def __init__(self, name="fake", device="cpu"):
        self.name = name

    def encode(self, texts, batch_size):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                out[row, zlib.crc32(word.encode()) % 64] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def test_resources_build_a_hybrid_reranked_pipeline_end_to_end(tmp_path):
    texts = [
        "Either party may terminate on ninety days notice.",
        "Governed by the laws of Delaware.",
        "Royalties are paid every quarter.",
        "Termination for cause requires notice of breach.",
    ]
    chunks = [
        Chunk(
            chunk_id=f"d1::section::{i:05d}",
            doc_id="d1",
            strategy="section",
            text=t,
            page_start=1,
            page_end=1,
            char_start=0,
            char_end=len(t),
            token_count=0,
        )
        for i, t in enumerate(texts)
    ]
    (tmp_path / "chunks").mkdir()
    (tmp_path / "chunks" / "section.jsonl").write_text("".join(c.model_dump_json() + "\n" for c in chunks))
    config = IndexConfig(
        chunks_dir=tmp_path / "chunks",
        index_dir=tmp_path / "indexes",
        strategies=["section"],
        dense={"models": {"fake": DenseModelConfig(name="fake")}},
        rerankers={"len": RerankerConfig(name="len")},
    )
    build_all(config, tmp_path, encoder_factory=FakeEncoder, log=lambda _: None)
    resources = RetrievalResources(
        config,
        tmp_path,
        encoder_factory=FakeEncoder,
        reranker_factory=lambda name, **kw: CrossEncoderReranker(name, scorer=LengthScorer()),
    )
    try:
        pipe = resources.pipeline(
            RetrievalConfig(
                chunking="section", sparse=True, dense_model="fake", fusion="rrf", reranker="len", k_final=2
            )
        )
        result = pipe.run("notice to terminate", qid="q", doc_id="d1")
        assert {"bm25", "dense", "fusion", "rerank"} <= set(result.stages) and len(result.final) == 2
    finally:
        resources.close()
