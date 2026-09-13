import json
import re
import zlib

import numpy as np
from qdrant_client import QdrantClient

from contract_rag.retrieval.bm25 import BM25Index, BM25Retriever
from contract_rag.retrieval.build import build_all, load_bm25_retriever
from contract_rag.retrieval.config import BM25Config, DenseModelConfig, IndexConfig
from contract_rag.retrieval.dense import DenseRetriever, build_collection
from contract_rag.retrieval.embed import EmbeddingCache, embed_texts
from contract_rag.schemas import Chunk, Retriever

TEXTS = {
    "d1": [
        "Either party may terminate this Agreement for convenience on ninety days notice.",
        "This Agreement is governed by the laws of the State of Delaware.",
    ],
    "d2": [
        "Termination for cause requires written notice of the breach.",
        "The licensee shall pay royalties to the licensor every quarter.",
    ],
}
CHUNKS = [
    Chunk(
        chunk_id=f"{doc}::section::{i:05d}",
        doc_id=doc,
        strategy="section",
        text=text,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(text),
        token_count=0,
    )
    for doc, texts in TEXTS.items()
    for i, text in enumerate(texts)
]


class FakeEncoder:
    """Deterministic bag-of-words vectors (crc32 hashing), so tests never download a model."""

    def __init__(self, name: str = "fake", device: str = "cpu"):
        self.name = name
        self.calls = 0

    def encode(self, texts, batch_size):
        self.calls += 1
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                out[row, zlib.crc32(word.encode()) % 64] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def test_bm25_stems_scores_and_filters_by_contract(tmp_path):
    retriever = BM25Retriever(BM25Index.build(CHUNKS, BM25Config()))
    hits = retriever.retrieve("termination notice", k=5)
    assert {h.chunk_id for h in hits} >= {
        "d1::section::00000",
        "d2::section::00000",
    }  # terminate ~ termination
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1)) and all(h.stage == "bm25" for h in hits)
    assert all(
        h.chunk_id.startswith("d1::") for h in retriever.retrieve("termination notice", k=5, doc_id="d1")
    )
    assert retriever.retrieve("zebra giraffe", k=5) == []  # no shared term, nothing returned
    assert retriever.retrieve("notice", k=5, doc_id="unknown") == []

    retriever.index.save(tmp_path / "bm25")
    reloaded = BM25Retriever(BM25Index.load(tmp_path / "bm25"))
    assert reloaded.retrieve("termination notice", k=5) == hits


def test_embedding_cache_never_reencodes_known_text(tmp_path):
    cache, encoder = EmbeddingCache(tmp_path / "e.sqlite"), FakeEncoder()
    texts = [c.text for c in CHUNKS] + [CHUNKS[0].text]  # a duplicate is embedded once
    first = embed_texts(texts, encoder, cache, batch_size=2)
    calls = encoder.calls
    second = embed_texts(texts, encoder, cache, batch_size=2)
    assert encoder.calls == calls and np.allclose(first, second) and first.shape == (5, 64)
    assert np.allclose(first[0], first[4])


def test_dense_retrieval_with_contract_filter(tmp_path):
    client, encoder = QdrantClient(path=str(tmp_path / "qdrant")), FakeEncoder()
    vectors = embed_texts([c.text for c in CHUNKS], encoder, EmbeddingCache(tmp_path / "e.sqlite"), 4)
    build_collection(client, "section__fake", CHUNKS, vectors)
    retriever = DenseRetriever(client, "section__fake", encoder)
    top = retriever.retrieve("governed by the laws of Delaware", k=2)
    assert top[0].chunk_id == "d1::section::00001" and top[0].stage == "dense"
    assert all(
        h.chunk_id.startswith("d2::") for h in retriever.retrieve("laws of Delaware", k=2, doc_id="d2")
    )
    assert isinstance(retriever, Retriever) and isinstance(
        BM25Retriever(BM25Index.build(CHUNKS, BM25Config())), Retriever
    )
    client.close()


def test_build_all_writes_indexes_and_a_report(tmp_path):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    (chunks_dir / "section.jsonl").write_text("".join(c.model_dump_json() + "\n" for c in CHUNKS))
    config = IndexConfig(
        chunks_dir=chunks_dir,
        index_dir=tmp_path / "indexes",
        strategies=["section"],
        dense={"models": {"fake": DenseModelConfig(name="fake", passage_prefix="passage: ")}},
        smoke_queries=["governed by the laws of Delaware"],  # the fake encoder only matches exact words
    )
    report = build_all(config, tmp_path, encoder_factory=FakeEncoder, log=lambda _: None)
    assert report["bm25"]["section"]["chunks"] == 4
    assert report["dense"]["section__fake"]["dim"] == 64
    assert report["dense"]["section__fake"]["smoke"][0]["top_chunk"] == "d1::section::00001"
    assert (
        json.loads((tmp_path / "indexes" / "index_report.json").read_text())["bm25"]["section"]["chunks"] == 4
    )
    assert load_bm25_retriever(tmp_path / "indexes", "section").retrieve("royalties", k=1)[0].chunk_id == (
        "d2::section::00001"
    )
