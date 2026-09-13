"""BM25 keyword retrieval over one chunking strategy, built with bm25s.

Every chunk is scored for each query, so restricting results to one contract (doc_id) is exact rather
than a post-filter over a truncated corpus-wide top-k.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import bm25s
import numpy as np

from contract_rag.retrieval.config import BM25Config
from contract_rag.schemas import Chunk, RetrievedChunk


def _stemmer(language: str | None):
    if not language:
        return None
    import Stemmer

    return Stemmer.Stemmer(language)


class BM25Index:
    def __init__(self, model: bm25s.BM25, chunk_ids: list[str], doc_ids: list[str], config: BM25Config):
        self.model = model
        self.chunk_ids = chunk_ids
        self.doc_ids = doc_ids
        self.config = config
        self._stem = _stemmer(config.stemmer)
        rows: dict[str, list[int]] = defaultdict(list)
        for row, doc_id in enumerate(doc_ids):
            rows[doc_id].append(row)
        self._rows_by_doc = {doc_id: np.array(r) for doc_id, r in rows.items()}
        self._all_rows = np.arange(len(chunk_ids))

    def tokenize(self, texts: Sequence[str]) -> list[list[str]]:
        return bm25s.tokenize(
            list(texts),
            stopwords=self.config.stopwords,
            stemmer=self._stem,
            return_ids=False,
            show_progress=False,
        )

    @classmethod
    def build(cls, chunks: Sequence[Chunk], config: BM25Config) -> BM25Index:
        index = cls(
            bm25s.BM25(k1=config.k1, b=config.b),
            [c.chunk_id for c in chunks],
            [c.doc_id for c in chunks],
            config,
        )
        index.model.index(index.tokenize([c.text for c in chunks]), show_progress=False)
        return index

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(str(directory / "bm25"))
        meta = {"chunk_ids": self.chunk_ids, "doc_ids": self.doc_ids, "config": self.config.model_dump()}
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> BM25Index:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        model = bm25s.BM25.load(str(directory / "bm25"))
        return cls(model, meta["chunk_ids"], meta["doc_ids"], BM25Config.model_validate(meta["config"]))

    def scores(self, query: str) -> np.ndarray:
        tokens = [t for t in self.tokenize([query])[0] if t in self.model.vocab_dict]
        if not tokens:
            return np.zeros(len(self.chunk_ids))
        return self.model.get_scores(tokens)

    def rows(self, doc_id: str | None) -> np.ndarray:
        return self._all_rows if doc_id is None else self._rows_by_doc.get(doc_id, np.array([], dtype=int))


class BM25Retriever:
    """Implements schemas.Retriever. Chunks sharing no term with the query are not returned."""

    def __init__(self, index: BM25Index):
        self.index = index

    def retrieve(self, query: str, k: int, doc_id: str | None = None) -> list[RetrievedChunk]:
        scores = self.index.scores(query)
        rows = self.index.rows(doc_id)
        rows = rows[scores[rows] > 0]
        top = rows[np.argsort(-scores[rows], kind="stable")[:k]]
        return [
            RetrievedChunk(chunk_id=self.index.chunk_ids[i], score=float(scores[i]), rank=rank, stage="bm25")
            for rank, i in enumerate(top, start=1)
        ]
