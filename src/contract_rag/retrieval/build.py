"""Build BM25 and dense (Qdrant) indexes for every chunking strategy and embedding model."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from qdrant_client import QdrantClient

from contract_rag.retrieval.bm25 import BM25Index, BM25Retriever
from contract_rag.retrieval.config import IndexConfig, collection_name, load_chunks
from contract_rag.retrieval.dense import DenseRetriever, build_collection
from contract_rag.retrieval.embed import (
    EmbeddingCache,
    Encoder,
    SentenceTransformerEncoder,
    embed_texts,
    text_key,
)
from contract_rag.schemas import Chunk, Retriever

EncoderFactory = Callable[[str, str], Encoder]


def _dir_size_mb(path: Path) -> float:
    return (
        round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 2)
        if path.exists()
        else 0.0
    )


def _smoke(retriever: Retriever, queries: list[str], chunk_text: dict[str, str]) -> list[dict[str, Any]]:
    out = []
    for query in queries:
        hits = retriever.retrieve(query, k=1)
        out.append(
            {
                "query": query,
                "top_chunk": hits[0].chunk_id if hits else None,
                "score": round(hits[0].score, 4) if hits else None,
                "text": chunk_text[hits[0].chunk_id][:120] if hits else None,
            }
        )
    return out


def load_bm25_retriever(index_dir: Path, strategy: str) -> BM25Retriever:
    return BM25Retriever(BM25Index.load(index_dir / "bm25" / strategy))


def build_all(
    config: IndexConfig,
    repo_root: Path,
    strategies: list[str] | None = None,
    model_keys: list[str] | None = None,
    encoder_factory: EncoderFactory = SentenceTransformerEncoder,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    strategies = strategies or list(config.strategies)
    model_keys = model_keys or list(config.dense.models)
    index_dir = repo_root / config.index_dir
    chunks: dict[str, list[Chunk]] = {
        s: load_chunks(repo_root / config.chunks_dir / f"{s}.jsonl") for s in strategies
    }
    chunk_text = {c.chunk_id: c.text for cs in chunks.values() for c in cs}
    report: dict[str, Any] = {"bm25": {}, "dense": {}}

    for strategy, cs in chunks.items():
        start = time.perf_counter()
        index = BM25Index.build(cs, config.bm25)
        index.save(index_dir / "bm25" / strategy)
        report["bm25"][strategy] = {
            "chunks": len(cs),
            "vocabulary": len(index.model.vocab_dict),
            "seconds": round(time.perf_counter() - start, 2),
            "size_mb": _dir_size_mb(index_dir / "bm25" / strategy),
            "smoke": _smoke(BM25Retriever(index), config.smoke_queries, chunk_text),
        }
        log(f"bm25/{strategy}: {len(cs)} chunks in {report['bm25'][strategy]['seconds']}s")

    cache = EmbeddingCache(index_dir / "embeddings.sqlite")
    client = QdrantClient(path=str(index_dir / "qdrant"))
    try:
        for key in model_keys:
            model_cfg = config.dense.models[key]
            encoder = encoder_factory(model_cfg.name, config.dense.device)
            texts = {s: [model_cfg.passage_prefix + c.text for c in cs] for s, cs in chunks.items()}
            all_keys = {text_key(t) for ts in texts.values() for t in ts}
            missing = sorted(all_keys - set(cache.get_many(encoder.name, all_keys)))
            if missing:  # time one batch to estimate the wall clock before the long part
                by_key = {text_key(t): t for ts in texts.values() for t in ts}
                sample = [by_key[k] for k in missing[: config.dense.batch_size]]
                start = time.perf_counter()
                cache.put_many(
                    encoder.name,
                    dict(
                        zip(
                            missing[: len(sample)],
                            encoder.encode(sample, config.dense.batch_size),
                            strict=True,
                        )
                    ),
                )
                per_text = (time.perf_counter() - start) / len(sample)
                log(f"{key}: {len(missing)} texts to embed, estimated {per_text * len(missing) / 60:.1f} min")
            for strategy, cs in chunks.items():
                start = time.perf_counter()
                vectors = embed_texts(texts[strategy], encoder, cache, config.dense.batch_size)
                name = collection_name(strategy, key)
                build_collection(client, name, cs, vectors)
                retriever = DenseRetriever(client, name, encoder, model_cfg.query_prefix)
                report["dense"][name] = {
                    "chunks": len(cs),
                    "dim": int(vectors.shape[1]),
                    "seconds": round(time.perf_counter() - start, 2),
                    "smoke": _smoke(retriever, config.smoke_queries, chunk_text),
                }
                log(f"dense/{name}: {len(cs)} chunks in {report['dense'][name]['seconds']}s")
    finally:
        client.close()
        cache.close()
    report["sizes_mb"] = {
        "bm25": _dir_size_mb(index_dir / "bm25"),
        "qdrant": _dir_size_mb(index_dir / "qdrant"),
        "embedding_cache": _dir_size_mb(index_dir / "embeddings.sqlite"),
    }
    (index_dir / "index_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/retrieval.yaml"))
    parser.add_argument("--strategy", action="append", choices=("fixed", "sentence_window", "section"))
    parser.add_argument("--model", action="append", help="embedding model key from the config")
    args = parser.parse_args()
    config = IndexConfig.model_validate(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    report = build_all(config, Path.cwd().resolve(), args.strategy, args.model)
    print(json.dumps(report["sizes_mb"], indent=2))


if __name__ == "__main__":
    main()
