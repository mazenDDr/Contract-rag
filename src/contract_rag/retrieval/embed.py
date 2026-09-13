"""Sentence embeddings with an on-disk cache keyed by (model, sha256 of the exact text embedded)."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np


class Encoder(Protocol):
    name: str

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        """L2-normalized float32 vectors, one row per text."""
        ...


class SentenceTransformerEncoder:
    def __init__(self, model_name: str, device: str = "auto"):
        import torch
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """SQLite store of vectors. A bug fix elsewhere never forces re-embedding unchanged text."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors (model TEXT, key TEXT, vector BLOB, PRIMARY KEY (model, key))"
        )

    def get_many(self, model: str, keys: Iterable[str]) -> dict[str, np.ndarray]:
        keys = list(dict.fromkeys(keys))
        found: dict[str, np.ndarray] = {}
        for start in range(0, len(keys), 900):  # stay under SQLite's bound-parameter limit
            part = keys[start : start + 900]
            marks = ",".join("?" * len(part))
            rows = self.conn.execute(
                f"SELECT key, vector FROM vectors WHERE model = ? AND key IN ({marks})", [model, *part]
            )
            found.update((key, np.frombuffer(blob, dtype=np.float32)) for key, blob in rows)
        return found

    def put_many(self, model: str, items: dict[str, np.ndarray]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO vectors (model, key, vector) VALUES (?, ?, ?)",
            [(model, key, np.asarray(vec, dtype=np.float32).tobytes()) for key, vec in items.items()],
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def embed_texts(
    texts: Sequence[str],
    encoder: Encoder,
    cache: EmbeddingCache,
    batch_size: int,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Vectors for `texts` in order. Only texts missing from the cache are encoded, and each encoded batch is
    written to the cache at once, so an interrupted run resumes where it stopped."""
    keys = [text_key(t) for t in texts]
    by_key = dict(zip(keys, texts, strict=True))
    found = cache.get_many(encoder.name, by_key)
    missing = [k for k in by_key if k not in found]
    step = batch_size * 8
    for start in range(0, len(missing), step):
        part = missing[start : start + step]
        vectors = encoder.encode([by_key[k] for k in part], batch_size)
        new = dict(zip(part, vectors, strict=True))
        cache.put_many(encoder.name, new)
        found.update(new)
        if progress:
            progress(min(start + step, len(missing)), len(missing))
    return np.stack([found[k] for k in keys]) if keys else np.zeros((0, 0), dtype=np.float32)
