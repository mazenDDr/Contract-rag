"""Stage the data the API image needs into build/image/, following symlinks.

The image serves the setup in configs/api.yaml (fixed chunks + BM25). It needs the parsed documents, that
setup's chunks and its BM25 index, about 13 MB, but not the vector index or the embedding cache. Docker
doesn't follow symlinks that point outside the build context, so the real files are copied here first.

Run: PYTHONPATH=src .venv/bin/python scripts/stage_image.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from contract_rag.api.service import load_config
from contract_rag.retrieval.config import IndexConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/image"


def main() -> None:
    cfg = load_config(ROOT / "configs/api.yaml")
    if cfg.retrieval.dense_model or cfg.retrieval.reranker:
        raise SystemExit(
            "the image only bakes in BM25; a dense model or reranker would need its index and model too"
        )
    index_cfg = IndexConfig.model_validate(
        yaml.safe_load((ROOT / cfg.retrieval_config).read_text(encoding="utf-8"))
    )
    strategy = cfg.retrieval.chunking
    wanted = [
        cfg.documents_path,
        index_cfg.chunks_dir / f"{strategy}.jsonl",
        index_cfg.index_dir / "bm25" / strategy,
    ]
    if OUT.exists():
        shutil.rmtree(OUT)
    total = 0
    for rel in wanted:
        src, dst = (ROOT / rel).resolve(), OUT / rel
        if not src.exists():
            raise SystemExit(f"missing {rel}: build the data and indexes first (see the README)")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        size = sum(p.stat().st_size for p in ([dst] if dst.is_file() else dst.rglob("*")) if p.is_file())
        total += size
        print(f"{rel}  {size / 1e6:.1f} MB")
    print(f"staged {total / 1e6:.1f} MB in {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
