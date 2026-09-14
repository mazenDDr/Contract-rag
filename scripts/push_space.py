"""Assemble the Hugging Face Space in build/space/ and, with --push, upload it.

The Space builds deploy/space/Dockerfile: the model server with qwen3.5:4b, the API and the Streamlit page
behind nginx. Its build context holds the package, the Space configs, the UI, and the data the served setup
needs (parsed documents, fixed chunks and their BM25 index, about 13 MB), staged by scripts/stage_image.py.

Run:
  PYTHONPATH=src .venv/bin/python scripts/push_space.py                          # assemble only
  PYTHONPATH=src .venv/bin/python scripts/push_space.py --push mazenDDr/contract-rag
The API key is the Space secret APP_API_KEY. --push sets it from the APP_API_KEY environment variable
when that is set, and otherwise leaves the Space's current secret alone.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/space"
CODE = [
    "pyproject.toml",
    "requirements-api.txt",
    "src/contract_rag",
    "ui/streamlit_app.py",
    "configs/api-space.yaml",
    "configs/retrieval.yaml",
    "deploy/space",
]

README = """---
title: contract-rag
emoji: ⚖️
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
short_description: Ask a contract, get the clause, with citations.
---

# contract-rag

Ask a question about one of 100 real commercial contracts. The answer cites the excerpts it came from,
or says the contract doesn't cover the question. Everything runs in this container on the free CPU tier:
BM25 search inside the chosen contract, then `qwen3.5:4b` through Ollama. **An answer takes about two to
three minutes** because there is no GPU; on a laptop GPU it takes about ten seconds.

- Page: this Space's front page.
- API: `/api/docs`. `POST /api/ask` and `GET /api/contracts` need an `X-API-Key` header;
  `/api/health` is open.
- Code, experiments and the field guide: https://github.com/mazenDDr/Contract-rag

Contracts and annotations: [CUAD v1](https://www.atticusprojectai.org/cuad) by The Atticus Project,
licensed CC BY 4.0.
"""


def assemble() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/stage_image.py")], check=True)
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(ROOT / "build/image", OUT)
    for rel in CODE:
        src, dst = ROOT / rel, OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, dst)
    shutil.copy2(ROOT / "deploy/space/Dockerfile", OUT / "Dockerfile")
    (OUT / "README.md").write_text(README, encoding="utf-8")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"assembled {OUT.relative_to(ROOT)}/ ({size / 1e6:.1f} MB)")


def push(repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    if key := os.environ.get("APP_API_KEY"):
        api.add_space_secret(repo_id, "APP_API_KEY", key)
    api.upload_folder(
        folder_path=OUT, repo_id=repo_id, repo_type="space", commit_message="Update the Space build"
    )
    print(f"pushed: https://huggingface.co/spaces/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--push", metavar="OWNER/NAME", help="upload to this Space after assembling")
    args = parser.parse_args()
    assemble()
    if args.push:
        push(args.push)


if __name__ == "__main__":
    main()
