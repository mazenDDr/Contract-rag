# The contract-rag API with the best setup's data baked in: fixed chunks + BM25, about 13 MB.
# The answering model runs in an Ollama server at $OLLAMA_HOST. On a Mac running Ollama, that is
# http://host.docker.internal:11434.
#
#   PYTHONPATH=src .venv/bin/python scripts/stage_image.py
#   docker build -t contract-rag-api .
#   docker run -p 8000:7860 -e APP_API_KEY=... -e OLLAMA_HOST=http://host.docker.internal:11434 contract-rag-api
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860

# Hugging Face Spaces run containers as uid 1000
RUN useradd --create-home --uid 1000 app
WORKDIR /app

# only what the API imports: no torch or sentence-transformers, since this setup doesn't embed
COPY requirements-api.txt ./
RUN pip install -r requirements-api.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps .

COPY configs/api.yaml configs/retrieval.yaml ./configs/
COPY build/image/ ./
RUN mkdir -p runs/api && chown -R app:app /app/runs
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'], timeout=4)"
# refuses to start without APP_API_KEY
CMD ["sh", "-c", "exec python -m contract_rag.api.main --host 0.0.0.0 --port \"$PORT\""]
