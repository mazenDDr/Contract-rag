# The live demo on a Hugging Face Docker Space: one container runs the model server (Ollama with
# qwen3.5:4b baked in), the API and the Streamlit page, behind nginx on port 7860.
#   /        the Streamlit page (public)
#   /api/    the FastAPI service (needs the X-API-Key header; /api/health and /api/docs are open)
# Assembled and pushed by scripts/push_space.py; the build context is build/space/.
FROM ollama/ollama:latest AS ollama

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OLLAMA_MODELS=/opt/models \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_KEEP_ALIVE=-1 \
    GENERATOR_MODEL=qwen3.5:4b

RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# the Ollama server with its CPU libraries only (the image's CUDA libraries are several GB)
COPY --from=ollama /usr/bin/ollama /usr/bin/ollama
COPY --from=ollama /usr/lib/ollama/lib* /usr/lib/ollama/
COPY --from=ollama /usr/lib/ollama/llama-* /usr/lib/ollama/

# bake the model into the image so a cold start doesn't download it
RUN mkdir -p "$OLLAMA_MODELS" \
    && (ollama serve >/tmp/ollama-build.log 2>&1 &) \
    && for i in $(seq 1 30); do ollama list >/dev/null 2>&1 && break; sleep 1; done \
    && ollama pull "$GENERATOR_MODEL" \
    && chmod -R a+rX "$OLLAMA_MODELS"

# Hugging Face Spaces run containers as uid 1000
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY requirements-api.txt ./
RUN pip install -r requirements-api.txt "streamlit>=1.36"
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps .

COPY configs ./configs
COPY data ./data
COPY indexes ./indexes
COPY ui ./ui
COPY deploy/space/nginx.conf deploy/space/start.sh ./deploy/
RUN mkdir -p runs/api /tmp/nginx && chown -R app:app /app/runs /tmp/nginx && chmod +x deploy/start.sh
USER app

EXPOSE 7860
CMD ["./deploy/start.sh"]
