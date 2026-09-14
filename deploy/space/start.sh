#!/bin/bash
# Start the model server, the API and the Streamlit page, then nginx in front of them on port 7860.
# If any of them exits, the container exits too, so the Space restarts it instead of half-running.
set -euo pipefail

if [ -z "${APP_API_KEY:-}" ]; then
    echo "APP_API_KEY is not set: add it as a secret in the Space settings" >&2
    exit 1
fi

ollama serve &
for i in $(seq 1 60); do
    ollama list >/dev/null 2>&1 && break
    sleep 1
done
# load the model now, so the first question doesn't pay for it
ollama run "$GENERATOR_MODEL" "" >/dev/null 2>&1 || true

python -m contract_rag.api.main --config configs/api-space.yaml --host 127.0.0.1 --port 8000 --root-path /api &

API_URL=http://127.0.0.1:8000 API_TIMEOUT=600 streamlit run ui/streamlit_app.py \
    --server.address 127.0.0.1 --server.port 8501 --server.headless true \
    --browser.gatherUsageStats false &

nginx -c /app/deploy/nginx.conf -g "daemon off;" &

wait -n
exit 1
