# API and UI

A small HTTP service around the project's best setup: fixed 512-token chunks with BM25 search inside the chosen contract, and `qwen3.5:4b` answering from the top 8 excerpts with a citation for each claim. It runs locally with Ollama; no paid APIs are used.

## Run it

You need Ollama running with `qwen3.5:4b`, plus the parsed data and indexes in `data/processed/` and `indexes/`.

```bash
export APP_API_KEY=choose-a-long-random-string
PYTHONPATH=src .venv/bin/python scripts/serve_api.py              # http://127.0.0.1:8000, settings in configs/api.yaml
PYTHONPATH=src .venv/bin/python scripts/serve_api.py --no-auth    # local development without a key
```

The web app needs the `ui` extra (`pip install -e ".[ui]"`):

```bash
API_URL=http://127.0.0.1:8000 APP_API_KEY=$APP_API_KEY streamlit run ui/streamlit_app.py
```

## Endpoints

| Method and path | Auth | What it does |
|---|---|---|
| `GET /health` | none | Status, model, retrieval setup, number of contracts, and whether auth is on |
| `GET /contracts` | `X-API-Key` | The contracts you can ask about: `doc_id`, `title`, `pages` |
| `POST /ask` | `X-API-Key` | Body `{"doc_id": "...", "question": "..."}` → an answer with citations |

Interactive docs are served at `/docs`.

A `POST /ask` response contains:

- **`answer`** and **`abstained`**: `abstained` is true when the model said the excerpts don't cover the question.
- **`citations`**: one entry per excerpt the answer cites, with `n` (the `[n]` in the answer), `chunk_id`, `page_start`, `page_end`, `section` and `text` (the exact excerpt the model read).
- **`invalid_citations`**: citation markers that point at no excerpt.
- **`truncated`**: true when the model hit its output limit.
- **`model`**, **`retrieval`** (the setup's id), **`prompt_tokens`**, **`completion_tokens`** and **`cost_usd`**. The cost is 0 locally; the field exists so a hosted model can report its cost.
- **`latency_ms`**: time per retrieval stage, `retrieval`, `generation` and `total`.

Errors:

| Status | When |
|---|---|
| 401 | Missing or wrong `X-API-Key` |
| 404 | Unknown `doc_id` |
| 422 | Invalid body, or a question over `max_question_chars` |
| 502 | The model server failed |

## Logging

Every request writes one JSON line to stdout, and to `log_path` if set (`runs/api/requests.jsonl` by default):

```json
{"ts": "2026-09-14T12:00:00+00:00", "request_id": "3f2a9c1b7d40", "method": "POST", "path": "/ask", "status": 200,
 "ms": 7912.4, "doc_id": "...", "abstained": false, "citations": 1, "prompt_tokens": 3120, "completion_tokens": 58,
 "retrieval_ms": 1.2, "generation_ms": 7890.1}
```

Pass an `X-Request-ID` header to trace a request end to end; the response echoes it back.

## Design notes

- **One question at a time.** The service serializes requests, because one laptop GPU runs the model.
- **Contract-scoped search is the whole point.** The ablation showed searching across all contracts ranks the right clause far lower (MRR 0.29 vs 0.73), so `/ask` always takes a `doc_id`.
- **The server refuses to start without `APP_API_KEY`** unless `--no-auth` is passed, so a deployment can't end up open by accident.
