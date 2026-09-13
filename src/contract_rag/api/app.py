"""HTTP API for contract-rag: list the contracts, ask a question about one of them.

- `GET /health` is open; `GET /contracts` and `POST /ask` need the `X-API-Key` header when a key is set.
- Every response carries an `X-Request-ID` (taken from the request if given), and every request writes
  one JSON log line with its status and latency; `/ask` adds the contract, whether the model declined,
  the number of citations, token counts and the retrieval/generation split.
"""

from __future__ import annotations

import hmac
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from contract_rag.api.service import AskResponse, Contract, QAService, UnknownContract

LOGGER_NAME = "contract_rag.api"


class AskRequest(BaseModel):
    doc_id: str = Field(min_length=1)
    question: str = Field(min_length=3, max_length=2000)


def create_app(service: QAService, api_key: str | None, logger: logging.Logger | None = None) -> FastAPI:
    """`api_key=None` turns authentication off (local development only; the server entry point refuses to
    start that way unless asked)."""
    log = logger or logging.getLogger(LOGGER_NAME)
    app = FastAPI(title="contract-rag", summary="Ask a contract, get the clause.", version="1.0")

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if api_key is None:
            return
        if not x_api_key or not hmac.compare_digest(x_api_key.encode(), api_key.encode()):
            raise HTTPException(status_code=401, detail="missing or wrong X-API-Key header")

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        request.state.log_extra = {}
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        record: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "ms": round((time.perf_counter() - started) * 1000, 1),
            **request.state.log_extra,
        }
        log.info(json.dumps(record))
        return response

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": service.model,
            "retrieval": service.retrieval_id,
            "contracts": len(service.contracts()),
            "auth": api_key is not None,
        }

    @app.get("/contracts", dependencies=[Depends(require_key)])
    def contracts() -> list[Contract]:
        return service.contracts()

    @app.post("/ask", dependencies=[Depends(require_key)])
    def ask(body: AskRequest, request: Request) -> AskResponse:
        question = body.question.strip()
        if len(question) > service.config.max_question_chars:
            raise HTTPException(
                status_code=422,
                detail=f"question is longer than {service.config.max_question_chars} characters",
            )
        try:
            answer = service.ask(body.doc_id, question, request.state.request_id)
        except UnknownContract:
            raise HTTPException(status_code=404, detail=f"unknown contract: {body.doc_id}") from None
        except Exception as exc:  # the model server is down or failed: say so instead of a bare 500
            failure = {
                "event": "answer_failed",
                "request_id": request.state.request_id,
                "error": type(exc).__name__,
            }
            log.error(json.dumps(failure | {"detail": str(exc)[:300]}))
            raise HTTPException(
                status_code=502, detail=f"the answering model failed: {type(exc).__name__}"
            ) from None
        request.state.log_extra = {
            "doc_id": answer.doc_id,
            "abstained": answer.abstained,
            "citations": len(answer.citations),
            "prompt_tokens": answer.prompt_tokens,
            "completion_tokens": answer.completion_tokens,
            "retrieval_ms": answer.latency_ms.get("retrieval"),
            "generation_ms": answer.latency_ms.get("generation"),
        }
        return answer

    return app
