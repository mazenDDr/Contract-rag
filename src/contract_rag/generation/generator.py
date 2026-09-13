"""Answer generation with a local open-weight LLM served by Ollama."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from contract_rag.generation.prompts import ABSTAIN_ANSWER, ANSWER_SCHEMA, build_messages, parse_citations
from contract_rag.schemas import Chunk, GenerationResult


class GeneratorConfig(BaseModel):
    model: str = "qwen3.5:4b"
    host: str | None = None  # None -> $OLLAMA_HOST or http://localhost:11434
    temperature: float = 0.0
    num_ctx: int = 8192  # Ollama's default context is small; 8 chunks + prompt need room
    max_tokens: int = 400
    thinking_max_tokens: int = 2048  # reasoning shares the output budget; 400 ran out before the answer
    seed: int = 0
    think: bool | None = False  # hidden reasoning off for answers; None = don't send the flag


def parse_output(raw: str) -> tuple[str, bool]:
    """(answer, abstained) from the model's JSON; falls back to the raw text if it isn't valid JSON."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip(), False
    if not isinstance(data, dict):
        return raw.strip(), False
    abstained = bool(data.get("abstained", False))
    answer = str(data.get("answer", "")).strip()
    return (answer or ABSTAIN_ANSWER) if abstained else answer, abstained


class OllamaGenerator:
    def __init__(self, config: GeneratorConfig | None = None, client: Any = None):
        self.config = config or GeneratorConfig()
        self.model = self.config.model
        if client is None:
            import ollama

            client = ollama.Client(host=self.config.host)
        self.client = client

    def generate(self, question: str, chunks: Sequence[Chunk], qid: str, config_id: str) -> GenerationResult:
        cfg = self.config
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "messages": build_messages(question, chunks),
            "format": ANSWER_SCHEMA,
            "options": {
                "temperature": cfg.temperature,
                "num_ctx": cfg.num_ctx,
                "num_predict": cfg.thinking_max_tokens if cfg.think else cfg.max_tokens,
                "seed": cfg.seed,
            },
        }
        if cfg.think is not None:
            kwargs["think"] = cfg.think

        start = time.perf_counter()
        response = self.client.chat(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        raw = response.message.content or ""
        answer, abstained = parse_output(raw)
        cited, invalid = parse_citations(answer, chunks)
        return GenerationResult(
            qid=qid,
            config_id=config_id,
            model=cfg.model,
            answer=answer,
            cited_chunk_ids=cited,
            abstained=abstained,
            prompt_tokens=response.prompt_eval_count or 0,
            completion_tokens=response.eval_count or 0,
            cost_usd=0.0,
            latency_ms=latency_ms,
            invalid_citations=invalid,
            raw_output=raw,
            truncated=getattr(response, "done_reason", None) == "length",
        )
