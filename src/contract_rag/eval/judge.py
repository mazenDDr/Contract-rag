"""LLM-as-judge for answer quality, run locally through Ollama.

Two focused calls per answer instead of one vague score:
- faithfulness: split the answer into atomic claims and check each against the excerpts it cites
- grading: correctness against the lawyer-derived reference answer, and relevance to the question

Citation validity and abstention correctness are computed in code, without the LLM, and a claim the judge
marks "supported" still counts as unsupported unless it cites at least one real excerpt.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from contract_rag.generation.prompts import format_context
from contract_rag.schemas import Chunk, EvalQuestion, EvalScores, GenerationResult

CORRECTNESS_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}

FAITHFULNESS_SYSTEM = """You verify answers about contracts against the numbered excerpts they cite.

Split the answer into atomic claims, one fact per claim. For each claim give:
- "claim": the claim in a few words
- "cited": the excerpt numbers the answer cites for that claim ([] if it cites none)
- "supported": true only if the cited excerpts state or directly imply the claim. A claim with no citation is
  not supported. Do not use outside knowledge. Judge facts, not style.
- "reason": one short sentence

Respond as JSON: {"claims": [{"claim": "...", "cited": [1], "supported": true, "reason": "..."}]}"""

GRADING_SYSTEM = """You grade an answer to a question about a contract.

You are given the question, a reference answer derived from a lawyer's annotation, the clause text the lawyer
highlighted, and the answer to grade.
- "correctness": "correct" if the answer states the reference's key facts without contradicting them;
  "partial" if it gets some key facts but misses or blurs others; "incorrect" if it contradicts the reference,
  misses the key facts, or claims the information is not available.
- "relevance": 1 to 5, how directly the answer addresses the question (5 = fully on point, nothing off-topic;
  1 = off-topic).
- "reason": one or two short sentences.

Respond as JSON: {"correctness": "correct", "relevance": 5, "reason": "..."}"""

FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "cited": {"type": "array", "items": {"type": "integer"}},
                    "supported": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["claim", "cited", "supported", "reason"],
            },
        }
    },
    "required": ["claims"],
}

GRADING_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "string", "enum": list(CORRECTNESS_SCORE)},
        "relevance": {"type": "integer", "minimum": 1, "maximum": 5},
        "reason": {"type": "string"},
    },
    "required": ["correctness", "relevance", "reason"],
}


class JudgeConfig(BaseModel):
    model: str = "gemma4:12b"
    host: str | None = None  # None -> $OLLAMA_HOST or http://localhost:11434
    temperature: float = 0.0
    num_ctx: int = 8192
    max_tokens: int = 1500
    seed: int = 0
    think: bool | None = False  # gemma4 can think; reasoning would eat the output budget. None = don't send


class Claim(BaseModel):
    claim: str
    cited: list[int] = []
    supported: bool
    reason: str = ""


class Grade(BaseModel):
    correctness: str
    relevance: int
    reason: str = ""


class OllamaJudge:
    def __init__(self, config: JudgeConfig | None = None, client: Any = None):
        self.config = config or JudgeConfig()
        self.model = self.config.model
        if client is None:
            import ollama

            client = ollama.Client(host=self.config.host)
        self.client = client
        self.latency_ms = 0.0
        self.calls = 0

    def _chat(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any] | None:
        cfg = self.config
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "options": {
                "temperature": cfg.temperature,
                "num_ctx": cfg.num_ctx,
                "num_predict": cfg.max_tokens,
                "seed": cfg.seed,
            },
        }
        if cfg.think is not None:
            kwargs["think"] = cfg.think
        start = time.perf_counter()
        response = self.client.chat(**kwargs)
        self.latency_ms += (time.perf_counter() - start) * 1000
        self.calls += 1
        try:
            data = json.loads(response.message.content or "")
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def claims(self, answer: str, chunks: Sequence[Chunk]) -> list[Claim] | None:
        user = f"Excerpts:\n\n{format_context(chunks)}\n\nAnswer to verify:\n{answer}"
        data = self._chat(FAITHFULNESS_SYSTEM, user, FAITHFULNESS_SCHEMA)
        if data is None:
            return None
        out = []
        for raw in data.get("claims", []):
            try:
                out.append(Claim.model_validate(raw))
            except ValidationError:
                continue
        return out

    def grade(self, question: str, reference: str, evidence: Sequence[str], answer: str) -> Grade | None:
        clauses = "\n".join(f"- {span}" for span in evidence)[:3000] or "(none)"
        user = (
            f"Question: {question}\n\nReference answer: {reference or '(none)'}\n\n"
            f"Lawyer-highlighted clause text:\n{clauses}\n\nAnswer to grade:\n{answer}"
        )
        data = self._chat(GRADING_SYSTEM, user, GRADING_SCHEMA)
        if data is None:
            return None
        try:
            grade = Grade.model_validate(data)
        except ValidationError:
            return None
        grade.correctness = grade.correctness.strip().lower()
        if grade.correctness not in CORRECTNESS_SCORE:
            return None
        grade.relevance = min(5, max(1, grade.relevance))
        return grade


def faithfulness_score(claims: Sequence[Claim], n_chunks: int) -> float | None:
    """Share of claims that the judge supports AND that cite at least one real excerpt."""
    if not claims:
        return None
    ok = sum(1 for c in claims if c.supported and c.cited and all(1 <= i <= n_chunks for i in c.cited))
    return ok / len(claims)


def citation_validity(gen: GenerationResult) -> float | None:
    """Share of citation markers pointing at a provided excerpt (0 for an uncited, non-abstaining answer)."""
    total = len(gen.cited_chunk_ids) + gen.invalid_citations
    if total == 0:
        return None if gen.abstained else 0.0
    return len(gen.cited_chunk_ids) / total


def score_answer(
    question: EvalQuestion, gen: GenerationResult, chunks: Sequence[Chunk], judge: OllamaJudge
) -> EvalScores:
    """Answer-side scores for one question; retrieval-side scores come from retrieval_metrics."""
    scores = EvalScores(qid=question.qid, config_id=gen.config_id, judge_model=judge.model)
    scores.abstention_correct = gen.abstained == (not question.answerable)
    scores.citation_validity = citation_validity(gen)
    detail: dict[str, Any] = {"claims": None, "grade": None}

    if gen.abstained or not gen.answer.strip():
        if question.answerable:
            scores.answer_correctness = 0.0  # refusing (or failing) on an answerable question is wrong
        scores.judge_rationale = json.dumps(detail)
        return scores

    claims = judge.claims(gen.answer, chunks)
    if claims is not None:
        scores.faithfulness = faithfulness_score(claims, len(chunks))
        detail["claims"] = [c.model_dump() for c in claims]
    if question.answerable:
        grade = judge.grade(question.question, question.reference_answer, question.evidence_spans, gen.answer)
        if grade is not None:
            scores.answer_correctness = CORRECTNESS_SCORE[grade.correctness]
            scores.answer_relevance = (grade.relevance - 1) / 4
            detail["grade"] = grade.model_dump()
    scores.judge_rationale = json.dumps(detail, ensure_ascii=False)
    return scores
