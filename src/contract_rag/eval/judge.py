"""LLM-as-judge for answer quality, run locally through Ollama.

The answer is split into statements in code, each with the excerpt numbers it cites in the answer text. The
judge then checks every statement against ONLY the excerpts that statement cites, so it cannot find support
somewhere else, and marks whether the statement addresses the question. A second call grades correctness
against the lawyer-derived reference answer.

Deterministic guards: a statement that cites nothing, or cites an excerpt that doesn't exist, is unsupported
whatever the judge says. Citation validity and abstention correctness never involve the LLM.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from contract_rag.schemas import Chunk, EvalQuestion, EvalScores, GenerationResult

CORRECTNESS_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}

_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'“])")

STATEMENT_SYSTEM = """You check statements taken from an answer about a contract.

Each numbered statement comes with ONLY the excerpts that the statement cites. For each statement give:
- "supported": true only if every fact in the statement is stated in, or directly implied by, its own
  excerpts. Any added specific that those excerpts do not contain (a section number, a unit of time, a date,
  an amount, a party, a condition) makes the statement unsupported. A statement with no excerpts is
  unsupported.
- "on_topic": true if the statement helps answer the question; false if it is about something else.
- "reason": one short sentence.

Respond as JSON: {"verdicts": [{"id": 1, "supported": true, "on_topic": true, "reason": "..."}]}"""

GRADING_SYSTEM = """You grade whether an answer to a question about a contract is correct.

You are given the question, a reference answer derived from a lawyer's annotation, the clause text the lawyer
highlighted, and the answer. The key facts are the ones that directly answer the question; the highlighted
clauses may contain extra context, so do not require facts the question did not ask for.
- "correctness": "correct" if the answer states the key facts and contradicts nothing in the reference;
  "partial" if it states some key facts but misses, blurs or contradicts others; "incorrect" if it misses the
  key facts, contradicts them, or says the information is not available.
- "reason": one or two short sentences.

Respond as JSON: {"correctness": "correct", "reason": "..."}"""

STATEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "supported": {"type": "boolean"},
                    "on_topic": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "supported", "on_topic", "reason"],
            },
        }
    },
    "required": ["verdicts"],
}

GRADING_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "string", "enum": list(CORRECTNESS_SCORE)},
        "reason": {"type": "string"},
    },
    "required": ["correctness", "reason"],
}


class JudgeConfig(BaseModel):
    model: str = "gemma4:12b"
    host: str | None = None  # None -> $OLLAMA_HOST or http://localhost:11434
    temperature: float = 0.0
    num_ctx: int = 12288  # each statement carries its full cited excerpts
    max_tokens: int = 1500
    seed: int = 0
    think: bool | None = False  # gemma4 can think; reasoning would eat the output budget. None = don't send


class Statement(BaseModel):
    text: str
    cited: list[int] = []


class Verdict(BaseModel):
    id: int
    supported: bool
    on_topic: bool = True
    reason: str = ""


class Grade(BaseModel):
    correctness: str
    reason: str = ""


def split_statements(answer: str) -> list[Statement]:
    """Sentences of the answer with the excerpt numbers each one cites. A citation group that ends up
    alone after a sentence break ("... notice. [2]") is attached to the previous sentence."""
    statements: list[Statement] = []
    for sentence in _SENTENCE_END.split(answer.strip()):
        cited = sorted({int(n) for group in _CITE.findall(sentence) for n in group.split(",")})
        text = _CITE.sub("", sentence).strip(" .;")
        if text:
            statements.append(Statement(text=text, cited=cited))
        elif cited and statements:
            statements[-1].cited = sorted(set(statements[-1].cited) | set(cited))
    return statements


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

    def check_statements(
        self, question: str, statements: Sequence[Statement], chunks: Sequence[Chunk]
    ) -> list[Verdict] | None:
        """One verdict per statement, in order; None if the judge's output is unusable."""
        if not statements:
            return []
        parts = []
        for i, st in enumerate(statements, start=1):
            excerpts = [
                f"[{n}] {' > '.join(chunks[n - 1].section_path[-2:]) or 'no section'}\n{chunks[n - 1].text}"
                for n in st.cited
                if 1 <= n <= len(chunks)
            ]
            parts.append(
                f"Statement {i}: {st.text}\nExcerpts cited by statement {i}:\n"
                + ("\n\n".join(excerpts) or "(no excerpts cited)")
            )
        user = f"Question: {question}\n\n" + "\n\n---\n\n".join(parts)
        data = self._chat(STATEMENT_SYSTEM, user, STATEMENT_SCHEMA)
        if data is None:
            return None
        by_id: dict[int, Verdict] = {}
        for raw in data.get("verdicts", []):
            try:
                verdict = Verdict.model_validate(raw)
            except ValidationError:
                continue
            by_id.setdefault(verdict.id, verdict)
        if any(i not in by_id for i in range(1, len(statements) + 1)):
            return None
        return [by_id[i] for i in range(1, len(statements) + 1)]

    def grade(self, question: str, reference: str, evidence: Sequence[str], answer: str) -> Grade | None:
        clauses = "\n".join(f"- {span}" for span in evidence)[:3000] or "(none)"
        user = (
            f"Question: {question}\n\nReference answer: {reference or '(none)'}\n\n"
            f"Lawyer-highlighted clause text:\n{clauses}\n\nAnswer to grade:\n{_CITE.sub('', answer)}"
        )
        data = self._chat(GRADING_SYSTEM, user, GRADING_SCHEMA)
        if data is None:
            return None
        try:
            grade = Grade.model_validate(data)
        except ValidationError:
            return None
        grade.correctness = grade.correctness.strip().lower()
        return grade if grade.correctness in CORRECTNESS_SCORE else None


def faithfulness_score(
    statements: Sequence[Statement], verdicts: Sequence[Verdict], n_chunks: int
) -> float | None:
    """Share of statements the judge supports AND that cite at least one real excerpt in the answer text."""
    if not statements:
        return None
    ok = sum(
        1
        for st, v in zip(statements, verdicts, strict=True)
        if v.supported and st.cited and all(1 <= n <= n_chunks for n in st.cited)
    )
    return ok / len(statements)


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
    detail: dict[str, Any] = {"statements": None, "grade": None}

    if gen.abstained or not gen.answer.strip():
        if question.answerable:
            scores.answer_correctness = 0.0  # refusing (or failing) on an answerable question is wrong
        scores.judge_rationale = json.dumps(detail)
        return scores

    statements = split_statements(gen.answer)
    verdicts = judge.check_statements(question.question, statements, chunks)
    if verdicts is not None and statements:
        scores.faithfulness = faithfulness_score(statements, verdicts, len(chunks))
        scores.answer_relevance = sum(v.on_topic for v in verdicts) / len(verdicts)
        detail["statements"] = [
            {**st.model_dump(), **v.model_dump(exclude={"id"})}
            for st, v in zip(statements, verdicts, strict=True)
        ]
    if question.answerable:
        grade = judge.grade(question.question, question.reference_answer, question.evidence_spans, gen.answer)
        if grade is not None:
            scores.answer_correctness = CORRECTNESS_SCORE[grade.correctness]
            detail["grade"] = grade.model_dump()
    scores.judge_rationale = json.dumps(detail, ensure_ascii=False)
    return scores
