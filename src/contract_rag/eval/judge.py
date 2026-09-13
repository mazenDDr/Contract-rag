"""LLM-as-judge for answer quality, run locally through Ollama.

The answer is split into statements in code, each with the excerpt numbers it cites in the answer text. The
judge then checks every statement against ONLY the excerpts that statement cites, so it cannot find support
somewhere else, and marks whether the statement addresses the question. A second call grades correctness
against the lawyer-derived reference answer. Details the answer adds beyond the reference count against it
only when they contradict it: whether they are supported by the contract is the statement check's job.

Deterministic guards: a statement that cites nothing, cites an excerpt that doesn't exist, or states a number
its cited excerpts never mention is unsupported whatever the judge says. Citation validity and abstention
correctness never involve the LLM.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from contract_rag.generation.prompts import ABSTAIN_ANSWER
from contract_rag.schemas import Chunk, EvalQuestion, EvalScores, GenerationResult

# recorded with every judgement, so runs graded by different rubrics can't be mixed up
RUBRIC = "contradiction-only-v5"

CORRECTNESS_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}

_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_CITE_GROUP = re.compile(r"(?:\s*\[\d+(?:\s*,\s*\d+)*\])+")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "forty-five": 45, "sixty": 60, "ninety": 90, "hundred": 100,
}  # fmt: skip
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
- "correctness": "correct" if the answer states the key facts, reaches the same conclusion as the reference,
  and contradicts nothing in the reference or the highlighted clauses; "partial" if it states some key facts
  but misses, blurs or contradicts others; "incorrect" if it misses the key facts, contradicts them, or says
  the information is not available.
- Details the answer adds that the reference and the highlighted clauses do not mention are NOT errors: they
  may come from elsewhere in the contract, and they are checked separately. Only details that contradict the
  reference or the highlighted clauses count against the answer.
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


def excerpt_text(chunk: Chunk) -> str:
    """The text the generator was shown for this chunk (the sentence window, when there is one)."""
    return chunk.context_text or chunk.text


def _clean(text: str) -> str:
    text = text.strip(" ,;.\n")
    return re.sub(r"^(?:and|or|but|while|and also)\s+", "", text, flags=re.IGNORECASE).strip()


def split_statements(answer: str) -> list[Statement]:
    """Clauses of the answer, each with the excerpt numbers that directly follow it.

    "A within 30 days [3], B within a reasonable time [4]." becomes two statements, so each clause is
    checked against its own excerpt only. Sentences with no citation become their own (uncited) statements;
    a citation group left alone after a sentence break is attached to the previous statement."""
    statements: list[Statement] = []
    pos = 0
    for match in _CITE_GROUP.finditer(answer):
        cited = sorted({int(n) for group in _CITE.findall(match.group()) for n in group.split(",")})
        sentences = [_clean(s) for s in _SENTENCE_END.split(answer[pos : match.start()])]
        sentences = [s for s in sentences if s]
        pos = match.end()
        if not sentences:
            if statements:
                statements[-1].cited = sorted(set(statements[-1].cited) | set(cited))
            continue
        statements.extend(Statement(text=s) for s in sentences[:-1])
        statements.append(Statement(text=sentences[-1], cited=cited))
    tail = [_clean(s) for s in _SENTENCE_END.split(answer[pos:])]
    statements.extend(Statement(text=s) for s in tail if s)
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
                f"[{n}] {' > '.join(chunks[n - 1].section_path[-2:]) or 'no section'}\n"
                f"{excerpt_text(chunks[n - 1])}"
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


def _numbers(text: str, include_words: bool) -> set[str]:
    found = {m.group().replace(",", "").rstrip(".") for m in _NUMBER.finditer(text)}
    if include_words:
        found |= {
            str(v) for word, v in _NUMBER_WORDS.items() if re.search(rf"\b{word}\b", text, re.IGNORECASE)
        }
    return {n.removesuffix(".00") for n in found}


def unsupported_numbers(statement: str, excerpts: Sequence[str]) -> list[str]:
    """Digits in a statement (amounts, days, dates, section numbers) that its cited excerpts never mention,
    as digits or as number words. Such a number is an added specific, so the statement is unsupported.
    Number words in the statement are ignored ("one party" is not a quantity)."""
    return sorted(_numbers(statement, include_words=False) - _numbers(" ".join(excerpts), include_words=True))


def statement_check(st: Statement, verdict: Verdict, chunks: Sequence[Chunk]) -> tuple[bool, list[str]]:
    """(supported, unsupported numbers). Supported needs the judge's yes, real citations in the answer text,
    and every number in the statement present in the cited excerpts."""
    if not (verdict.supported and st.cited and all(1 <= n <= len(chunks) for n in st.cited)):
        return False, []
    missing = unsupported_numbers(st.text, [excerpt_text(chunks[n - 1]) for n in st.cited])
    return not missing, missing


def faithfulness_score(
    statements: Sequence[Statement], verdicts: Sequence[Verdict], chunks: Sequence[Chunk]
) -> float | None:
    """Share of statements that pass `statement_check`."""
    if not statements:
        return None
    checks = [statement_check(st, v, chunks)[0] for st, v in zip(statements, verdicts, strict=True)]
    return sum(checks) / len(statements)


def citation_validity(gen: GenerationResult) -> float | None:
    """Share of citation markers pointing at a provided excerpt (0 for an uncited, non-abstaining answer)."""
    total = len(gen.cited_chunk_ids) + gen.invalid_citations
    if total == 0:
        return None if gen.abstained else 0.0
    return len(gen.cited_chunk_ids) / total


_DECLINES = re.compile(
    r"\b(?:does|do|did) not (?:explicitly |expressly |specifically )?"
    r"(?:contain|specify|mention|include|state|provide|address|define|set out)"
    r"|\b(?:doesn't|don't|didn't) (?:contain|specify|mention|include|state|provide|address|define)"
    r"|\bnot (?:specified|stated|mentioned)\b|\bno (?:specific|explicit)\b",
    re.IGNORECASE,
)


def declines(gen: GenerationResult) -> bool:
    """The answer refuses: by its flag, or because its first sentence says the contract doesn't cover it.
    Many answers write "The agreement does not specify a warranty period" and leave the flag off."""
    text = gen.answer.strip()
    if gen.abstained or not text:
        return True
    return bool(_DECLINES.search(_SENTENCE_END.split(text, maxsplit=1)[0]))


def reuse_verdicts(prior_rationale: str | None, statements: Sequence[Statement]) -> list[Verdict] | None:
    """Statement verdicts from an earlier judging of the same answer, when it splits into the same statements.
    Lets a run be re-graded with a new correctness rubric without re-checking every statement."""
    if not prior_rationale:
        return None
    prior = json.loads(prior_rationale).get("statements")
    if not prior or [p["text"] for p in prior] != [s.text for s in statements]:
        return None
    return [
        Verdict(id=i, supported=p["supported"], on_topic=p.get("on_topic", True), reason=p.get("reason", ""))
        for i, p in enumerate(prior, start=1)
    ]


def score_answer(
    question: EvalQuestion,
    gen: GenerationResult,
    chunks: Sequence[Chunk],
    judge: OllamaJudge,
    prior_rationale: str | None = None,
) -> EvalScores:
    """Answer-side scores for one question; retrieval-side scores come from retrieval_metrics.

    The answer is judged on its text. Abstention accuracy uses `declines` (the flag or the wording); a refusal
    with no citations is not sent to the judge, but a "refusal" that reports cited content (for example, that
    the value is redacted) is graded like any other answer."""
    scores = EvalScores(qid=question.qid, config_id=gen.config_id, judge_model=judge.model)
    scores.abstention_correct = declines(gen) == (not question.answerable)
    scores.citation_validity = citation_validity(gen)
    detail: dict[str, Any] = {"rubric": RUBRIC, "statements": None, "grade": None}

    text = gen.answer.strip()
    if not text or text == ABSTAIN_ANSWER or (declines(gen) and not _CITE.search(text)):
        if question.answerable:
            scores.answer_correctness = 0.0  # refusing (or failing) on an answerable question is wrong
        scores.judge_rationale = json.dumps(detail)
        return scores

    statements = split_statements(gen.answer)
    verdicts = reuse_verdicts(prior_rationale, statements)
    if verdicts is None:
        verdicts = judge.check_statements(question.question, statements, chunks)
    if verdicts is not None and statements:
        scores.faithfulness = faithfulness_score(statements, verdicts, chunks)
        scores.answer_relevance = sum(v.on_topic for v in verdicts) / len(verdicts)
        detail["statements"] = [
            {
                **st.model_dump(),
                **v.model_dump(exclude={"id"}),
                "unsupported_numbers": statement_check(st, v, chunks)[1],
            }
            for st, v in zip(statements, verdicts, strict=True)
        ]
    if question.answerable:
        grade = judge.grade(question.question, question.reference_answer, question.evidence_spans, gen.answer)
        if grade is not None:
            scores.answer_correctness = CORRECTNESS_SCORE[grade.correctness]
            detail["grade"] = grade.model_dump()
    scores.judge_rationale = json.dumps(detail, ensure_ascii=False)
    return scores
