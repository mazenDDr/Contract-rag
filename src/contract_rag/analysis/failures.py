"""Failure triage: where along the pipeline did each answer go wrong?

Every (configuration, question) row of an answer-quality run is traced through the pipeline in order, and
the first check that fails names the row's category.

Answerable question, answer not fully correct:
  1. no chunk of the contract contains any evidence      -> lost_in_chunking
  2. no retriever ranked a chunk holding any evidence     -> retrieval_miss
  3. evidence ranked, but none reached the generator:
       it was inside the final cut before the reranker   -> reranker_demotion
       otherwise                                         -> ranking_miss
  4. only some evidence pieces reached the generator      -> partial_evidence
  5. all evidence reached it, and it refused              -> refused_with_evidence
  6. all evidence reached it, and the answer is wrong or incomplete:
       some statement is unsupported by its citations    -> unsupported_claims
       every checked statement is supported              -> misread_evidence
Unanswerable question answered instead of refused         -> answered_unanswerable

"Reached the generator" is judged on the text the generator was shown: the sentence window when there is
one, else the chunk. It uses the same overlap rule as the relevance labels, so on sentence-window chunks
it can be higher than recall@k, which only looks at the matched sentence.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import Stemmer
import yaml
from pydantic import BaseModel

from contract_rag.eval.labels import MIN_OVERLAP_FRAC, QuestionLabels, SpanAlignment, build_labels, overlap
from contract_rag.retrieval.config import IndexConfig, load_chunks
from contract_rag.schemas import (
    Chunk,
    Document,
    EvalQuestion,
    EvalScores,
    GenerationResult,
    RetrievalResult,
    RetrievedChunk,
)

CATEGORIES = (
    "lost_in_chunking",
    "retrieval_miss",
    "ranking_miss",
    "reranker_demotion",
    "partial_evidence",
    "refused_with_evidence",
    "unsupported_claims",
    "misread_evidence",
    "answered_unanswerable",
)
RETRIEVAL_FAILURES = (
    "lost_in_chunking",
    "retrieval_miss",
    "ranking_miss",
    "reranker_demotion",
    "partial_evidence",
)
STAGE_OF = {c: "retrieval" if c in RETRIEVAL_FAILURES else "generation" for c in CATEGORIES}

# Patterns seen in the hand review, counted over every row by `checks`.
EXTRA_DETAIL = re.compile(  # the grader faults facts that aren't in the highlighted clause or the reference
    r"not (?:present|mentioned|found|included|stated|contained|specified) in (?:the )?"
    r"(?:provided |highlighted )?(?:clause|text|reference)|extraneous|not requested|not asked",
    re.IGNORECASE,
)
SAYS_NOT_FOUND = re.compile(  # the answer itself says the contract doesn't have it
    r"\b(?:does not|do not|doesn't|did not) (?:contain|specify|mention|include|state|provide|address|define)"
    r"|not (?:specified|explicitly|stated|mentioned)|no (?:specific|explicit)",
    re.IGNORECASE,
)
_STOP = frozenset(
    re.findall(
        r"\w+",
        "the a an of to in on for and or by with under between is are be does do what which who when how "
        "long any either party parties agreement this that it its from as at can may shall will",
    )
)
_STEMMER = Stemmer.Stemmer("english")


def content_words(text: str) -> set[str]:
    """Stemmed words that carry meaning (no stopwords, no words under three letters)."""
    words = [w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 2]
    return set(_STEMMER.stemWords(words))


class SpanTrace(BaseModel):
    """Where one piece of gold evidence got to. Ranks are 1-indexed; None means never ranked."""

    span_index: int
    covered: bool  # some chunk of the contract holds it
    candidate_rank: int | None  # best rank of an evidence chunk in any first-stage list (BM25 / dense)
    prerank_rank: int | None  # best rank in the list the final cut comes from, before any reranker
    final_rank: int | None  # best rank among the chunks handed to the generator
    delivered: bool  # the text shown to the generator contains it


class Triage(BaseModel):
    config_id: str
    qid: str
    qtype: str
    answerable: bool
    category: str  # "correct" or one of CATEGORIES
    correctness: float | None
    faithfulness: float | None
    abstained: bool
    spans: list[SpanTrace] = []
    final_recall: float | None = None  # share of evidence pieces with a relevant chunk in the final list
    delivered_recall: float | None = None  # share of evidence pieces inside the text the generator saw


def shown_range(chunk: Chunk, full_text: str | None) -> tuple[int, int]:
    """Character range of the text the generator was shown for this chunk (its sentence window, if any)."""
    window = chunk.context_text
    if not window or full_text is None:
        return chunk.char_start, chunk.char_end
    start = full_text.find(window, max(0, chunk.char_start - len(window)))
    if start == -1 or not (start <= chunk.char_start and chunk.char_end <= start + len(window)):
        return chunk.char_start, chunk.char_end  # the window isn't around the sentence: don't guess
    return start, start + len(window)


def covers(start: int, end: int, al: SpanAlignment, min_frac: float = MIN_OVERLAP_FRAC) -> bool:
    """The labels' relevance rule applied to an arbitrary range: overlap >= min_frac of the smaller side."""
    ov = overlap(start, end, al.char_start, al.char_end)
    return ov > 0 and ov >= min_frac * min(end - start, al.char_end - al.char_start)


def prerank_list(result: RetrievalResult) -> list[RetrievedChunk]:
    """The ranking the final cut is taken from, before any reranker: fusion, else the single retriever."""
    for stage in ("fusion", "dense", "bm25"):
        if stage in result.stages:
            return result.stages[stage]
    return result.final


def _best_rank(items: Iterable[RetrievedChunk], ids: set[str]) -> int | None:
    ranks = [r.rank for r in items if r.chunk_id in ids]
    return min(ranks) if ranks else None


def trace_spans(
    labels: QuestionLabels, result: RetrievalResult, chunks: dict[str, Chunk], full_text: str | None
) -> list[SpanTrace]:
    k_final = len(result.final)
    first_stage = [r for stage in ("bm25", "dense") for r in result.stages.get(stage, [])]
    prerank = prerank_list(result)[:k_final] if "rerank" in result.stages else result.final
    shown = [shown_range(chunks[r.chunk_id], full_text) for r in result.final if r.chunk_id in chunks]
    traces = []
    for al, group in zip(labels.alignments, labels.span_chunks, strict=True):
        ids = set(group)
        traces.append(
            SpanTrace(
                span_index=al.span_index,
                covered=bool(ids),
                candidate_rank=_best_rank(first_stage or result.final, ids),
                prerank_rank=_best_rank(prerank, ids),
                final_rank=_best_rank(result.final, ids),
                delivered=any(covers(s, e, al) for s, e in shown),
            )
        )
    return traces


def declined_as_scored(question: EvalQuestion, generation: GenerationResult, score: EvalScores) -> bool:
    """Whether the answer counts as a refusal, as its scores define it (the flag, when there is no score)."""
    if score.abstention_correct is None:
        return generation.abstained
    return score.abstention_correct == (not question.answerable)


def categorize(
    question: EvalQuestion,
    spans: Sequence[SpanTrace],
    generation: GenerationResult,
    score: EvalScores,
    reranked: bool,
) -> str:
    declined = declined_as_scored(question, generation, score)
    if not question.answerable:
        return "correct" if declined else "answered_unanswerable"
    if score.answer_correctness == 1.0:
        return "correct"  # graded correct on its content, even if the abstention flag was set
    if not spans:
        return "lost_in_chunking"  # evidence exists (answerable) but could not be placed in any chunk
    if not any(s.delivered for s in spans):
        if not any(s.covered for s in spans):
            return "lost_in_chunking"
        if not any(s.candidate_rank for s in spans):
            return "retrieval_miss"
        if reranked and any(s.prerank_rank for s in spans):
            return "reranker_demotion"
        return "ranking_miss"
    if not all(s.delivered for s in spans):
        return "partial_evidence"
    if declined:
        return "refused_with_evidence"
    if score.faithfulness is not None and score.faithfulness < 1.0:
        return "unsupported_claims"
    return "misread_evidence"


def triage_row(
    question: EvalQuestion,
    labels: QuestionLabels,
    result: RetrievalResult,
    generation: GenerationResult,
    score: EvalScores,
    chunks: dict[str, Chunk],
    full_text: str | None,
) -> Triage:
    spans = trace_spans(labels, result, chunks, full_text) if question.answerable else []
    n = len(spans)
    return Triage(
        config_id=score.config_id,
        qid=question.qid,
        qtype=question.qtype,
        answerable=question.answerable,
        category=categorize(question, spans, generation, score, "rerank" in result.stages),
        correctness=score.answer_correctness,
        faithfulness=score.faithfulness,
        abstained=declined_as_scored(question, generation, score),
        spans=spans,
        final_recall=sum(s.final_rank is not None for s in spans) / n if n else None,
        delivered_recall=sum(s.delivered for s in spans) / n if n else None,
    )


# ---------- summary and report ----------


def _mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def summarize(rows: Sequence[Triage], descriptions: dict[str, str]) -> dict[str, Any]:
    by_config: dict[str, list[Triage]] = defaultdict(list)
    for r in rows:
        by_config[r.config_id].append(r)
    configs = {}
    for key, rs in by_config.items():
        configs[key] = {
            "description": descriptions.get(key, key),
            "categories": dict(Counter(r.category for r in rs)),
            "final_recall": _mean(r.final_recall for r in rs),
            "delivered_recall": _mean(r.delivered_recall for r in rs),
        }
    failures = [r for r in rows if r.category != "correct"]
    per_question: dict[str, list[Triage]] = defaultdict(list)
    for r in rows:
        per_question[r.qid].append(r)
    hard = sorted(
        (
            {
                "qid": qid,
                "qtype": rs[0].qtype,
                "mean_correctness": _mean(r.correctness for r in rs),
                "categories": dict(Counter(r.category for r in rs)),
            }
            for qid, rs in per_question.items()
            if rs[0].answerable
        ),
        key=lambda h: (h["mean_correctness"] if h["mean_correctness"] is not None else 1.0, h["qid"]),
    )
    return {
        "rows": len(rows),
        "failures": len(failures),
        "overall": dict(Counter(r.category for r in failures)),
        "by_stage": dict(Counter(STAGE_OF[r.category] for r in failures)),
        "by_qtype": {
            t: dict(Counter(r.category for r in failures if r.qtype == t))
            for t in sorted({r.qtype for r in rows})
        },
        "correct_without_delivered_evidence": sum(
            1 for r in rows if r.category == "correct" and r.answerable and r.delivered_recall == 0
        ),
        "configs": configs,
        "hardest_questions": [h for h in hard if (h["mean_correctness"] or 0) <= 0.25],
    }


def checks(
    rows: Sequence[Triage],
    scores: dict[tuple[str, str], EvalScores],
    generations: dict[tuple[str, str], GenerationResult],
    questions: dict[str, EvalQuestion],
    preamble_questions: set[str],
) -> dict[str, Any]:
    """Evidence for or against the patterns seen in the hand review, over every row rather than 20 cases."""
    reasons = []
    for r in rows:
        rationale = scores[(r.config_id, r.qid)].judge_rationale
        grade = json.loads(rationale).get("grade") if rationale else None
        if grade and grade["correctness"] != "correct":
            reasons.append((r, grade["reason"]))
    extra = [r for r, reason in reasons if EXTRA_DETAIL.search(reason)]

    answered = [r for r in rows if r.category == "answered_unanswerable"]
    not_found = [r for r in answered if SAYS_NOT_FOUND.search(generations[(r.config_id, r.qid)].answer)]
    refusals = [r for r in rows if r.answerable and r.abstained]
    redacted = []
    for r in refusals:
        answer = generations[(r.config_id, r.qid)].answer
        if "[***]" in answer or "redact" in answer.lower():
            redacted.append(r)

    shared: dict[bool, list[float]] = {True: [], False: []}
    for r in rows:
        q = questions[r.qid]
        words = content_words(q.question.replace(q.contract_name, " ") if q.contract_name else q.question)
        for s in r.spans:
            if words:
                shared[s.delivered].append(
                    len(words & content_words(q.evidence_spans[s.span_index])) / len(words)
                )

    answerable = {r.qid for r in rows if r.answerable}

    def retrieval_failure_rate(qids: set[str]) -> float | None:
        rs = [r for r in rows if r.qid in qids]
        return _mean(float(STAGE_OF.get(r.category) == "retrieval") for r in rs)

    return {
        "grades_not_correct": len(reasons),
        "grader_cites_detail_outside_clause": len(extra),
        "grader_cites_detail_outside_clause_by_category": dict(Counter(r.category for r in extra)),
        "answered_unanswerable": len(answered),
        "answered_unanswerable_text_says_not_found": len(not_found),
        "refusals_on_answerable": len(refusals),
        "refusals_explaining_a_redaction": len(redacted),
        "question_words_in_delivered_evidence": _mean(shared[True]),
        "question_words_in_missed_evidence": _mean(shared[False]),
        "evidence_pieces_delivered": len(shared[True]),
        "evidence_pieces_missed": len(shared[False]),
        "preamble_questions": len(preamble_questions & answerable),
        "retrieval_failure_rate_preamble": retrieval_failure_rate(preamble_questions & answerable),
        "retrieval_failure_rate_other": retrieval_failure_rate(answerable - preamble_questions),
    }


def render_report(run_id: str, summary: dict[str, Any]) -> str:
    cats = [c for c in CATEGORIES if any(c in v["categories"] for v in summary["configs"].values())]
    lines = [
        "# Failure triage",
        "",
        f"Source run `{run_id}` · {summary['rows']} answers · {summary['failures']} not fully correct "
        "(wrong, partial, refused when answerable, or answered when unanswerable).",
        "",
        "Each failure is traced through the pipeline and assigned the first stage that failed "
        "(see `src/contract_rag/analysis/failures.py`).",
        "",
        "## Failures by category",
        "",
        "| Category | Stage | Count | Share of failures |",
        "|---|---|---|---|",
    ]
    total = max(summary["failures"], 1)
    for c, n in sorted(summary["overall"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {c} | {STAGE_OF[c]} | {n} | {n / total:.0%} |")
    lines += [
        "",
        f"Correct answers where none of the labelled evidence reached the generator: "
        f"{summary['correct_without_delivered_evidence']} (the answer came from text the labels don't mark).",
        "",
        "## By configuration",
        "",
        "| Configuration | recall in final list | evidence the generator saw | " + " | ".join(cats) + " |",
        "|---|---|---|" + "---|" * len(cats),
    ]
    for c in summary["configs"].values():
        counts = " | ".join(str(c["categories"].get(k, 0)) for k in cats)
        lines.append(
            f"| {c['description']} | {c['final_recall']:.2f} | {c['delivered_recall']:.2f} | {counts} |"
        )
    lines += [
        "",
        "## By question type",
        "",
        "| Type | " + " | ".join(cats) + " |",
        "|---|" + "---|" * len(cats),
    ]
    for t, counts in summary["by_qtype"].items():
        lines.append(f"| {t} | " + " | ".join(str(counts.get(k, 0)) for k in cats) + " |")
    lines += [
        "",
        "## Hardest questions (mean correctness across configurations <= 0.25)",
        "",
        "| Question | Type | Mean correctness | Categories |",
        "|---|---|---|---|",
    ]
    for h in summary["hardest_questions"]:
        mix = ", ".join(f"{k} {v}" for k, v in sorted(h["categories"].items(), key=lambda kv: -kv[1]))
        lines.append(f"| {h['qid']} | {h['qtype']} | {h['mean_correctness']:.2f} | {mix} |")
    c = summary.get("checks")
    if c:

        def pct(part: float, whole: float) -> str:
            return f"{part / whole:.0%}" if whole else "n/a"

        cited, not_correct = c["grader_cites_detail_outside_clause"], c["grades_not_correct"]
        lines += [
            "",
            "## Checks on patterns from the hand review (all rows)",
            "",
            f"- **Grader strictness:** of {not_correct} answers not graded correct, {cited} "
            f"({pct(cited, not_correct)}) give a reason citing details outside the highlighted clause or "
            "the reference.",
            f"- **Abstention flag:** {c['answered_unanswerable_text_says_not_found']} of "
            f"{c['answered_unanswerable']} answers to unanswerable questions say in their text that the "
            "information isn't there, without setting the abstention flag; "
            f"{c['refusals_explaining_a_redaction']} of {c['refusals_on_answerable']} refusals on "
            "answerable questions explain a redacted value.",
            f"- **Vocabulary gap:** {c['question_words_in_delivered_evidence']:.2f} of a question's "
            "content words appear in evidence that reached the generator "
            f"(n={c['evidence_pieces_delivered']}), against {c['question_words_in_missed_evidence']:.2f} "
            f"in evidence that didn't (n={c['evidence_pieces_missed']}).",
            f"- **Preamble:** {c['preamble_questions']} questions have evidence in a contract's first "
            "fixed-size chunk; their retrieval-failure rate is "
            f"{pct(c['retrieval_failure_rate_preamble'] or 0, 1)} against "
            f"{pct(c['retrieval_failure_rate_other'] or 0, 1)} for the rest.",
        ]
    return "\n".join(lines) + "\n"


# ---------- running ----------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def run(
    run_dir: Path,
    out_dir: Path,
    repo_root: Path,
    questions_path: Path = Path("data/eval/questions.jsonl"),
    documents_path: Path = Path("data/processed/documents.jsonl"),
    retrieval_config: Path = Path("configs/retrieval.yaml"),
) -> dict[str, Any]:
    summary_in = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    keys = list(summary_in["configs"])
    descriptions = {k: v["description"] for k, v in summary_in["configs"].items()}
    lines = (repo_root / questions_path).read_text(encoding="utf-8").splitlines()
    questions = {q.qid: q for q in (EvalQuestion.model_validate_json(x) for x in lines if x.strip())}
    doc_lines = (repo_root / documents_path).read_text(encoding="utf-8").splitlines()
    docs = {d.doc_id: d for d in (Document.model_validate_json(x) for x in doc_lines if x.strip())}
    index_cfg = IndexConfig.model_validate(yaml.safe_load((repo_root / retrieval_config).read_text()))

    retrieved = {(r["config_id"], r["qid"]): r for r in _read_jsonl(run_dir / "retrieval.jsonl")}
    generated = {
        (g["config_id"], g["qid"]): g["generation"] for g in _read_jsonl(run_dir / "generation.jsonl")
    }
    scores = [EvalScores.model_validate(s) for s in _read_jsonl(run_dir / "scores.jsonl")]
    scored_q = [questions[q] for q in sorted({s.qid for s in scores})]

    chunks: dict[str, Chunk] = {}
    labels: dict[tuple[str, str], QuestionLabels] = {}
    for chunking in sorted({k.split("__")[0] for k in keys}):
        loaded = load_chunks(repo_root / index_cfg.chunks_dir / f"{chunking}.jsonl")
        chunks.update({c.chunk_id: c for c in loaded})
        for lab in build_labels(scored_q, docs, loaded, chunking):  # type: ignore[arg-type]
            labels[(chunking, lab.qid)] = lab

    rows = []
    for s in scores:
        if s.config_id not in keys:
            continue
        q = questions[s.qid]
        doc = docs.get(q.doc_id or "")
        rows.append(
            triage_row(
                q,
                labels[(s.config_id.split("__")[0], s.qid)],
                RetrievalResult.model_validate(retrieved[(s.config_id, s.qid)]),
                GenerationResult.model_validate(generated[(s.config_id, s.qid)]),
                s,
                chunks,
                doc.full_text if doc else None,
            )
        )
    summary = summarize(rows, descriptions)
    preamble = {
        qid
        for (chunking, qid), lab in labels.items()
        if chunking == "fixed" and any(cid.endswith("::00000") for group in lab.span_chunks for cid in group)
    }
    summary["checks"] = checks(
        rows,
        {(s.config_id, s.qid): s for s in scores},
        {k: GenerationResult.model_validate(v) for k, v in generated.items()},
        questions,
        preamble,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "triage.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(run_dir.name, summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=Path("runs/matrix-v1"))
    parser.add_argument("--out", type=Path, default=Path("runs/failure-analysis-v1"))
    args = parser.parse_args()
    summary = run(args.run_dir, args.out, Path.cwd().resolve())
    print(json.dumps({k: summary[k] for k in ("rows", "failures", "overall", "by_stage")}, indent=2))


if __name__ == "__main__":
    main()
