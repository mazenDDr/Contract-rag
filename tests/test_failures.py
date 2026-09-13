import json

from contract_rag.analysis.failures import (
    checks,
    content_words,
    render_report,
    shown_range,
    summarize,
    triage_row,
)
from contract_rag.eval.labels import build_labels
from contract_rag.schemas import (
    Chunk,
    Document,
    EvalQuestion,
    EvalScores,
    GenerationResult,
    RetrievalResult,
    RetrievedChunk,
)

CLAUSES = [
    "This Agreement is governed by the laws of the State of Delaware.",
    "Either party may terminate this Agreement on ninety days notice.",
    "The licensee shall pay royalties every quarter.",
    "Each party shall keep the other's information confidential.",
]
TEXT = " ".join(CLAUSES)
DOC = Document(doc_id="d1", title="t", source_path="x", num_pages=1, full_text=TEXT)


def _chunks(strategy="fixed", windows=False):
    out, pos = [], 0
    for i, clause in enumerate(CLAUSES):
        window = None
        if windows:  # the sentence plus its neighbours, as the sentence-window chunker stores it
            window = " ".join(CLAUSES[max(0, i - 1) : i + 2])
        out.append(
            Chunk(
                chunk_id=f"d1::{strategy}::{i:05d}",
                doc_id="d1",
                strategy=strategy,
                text=clause,
                context_text=window,
                page_start=1,
                page_end=1,
                char_start=pos,
                char_end=pos + len(clause),
                token_count=0,
            )
        )
        pos += len(clause) + 1
    return out


def _question(spans, answerable=True, qtype="cuad_derived"):
    return EvalQuestion(
        qid="q1",
        question="?",
        doc_id="d1",
        qtype=qtype if answerable else "unanswerable",
        category="c",
        reference_answer="r" if answerable else "",
        evidence_spans=spans,
        answerable=answerable,
        split="test",
    )


def _ranked(ids, stage):
    return [RetrievedChunk(chunk_id=c, score=1.0, rank=i, stage=stage) for i, c in enumerate(ids, 1)]


def _result(final, stages):
    return RetrievalResult(
        qid="q1",
        config_id="cfg",
        final=_ranked(final, "bm25"),
        stages={s: _ranked(ids, s) for s, ids in stages.items()},
    )


def _triage(question, result, chunks, abstained=False, correctness=0.0, faithfulness=1.0):
    labels = build_labels([question], {"d1": DOC}, chunks, chunks[0].strategy)[0]
    gen = GenerationResult(qid="q1", config_id="cfg", model="m", answer="a", abstained=abstained)
    score = EvalScores(qid="q1", config_id="cfg", answer_correctness=correctness, faithfulness=faithfulness)
    return triage_row(question, labels, result, gen, score, {c.chunk_id: c for c in chunks}, TEXT)


C = [c.chunk_id for c in _chunks()]


def test_each_failure_stage_is_named():
    chunks = _chunks()
    one = _question([CLAUSES[1]])
    assert _triage(one, _result([C[1]], {"bm25": [C[1]]}), chunks, correctness=1.0).category == "correct"
    assert _triage(one, _result([C[0]], {"bm25": [C[0], C[2]]}), chunks).category == "retrieval_miss"
    assert _triage(one, _result([C[0]], {"bm25": [C[0], C[1]]}), chunks).category == "ranking_miss"
    # evidence was first before the reranker, and the reranker cut it
    reranked = _result([C[0]], {"bm25": [C[1], C[0]], "rerank": [C[0], C[1]]})
    assert _triage(one, reranked, chunks).category == "reranker_demotion"
    found = _result([C[1]], {"bm25": [C[1]]})
    assert _triage(one, found, chunks, abstained=True).category == "refused_with_evidence"
    assert _triage(one, found, chunks, faithfulness=0.5).category == "unsupported_claims"
    assert _triage(one, found, chunks, faithfulness=1.0, correctness=0.5).category == "misread_evidence"
    two = _question([CLAUSES[1], CLAUSES[3]], qtype="multi_span")
    partial = _triage(two, _result([C[1]], {"bm25": [C[1], C[3]]}), chunks)
    assert partial.category == "partial_evidence" and partial.delivered_recall == 0.5


def test_unanswerable_questions_are_judged_on_abstention():
    chunks = _chunks()
    none = _question([], answerable=False)
    result = _result([C[0]], {"bm25": [C[0]]})
    assert _triage(none, result, chunks, abstained=True).category == "correct"
    assert _triage(none, result, chunks, abstained=False).category == "answered_unanswerable"


def test_evidence_in_a_sentence_window_counts_as_delivered():
    chunks = _chunks("sentence_window", windows=True)
    ids = [c.chunk_id for c in chunks]
    assert shown_range(chunks[0], TEXT) == (0, len(CLAUSES[0]) + 1 + len(CLAUSES[1]))
    # the evidence is clause 1; the final list only has clause 0, whose window includes clause 1
    row = _triage(
        _question([CLAUSES[1]]), _result([ids[0]], {"bm25": [ids[0], ids[1]]}), chunks, faithfulness=0.5
    )
    assert row.final_recall == 0.0 and row.delivered_recall == 1.0
    assert row.category == "unsupported_claims"  # judged on what the generator saw, not on recall@k


def test_summary_and_report():
    chunks = _chunks()
    q = _question([CLAUSES[1]])
    rows = [
        _triage(q, _result([C[1]], {"bm25": [C[1]]}), chunks, correctness=1.0),
        _triage(q, _result([C[0]], {"bm25": [C[0]]}), chunks).model_copy(update={"config_id": "cfg2"}),
    ]
    summary = summarize(rows, {"cfg": "one", "cfg2": "two"})
    assert summary["failures"] == 1 and summary["overall"] == {"retrieval_miss": 1}
    assert summary["by_stage"] == {"retrieval": 1}
    assert summary["hardest_questions"] == []  # mean correctness 0.5 is not among the hardest
    report = render_report("m1", summary)
    assert report.startswith("# Failure triage") and "| retrieval_miss | retrieval | 1 | 100% |" in report


def test_checks_count_the_hand_review_patterns():
    chunks = _chunks()
    answerable = _question([CLAUSES[1]])
    unanswerable = _question([], answerable=False).model_copy(update={"qid": "q2"})
    misread = _triage(answerable, _result([C[1]], {"bm25": [C[1]]}), chunks, correctness=0.5)
    answered = _triage(unanswerable, _result([C[0]], {"bm25": [C[0]]}), chunks).model_copy(
        update={"qid": "q2", "config_id": "cfg2"}
    )
    grade = {"correctness": "partial", "reason": "It adds a fee not present in the provided clause."}
    scores = {
        ("cfg", "q1"): EvalScores(qid="q1", config_id="cfg", judge_rationale=json.dumps({"grade": grade})),
        ("cfg2", "q2"): EvalScores(qid="q2", config_id="cfg2", judge_rationale=json.dumps({"grade": None})),
    }
    generations = {
        ("cfg", "q1"): GenerationResult(qid="q1", config_id="cfg", model="m", answer="Ninety days [1]."),
        ("cfg2", "q2"): GenerationResult(
            qid="q2", config_id="cfg2", model="m", answer="The agreement does not specify a warranty period."
        ),
    }
    out = checks([misread, answered], scores, generations, {"q1": answerable, "q2": unanswerable}, {"q1"})
    assert out["grades_not_correct"] == 1 and out["grader_cites_detail_outside_clause"] == 1
    assert out["answered_unanswerable"] == 1 and out["answered_unanswerable_text_says_not_found"] == 1
    assert out["preamble_questions"] == 1 and out["retrieval_failure_rate_preamble"] == 0.0
    assert content_words("Does the Agreement restrict either party from competing?") == {"restrict", "compet"}
