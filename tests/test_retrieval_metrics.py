import math

import pytest

from contract_rag.eval.labels import QuestionLabels
from contract_rag.eval.retrieval_metrics import (
    aggregate,
    bootstrap_ci,
    chunk_recall_at_k,
    context_precision_at_k,
    evidence_recall_at_k,
    hit_at_k,
    ndcg_at_k,
    paired_difference,
    reciprocal_rank,
    score_retrieval,
)
from contract_rag.schemas import EvalScores, RetrievalResult, RetrievedChunk

# two evidence spans: the first was cut into chunks a+b, the second lives in chunk c
LABELS = QuestionLabels(
    qid="q1", doc_id="d1", strategy="fixed", answerable=True, span_chunks=[["a", "b"], ["c"]]
)
RANKED = ["x", "b", "y", "c", "z"]
RELEVANT = {"a", "b", "c"}


def test_evidence_recall_counts_spans_not_chunks():
    assert evidence_recall_at_k(RANKED, LABELS, 1) == 0.0
    assert evidence_recall_at_k(RANKED, LABELS, 2) == 0.5
    assert evidence_recall_at_k(RANKED, LABELS, 4) == 1.0  # "a" missing doesn't matter: its span has "b"
    assert chunk_recall_at_k(RANKED, RELEVANT, 4) == pytest.approx(2 / 3)


def test_uncovered_span_counts_as_miss():
    labels = QuestionLabels(qid="q", doc_id="d", strategy="fixed", answerable=True, span_chunks=[["a"], []])
    assert evidence_recall_at_k(["a"], labels, 1) == 0.5


def test_rank_metrics():
    assert hit_at_k(RANKED, RELEVANT, 1) == 0.0 and hit_at_k(RANKED, RELEVANT, 2) == 1.0
    assert reciprocal_rank(RANKED, RELEVANT) == 0.5
    assert reciprocal_rank(["x"], RELEVANT) == 0.0
    ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, RELEVANT, 5) == pytest.approx((1 / math.log2(3) + 1 / math.log2(5)) / ideal)
    assert context_precision_at_k(RANKED, RELEVANT, 5) == pytest.approx((1 / 2 + 2 / 4) / 2)
    assert context_precision_at_k(["a", "b", "c"], RELEVANT, 3) == 1.0


def test_metrics_undefined_without_evidence():
    empty = QuestionLabels(qid="q", doc_id="d", strategy="fixed", answerable=False)
    assert evidence_recall_at_k(RANKED, empty, 5) is None
    assert reciprocal_rank(RANKED, set()) is None


def _ranked(ids, stage):
    return [
        RetrievedChunk(chunk_id=c, score=1.0 / (i + 1), rank=i + 1, stage=stage) for i, c in enumerate(ids)
    ]


def test_score_retrieval_final_and_stage():
    result = RetrievalResult(
        qid="q1",
        config_id="cfg",
        final=_ranked(["b", "c", "x"], "rerank"),
        stages={"bm25": _ranked(["x", "y", "z", "b"], "bm25")},
    )
    final = score_retrieval(result, LABELS)
    assert final.recall_at[1] == 0.5 and final.recall_at[3] == 1.0
    assert final.mrr == 1.0 and final.context_recall == 1.0

    bm25 = score_retrieval(result, LABELS, stage="bm25")
    assert bm25.config_id == "cfg@bm25"
    assert bm25.mrr == 0.25
    assert bm25.context_recall == 0.0  # right chunk is at rank 4, outside the 3-chunk budget


def test_bootstrap_and_aggregate():
    assert bootstrap_ci([1.0, 1.0, 1.0]) == (1.0, 1.0, 1.0)
    assert bootstrap_ci([0.0, 1.0] * 10, seed=1) == bootstrap_ci([0.0, 1.0] * 10, seed=1)
    scores = [
        EvalScores(qid="q1", config_id="c", recall_at={5: 1.0}, mrr=1.0),
        EvalScores(qid="q2", config_id="c", recall_at={5: 0.0}, mrr=0.5),
        EvalScores(qid="q3", config_id="c"),  # unanswerable: skipped, not counted as 0
    ]
    agg = aggregate(scores)
    assert agg["recall@5"]["mean"] == 0.5 and agg["recall@5"]["n"] == 2
    assert agg["mrr"]["mean"] == 0.75


def test_paired_difference():
    base = [EvalScores(qid=q, config_id="a", recall_at={5: v}) for q, v in (("q1", 0), ("q2", 0), ("q3", 1))]
    cand = [EvalScores(qid=q, config_id="b", recall_at={5: 1.0}) for q in ("q1", "q2", "q3")]
    diff = paired_difference(base, cand, "recall@5")
    assert diff["mean_diff"] == pytest.approx(2 / 3) and diff["n"] == 3
