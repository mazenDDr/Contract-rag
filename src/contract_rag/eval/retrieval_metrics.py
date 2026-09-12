"""Label-based retrieval metrics. This is the only place recall is computed, so every run is comparable.

Two notions of recall:
- evidence recall@k (primary): fraction of gold evidence spans with at least one relevant chunk in the
  top k. The generator needs each piece of evidence once, however many chunks it was cut into.
- chunk recall@k (diagnostic): fraction of all relevant chunks in the top k. It penalizes chunkers that
  split one span into many pieces, so it is NOT comparable across chunking strategies.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np

from contract_rag.eval.labels import QuestionLabels
from contract_rag.schemas import EvalScores, RetrievalResult

DEFAULT_KS = (1, 3, 5, 10, 20)
FLOAT_METRICS = (
    "mrr",
    "ndcg_at_10",
    "context_precision",
    "context_recall",
    "faithfulness",
    "citation_validity",
    "answer_relevance",
    "answer_correctness",
)


# ---------- per-question metrics on a ranked list of chunk_ids ----------


def evidence_recall_at_k(ranked: Sequence[str], labels: QuestionLabels, k: int) -> float | None:
    if not labels.has_evidence:
        return None
    top = set(ranked[:k])
    return sum(bool(top.intersection(group)) for group in labels.span_chunks) / len(labels.span_chunks)


def chunk_recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    return float(bool(relevant.intersection(ranked[:k])))


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float | None:
    if not relevant:
        return None
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float | None:
    """Binary-relevance nDCG: rewards relevant chunks near the top, normalized by the best possible order."""
    if not relevant:
        return None
    dcg = sum(1.0 / math.log2(i + 1) for i, cid in enumerate(ranked[:k], start=1) if cid in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def context_precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float | None:
    """RAGAS-style context precision: mean of precision@i over the ranks i <= k that hold a relevant chunk."""
    if not relevant:
        return None
    hits, total = 0, 0.0
    for i, cid in enumerate(ranked[:k], start=1):
        if cid in relevant:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


def score_retrieval(
    result: RetrievalResult,
    labels: QuestionLabels,
    ks: Sequence[int] = DEFAULT_KS,
    stage: str | None = None,
) -> EvalScores:
    """Score the final list (default) or one intermediate stage ("bm25", "dense", "fusion", "rerank").

    Context precision/recall are measured at the generator's cut-off (len(result.final)) so that stage
    lists are judged against the same budget as the final list.
    """
    items = result.final if stage is None else result.stages[stage]
    ranked = [r.chunk_id for r in sorted(items, key=lambda r: r.rank)]
    scores = EvalScores(
        qid=result.qid, config_id=result.config_id if stage is None else f"{result.config_id}@{stage}"
    )
    if not labels.has_evidence:
        return scores  # unanswerable or unlabeled: retrieval metrics are undefined
    relevant = labels.relevant
    k_final = len(result.final)
    scores.recall_at = {k: evidence_recall_at_k(ranked, labels, k) for k in ks}
    scores.mrr = reciprocal_rank(ranked, relevant)
    scores.ndcg_at_10 = ndcg_at_k(ranked, relevant, 10)
    scores.context_precision = context_precision_at_k(ranked, relevant, k_final)
    scores.context_recall = evidence_recall_at_k(ranked, labels, k_final)
    return scores


# ---------- aggregation with uncertainty ----------


def metric_table(scores: Iterable[EvalScores]) -> dict[str, dict[str, float]]:
    """{metric_name: {qid: value}}, skipping undefined values. recall_at[k] becomes "recall@k"."""
    table: dict[str, dict[str, float]] = defaultdict(dict)
    for s in scores:
        for k, v in s.recall_at.items():
            table[f"recall@{k}"][s.qid] = v
        for name in FLOAT_METRICS:
            v = getattr(s, name)
            if v is not None:
                table[name][s.qid] = v
        if s.abstention_correct is not None:
            table["abstention_accuracy"][s.qid] = float(s.abstention_correct)
    return dict(table)


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(mean, ci_low, ci_high) from a percentile bootstrap over questions."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def aggregate(scores: Iterable[EvalScores], **ci_kwargs) -> dict[str, dict[str, float]]:
    out = {}
    for metric, by_qid in metric_table(scores).items():
        mean, lo, hi = bootstrap_ci(list(by_qid.values()), **ci_kwargs)
        out[metric] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n": len(by_qid)}
    return out


def paired_difference(
    baseline: Iterable[EvalScores], candidate: Iterable[EvalScores], metric: str, **ci_kwargs
) -> dict[str, float]:
    """Candidate minus baseline on the questions both scored. A CI that excludes 0 is a real difference."""
    a = metric_table(baseline).get(metric, {})
    b = metric_table(candidate).get(metric, {})
    shared = sorted(a.keys() & b.keys())
    mean, lo, hi = bootstrap_ci([b[q] - a[q] for q in shared], **ci_kwargs)
    return {"mean_diff": mean, "ci_low": lo, "ci_high": hi, "n": len(shared)}
