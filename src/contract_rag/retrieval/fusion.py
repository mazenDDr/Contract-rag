"""Merging ranked lists from different retrievers into one.

BM25 scores and cosine similarities live on different scales, so they cannot simply be added. Reciprocal
Rank Fusion ignores scores and uses positions; weighted fusion rescales each list to [0, 1] first.
"""

from __future__ import annotations

from collections.abc import Sequence

from contract_rag.schemas import RetrievedChunk


def _ranked(scores: dict[str, float], tiebreak: dict[str, tuple]) -> list[RetrievedChunk]:
    order = sorted(scores, key=lambda cid: (-scores[cid], *tiebreak[cid], cid))
    return [
        RetrievedChunk(chunk_id=cid, score=scores[cid], rank=rank, stage="fusion")
        for rank, cid in enumerate(order, start=1)
    ]


def reciprocal_rank_fusion(lists: Sequence[Sequence[RetrievedChunk]], k: int = 60) -> list[RetrievedChunk]:
    """score = sum over lists of 1 / (k + rank). Small k trusts first places; large k rewards agreement."""
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranked in lists:
        for item in ranked:
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (k + item.rank)
            best_rank[item.chunk_id] = min(best_rank.get(item.chunk_id, item.rank), item.rank)
    return _ranked(scores, {cid: (best_rank[cid],) for cid in scores})


def _min_max(items: Sequence[RetrievedChunk]) -> dict[str, float]:
    if not items:
        return {}
    low, high = min(i.score for i in items), max(i.score for i in items)
    return {i.chunk_id: 1.0 if high == low else (i.score - low) / (high - low) for i in items}


def weighted_fusion(
    sparse: Sequence[RetrievedChunk], dense: Sequence[RetrievedChunk], alpha: float
) -> list[RetrievedChunk]:
    """alpha * dense + (1 - alpha) * sparse after min-max scaling each list; a chunk absent from a list
    gets 0 there. alpha must be tuned on dev questions only."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    s, d = _min_max(sparse), _min_max(dense)
    scores = {cid: alpha * d.get(cid, 0.0) + (1 - alpha) * s.get(cid, 0.0) for cid in s.keys() | d.keys()}
    return _ranked(scores, {cid: () for cid in scores})
