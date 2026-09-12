"""Gold relevance labels: which chunks contain the evidence for a question.

CUAD gives evidence as verbatim text spans. Its char offsets index CUAD's own .txt files, not our PDF
parse, so we locate each span in `Document.full_text` by normalized text matching (exact first, fuzzy
fallback), then mark every chunk whose char range overlaps the located span enough.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from contract_rag.schemas import Chunk, ChunkStrategy, Document, EvalQuestion

FUZZY_THRESHOLD = 85.0
MIN_OVERLAP_FRAC = 0.5


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Lowercase, keep alphanumerics, collapse every run of anything else to one space.

    Returns the normalized string plus, for each of its characters, the index in `text` it came from,
    so a match in normalized space can be mapped back to original offsets.
    """
    out: list[str] = []
    index_map: list[int] = []
    pending_space = False
    for i, ch in enumerate(text):
        for c in ch.lower():
            if c.isalnum():
                if pending_space and out:
                    out.append(" ")
                    index_map.append(i)
                out.append(c)
                index_map.append(i)
                pending_space = False
            else:
                pending_space = True
    return "".join(out), index_map


def normalize(text: str) -> str:
    return normalize_with_map(text)[0]


class SpanAlignment(BaseModel):
    span_index: int
    char_start: int
    char_end: int
    score: float
    method: Literal["exact", "fuzzy"]
    n_occurrences: int = 1  # exact matches only; >1 means the span is ambiguous within the doc


class QuestionLabels(BaseModel):
    qid: str
    doc_id: str | None
    strategy: ChunkStrategy
    answerable: bool
    alignments: list[SpanAlignment] = Field(default_factory=list)
    # relevant chunk_ids per *aligned* evidence span (parallel to `alignments`); an empty group means
    # no chunk covers that evidence, which is a real pipeline miss (e.g. text dropped by the parser)
    span_chunks: list[list[str]] = Field(default_factory=list)
    unaligned_spans: list[int] = Field(default_factory=list)  # evidence we could not locate: label noise

    @property
    def relevant(self) -> set[str]:
        return {cid for group in self.span_chunks for cid in group}

    @property
    def has_evidence(self) -> bool:
        return bool(self.span_chunks)


class DocText:
    """Normalized view of one document, built once and reused for every question about it."""

    def __init__(self, doc: Document):
        self.doc_id = doc.doc_id
        self.norm, self.index_map = normalize_with_map(doc.full_text)

    def locate(
        self, span: str, span_index: int = 0, threshold: float = FUZZY_THRESHOLD
    ) -> SpanAlignment | None:
        needle = normalize(span)
        if not needle:
            return None
        pos = self.norm.find(needle)
        if pos != -1:
            start, end = pos, pos + len(needle)
            score, method, n = 100.0, "exact", self.norm.count(needle)
        else:
            al = fuzz.partial_ratio_alignment(needle, self.norm, score_cutoff=threshold)
            if al is None or al.dest_end <= al.dest_start:
                return None
            start, end = al.dest_start, al.dest_end
            score, method, n = al.score, "fuzzy", 1
        return SpanAlignment(
            span_index=span_index,
            char_start=self.index_map[start],
            char_end=self.index_map[end - 1] + 1,
            score=score,
            method=method,
            n_occurrences=n,
        )


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def is_relevant(chunk: Chunk, al: SpanAlignment, min_frac: float = MIN_OVERLAP_FRAC) -> bool:
    """Relevant if the overlap covers at least `min_frac` of the smaller of (span, chunk).

    A short span inside a big chunk is relevant. A long span split over several chunks makes each chunk
    that sits mostly inside it relevant. A chunk that only grazes the edge of the span is not.
    """
    ov = overlap(chunk.char_start, chunk.char_end, al.char_start, al.char_end)
    smaller = min(chunk.char_end - chunk.char_start, al.char_end - al.char_start)
    return ov > 0 and ov >= min_frac * smaller


def label_question(
    question: EvalQuestion,
    doc_text: DocText | None,
    doc_chunks: Iterable[Chunk],
    strategy: ChunkStrategy,
) -> QuestionLabels:
    labels = QuestionLabels(
        qid=question.qid, doc_id=question.doc_id, strategy=strategy, answerable=question.answerable
    )
    if not question.answerable:
        return labels
    if doc_text is None:
        labels.unaligned_spans = list(range(len(question.evidence_spans)))
        return labels
    doc_chunks = list(doc_chunks)
    for i, span in enumerate(question.evidence_spans):
        al = doc_text.locate(span, i)
        if al is None:
            labels.unaligned_spans.append(i)
            continue
        labels.alignments.append(al)
        labels.span_chunks.append([c.chunk_id for c in doc_chunks if is_relevant(c, al)])
    return labels


def build_labels(
    questions: Iterable[EvalQuestion],
    documents: dict[str, Document],
    chunks: Iterable[Chunk],
    strategy: ChunkStrategy,
) -> list[QuestionLabels]:
    chunks_by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        chunks_by_doc[c.doc_id].append(c)
    doc_texts: dict[str, DocText] = {}
    out = []
    for q in questions:
        doc_text = None
        if q.doc_id is not None and q.doc_id in documents:
            if q.doc_id not in doc_texts:
                doc_texts[q.doc_id] = DocText(documents[q.doc_id])
            doc_text = doc_texts[q.doc_id]
        out.append(label_question(q, doc_text, chunks_by_doc.get(q.doc_id or "", []), strategy))
    return out


def alignment_report(labels: Iterable[QuestionLabels]) -> dict[str, float]:
    """How trustworthy are the labels? Report this with every result table."""
    counts: Counter[str] = Counter()
    for lab in labels:
        counts["questions"] += 1
        counts["unaligned_spans"] += len(lab.unaligned_spans)
        counts["uncovered_spans"] += sum(1 for g in lab.span_chunks if not g)
        for al in lab.alignments:
            counts[f"{al.method}_spans"] += 1
            counts["ambiguous_spans"] += al.n_occurrences > 1
    total = counts["exact_spans"] + counts["fuzzy_spans"] + counts["unaligned_spans"]
    report = dict(counts)
    report["aligned_rate"] = (total - counts["unaligned_spans"]) / total if total else 0.0
    return report
