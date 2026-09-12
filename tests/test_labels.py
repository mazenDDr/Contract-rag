from contract_rag.eval.labels import (
    DocText,
    SpanAlignment,
    alignment_report,
    build_labels,
    is_relevant,
    label_question,
    normalize,
    normalize_with_map,
)
from contract_rag.schemas import Chunk, Document, EvalQuestion

DOC_TEXT = (
    "1. DEFINITIONS\n“Affiliate” means any entity controlling a Party.\n\n"
    "12. TERMINATION\n12.1 Either party may terminate this Agree-\nment for convenience upon ninety (90) "
    "days’ prior written notice.\n\n"
    "15. GOVERNING LAW\nThis Agreement shall be governed by the laws of the State of Delaware.\n"
)
GOV_LAW = "governed by the laws of the State of Delaware"
TERMINATION = (
    "Either party may terminate this Agreement for convenience upon ninety (90) days' prior written notice."
)


def make_doc() -> Document:
    return Document(doc_id="d1", title="t", source_path="d1.pdf", num_pages=1, full_text=DOC_TEXT)


def chunk(i: int, start: int, end: int, doc_id: str = "d1") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}::fixed::{i:05d}",
        doc_id=doc_id,
        strategy="fixed",
        text=DOC_TEXT[start:end],
        page_start=1,
        page_end=1,
        char_start=start,
        char_end=end,
        token_count=0,
    )


def fixed_chunks(size: int) -> list[Chunk]:
    return [chunk(i, s, min(s + size, len(DOC_TEXT))) for i, s in enumerate(range(0, len(DOC_TEXT), size))]


def question(spans: list[str], qid: str = "q1", doc_id: str | None = "d1", answerable: bool = True):
    return EvalQuestion(
        qid=qid,
        question="?",
        doc_id=doc_id,
        qtype="cuad_derived",
        category="x",
        reference_answer="",
        evidence_spans=spans,
        answerable=answerable,
        split="dev",
    )


def test_normalize_collapses_punctuation_and_case():
    assert normalize("  “Affiliate”  means—ANY entity. ") == "affiliate means any entity"


def test_index_map_points_back_to_source():
    norm, index_map = normalize_with_map("Ab, cd")
    assert norm == "ab cd"
    assert len(index_map) == len(norm)
    assert "Ab, cd"[index_map[0]] == "A" and "Ab, cd"[index_map[-1]] == "d"


def test_exact_alignment_recovers_original_offsets():
    al = DocText(make_doc()).locate(GOV_LAW)
    assert al is not None and al.method == "exact"
    assert DOC_TEXT[al.char_start : al.char_end] == GOV_LAW


def test_fuzzy_alignment_survives_pdf_hyphenation_and_curly_quotes():
    al = DocText(make_doc()).locate(TERMINATION)
    assert al is not None and al.method == "fuzzy" and al.score >= 85
    assert "terminate this Agree" in DOC_TEXT[al.char_start : al.char_end]


def test_unlocatable_span_returns_none():
    assert (
        DocText(make_doc()).locate("Licensee shall pay royalties of five percent quarterly in arrears")
        is None
    )


def test_relevance_rule():
    short_span = SpanAlignment(span_index=0, char_start=40, char_end=60, score=100, method="exact")
    assert is_relevant(chunk(0, 0, 100), short_span)

    long_span = SpanAlignment(span_index=0, char_start=0, char_end=300, score=100, method="exact")
    assert all(is_relevant(chunk(i, s, s + 100), long_span) for i, s in enumerate((0, 100, 200)))
    assert not is_relevant(chunk(3, 290, 400), long_span)  # grazes the edge only


def test_label_question_maps_each_span_to_chunks():
    labels = label_question(question([GOV_LAW, TERMINATION]), DocText(make_doc()), fixed_chunks(60), "fixed")
    assert labels.unaligned_spans == []
    assert len(labels.span_chunks) == 2 and all(labels.span_chunks)
    for al, group in zip(labels.alignments, labels.span_chunks, strict=True):
        for c in fixed_chunks(60):
            if c.chunk_id in group:
                assert c.char_start < al.char_end and al.char_start < c.char_end


def test_unanswerable_question_has_no_evidence():
    labels = label_question(question([], answerable=False), DocText(make_doc()), fixed_chunks(60), "fixed")
    assert not labels.has_evidence and labels.relevant == set()


def test_build_labels_and_report():
    qs = [question([GOV_LAW], qid="q1"), question([GOV_LAW], qid="q2", doc_id="missing")]
    labels = build_labels(qs, {"d1": make_doc()}, fixed_chunks(60), "fixed")
    assert labels[0].has_evidence
    assert labels[1].unaligned_spans == [0]
    report = alignment_report(labels)
    assert report["exact_spans"] == 1 and report["unaligned_spans"] == 1
    assert report["aligned_rate"] == 0.5
