import random

from contract_rag.eval.calibration import label_item, oracle_chunks
from contract_rag.schemas import Block, Document, EvalQuestion, GenerationResult

PARAS = [
    "12.2 Either party may terminate this Agreement for convenience on ninety (90) days' written notice.",
    "The Distributor shall keep complete and accurate records of all sales made under this Agreement.",
    "All notices under this Agreement shall be in writing and delivered by courier or certified mail.",
    "This Agreement is governed by the laws of the State of Delaware, without regard to conflicts rules.",
    "Neither party shall be liable for indirect, incidental or consequential damages of any kind whatsoever.",
    "The Distributor shall maintain commercial general liability insurance of at least one million dollars.",
]
TEXT = "\n\n".join(PARAS)


def _fixture():
    doc = Document(doc_id="d1", title="t", source_path="x.pdf", num_pages=1, full_text=TEXT)
    blocks, pos = [], 0
    for i, p in enumerate(PARAS):
        blocks.append(
            Block(
                doc_id="d1",
                block_id=f"d1::b{i:05d}",
                block_type="paragraph",
                text=p,
                page=1,
                char_start=pos,
                char_end=pos + len(p),
            )
        )
        pos += len(p) + 2
    return doc, blocks


def _question(spans, answerable=True):
    return EvalQuestion(
        qid="q1",
        question="?",
        doc_id="d1",
        qtype="cuad_derived" if answerable else "unanswerable",
        category="x",
        reference_answer="",
        evidence_spans=spans,
        answerable=answerable,
        split="dev",
    )


def test_oracle_context_contains_evidence_plus_distractors():
    doc, blocks = _fixture()
    q = _question(["terminate this Agreement for convenience on ninety (90) days' written notice"])
    chunks = oracle_chunks(q, doc, blocks, random.Random(1), n_distractors=3)
    assert len(chunks) == 4
    assert sum("ninety (90) days" in c.text for c in chunks) == 1
    assert [c.chunk_id for c in chunks] == [f"d1::oracle::{i:05d}" for i in range(4)]
    again = oracle_chunks(q, doc, blocks, random.Random(1), n_distractors=3)
    assert [c.text for c in chunks] == [c.text for c in again]  # deterministic for a given seed


def test_unanswerable_questions_get_only_distractors():
    doc, blocks = _fixture()
    chunks = oracle_chunks(_question([], answerable=False), doc, blocks, random.Random(1), n_distractors=3)
    assert len(chunks) == 5


def test_label_item_hides_judge_output():
    doc, blocks = _fixture()
    q = _question(["governed by the laws of the State of Delaware"])
    chunks = oracle_chunks(q, doc, blocks, random.Random(2))
    gen = GenerationResult(qid="q1", config_id="c", model="m", answer="Delaware law [1].")
    item = label_item(q, chunks, gen)
    assert set(item) == {
        "qid",
        "qtype",
        "question",
        "reference",
        "evidence",
        "excerpts",
        "answer",
        "abstained",
        "truncated",
    }
    assert item["excerpts"][0]["n"] == 1
