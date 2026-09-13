from contract_rag.eval.build_questions import (
    _AMOUNT,
    _REDACTED,
    agreement_name,
    build_questions,
    load_contracts,
    parse_parties,
    short_party,
    split_documents,
    title_case,
)
from contract_rag.eval.labels import DocText
from contract_rag.schemas import Document

FILLER = "The parties shall cooperate in good faith on all matters described in this Agreement. " * 25
GOV = "This Agreement is governed by the laws of the State of Delaware."
TERM = "The initial term of this Agreement ends on December 31, 2025."
RENEW = "Thereafter this Agreement renews automatically for successive one (1) year terms."
NOTICE = "Either party may prevent renewal by giving ninety (90) days prior written notice."


def _qa(category: str, texts: list[str]) -> dict:
    return {
        "question": f'Highlight the parts (if any) of this contract related to "{category}".',
        "answers": [{"text": t, "answer_start": 0} for t in texts],
    }


def _corpus(n: int = 6):
    cuad, rows, manifest, documents = {"data": []}, [], [], {}
    for i in range(n):
        title = f"ACME{i}_2020_EX-10.1_Supply Agreement"
        doc_id = f"acme{i}-supply-agreement"
        full_text = "\n\n".join([GOV, TERM, FILLER, RENEW, NOTICE])
        documents[doc_id] = Document(
            doc_id=doc_id, title=title, source_path="x.pdf", num_pages=3, full_text=full_text
        )
        manifest.append({"doc_id": doc_id, "title": title, "contract_type": "Supply Agreement"})
        qas = [
            _qa("Governing Law", [GOV]),
            _qa("Expiration Date", [TERM]),
            _qa("Renewal Term", [RENEW]),
            _qa("Notice Period To Terminate Renewal", [NOTICE, "[●]"]),
            _qa("Non-Compete", []),
        ]
        cuad["data"].append({"title": title, "paragraphs": [{"context": "", "qas": qas}]})
        rows.append(
            {
                "Filename": title + ".pdf",
                "Document Name-Answer": "SUPPLY AGREEMENT",
                "Parties-Answer": f'Acme {i}, Inc. ("Acme"); Beta LLC ("Beta")',
                "Governing Law-Answer": "Delaware",
                "Non-Compete-Answer": "No",
            }
        )
    return cuad, rows, manifest, documents


QUOTAS = {
    "dev": {"multi_span": 1, "numeric": 1, "unanswerable": 1, "cuad_derived": 1},
    "test": {"multi_span": 2, "numeric": 2, "unanswerable": 2, "cuad_derived": 2},
}


def test_name_helpers():
    assert title_case("SUPPLY AND DISTRIBUTION AGREEMENT") == "Supply and Distribution Agreement"
    assert title_case("Supply Agreement") == "Supply Agreement"
    parties = parse_parties('Acme, Inc. ("Acme"); Beta LLC ("Beta"); acme, inc.')
    assert parties == ["Acme, Inc.", "Beta LLC"]
    assert (
        agreement_name("THE SUPPLY AGREEMENT", parties, "x")
        == "the Supply Agreement between Acme, Inc. and Beta LLC"
    )
    assert agreement_name("", [], "Supply Agreement") == "the Supply Agreement"
    assert agreement_name("STRATEGIC ALLIANCE AGREEMENT, d", [], "x") == "the Strategic Alliance Agreement"
    long_name = "Mellon Investor Services LLC operating with the service name BNY Mellon Shareowner Services"
    assert short_party(long_name) == "Mellon Investor Services LLC"
    assert short_party("IMPCO Technologies Inc. including its successors and permitted assigns") == (
        "IMPCO Technologies Inc."
    )
    assert agreement_name("Manufacturing, Design and Marketing Agreement (t", [], "x") == (
        "the Manufacturing, Design and Marketing Agreement"
    )


def test_numeric_detection_needs_real_amounts():
    for text in ("within sixty (60) days", "a fee of $45,420.00", "a two-year renewal", "5% of net sales"):
        assert _AMOUNT.search(text), text
    assert _AMOUNT.search("expires on December 31, 2025")
    assert not _AMOUNT.search("subject to Section 2.9.2 and Section 9.3")
    assert _REDACTED.search("liability shall not exceed [***].")


def test_load_contracts_aligns_and_drops_placeholders():
    contracts = load_contracts(*_corpus(2))
    c = contracts[0]
    assert c.name.startswith("the Supply Agreement between Acme")
    assert c.spans["Notice Period To Terminate Renewal"] == [NOTICE]
    assert c.dropped_spans == 1  # the "[●]" redaction placeholder
    assert "Non-Compete" not in c.present
    assert c.positions["Renewal Term"][0] - c.positions["Expiration Date"][0] > 1500


def test_split_is_by_document_and_deterministic():
    ids = [f"d{i}" for i in range(10)]
    first, second = split_documents(ids, 0.3, 13), split_documents(ids, 0.3, 13)
    assert first == second
    assert sum(v == "dev" for v in first.values()) == 3


def test_build_questions_meets_quotas_and_invariants():
    cuad, rows, manifest, documents = _corpus(6)
    contracts = load_contracts(cuad, rows, manifest, documents)
    questions, report = build_questions(contracts, QUOTAS, seed=13, dev_fraction=0.34, per_doc_cap=2)
    again, _ = build_questions(load_contracts(cuad, rows, manifest, documents), QUOTAS, 13, 0.34, 2)

    assert [q.model_dump() for q in questions] == [q.model_dump() for q in again]
    assert report["shortfalls"] == {}
    assert len({q.qid for q in questions}) == len(questions) == 12
    dev_docs = {q.doc_id for q in questions if q.split == "dev"}
    test_docs = {q.doc_id for q in questions if q.split == "test"}
    assert not dev_docs & test_docs  # no contract appears in both splits

    for q in questions:
        if q.qtype == "unanswerable":
            assert not q.answerable and q.evidence_spans == [] and q.reference_answer == ""
        else:
            assert q.answerable and q.evidence_spans
            text = DocText(documents[q.doc_id])
            assert all(text.locate(span) is not None for span in q.evidence_spans)
    multi = next(q for q in questions if q.qtype == "multi_span")
    assert multi.category == "Expiration Date + Renewal Term" and len(multi.evidence_spans) == 2
    governing = [q for q in questions if q.category == "Governing Law"]
    assert all(q.reference_answer == "Delaware" for q in governing)


def test_replacement_keeps_the_qid_and_every_other_question():
    cuad, rows, manifest, documents = _corpus(6)
    base, _ = build_questions(
        load_contracts(cuad, rows, manifest, documents), QUOTAS, 13, 0.34, per_doc_cap=3
    )
    target = next(q for q in base if q.split == "test" and q.qtype == "cuad_derived")
    fixed, report = build_questions(
        load_contracts(cuad, rows, manifest, documents),
        QUOTAS,
        13,
        0.34,
        per_doc_cap=3,
        replace={target.qid: "evidence does not answer the question"},
    )
    new = next(q for q in fixed if q.qid == target.qid)
    assert (new.split, new.qtype) == (target.split, target.qtype)
    assert (new.doc_id, new.category) != (target.doc_id, target.category)
    assert new.notes == "replacement after review" and target.qid in report["replaced"]
    others = [q.model_dump() for q in fixed if q.qid != target.qid]
    assert others == [q.model_dump() for q in base if q.qid != target.qid]
