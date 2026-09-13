import json
from types import SimpleNamespace

import pytest

from contract_rag.eval.agreement import agreement_report, cohen_kappa, judge_labels, kappa_band
from contract_rag.eval.judge import (
    JudgeConfig,
    OllamaJudge,
    Statement,
    Verdict,
    citation_validity,
    faithfulness_score,
    score_answer,
    split_statements,
    unsupported_numbers,
)
from contract_rag.schemas import Chunk, EvalQuestion, EvalScores, GenerationResult

CHUNKS = [
    Chunk(
        chunk_id=f"d1::section::{i:05d}",
        doc_id="d1",
        strategy="section",
        text=text,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(text),
        token_count=0,
    )
    for i, text in enumerate(["Governed by Delaware law.", "Terminable on ninety (90) days' notice."])
]


def question(answerable: bool = True) -> EvalQuestion:
    return EvalQuestion(
        qid="q1",
        question="What notice is needed to terminate?",
        doc_id="d1",
        qtype="cuad_derived" if answerable else "unanswerable",
        category="Termination For Convenience",
        reference_answer="90 days" if answerable else "",
        evidence_spans=["Terminable on ninety (90) days' notice."] if answerable else [],
        answerable=answerable,
        split="dev",
    )


def generation(answer: str, cited: list[str], abstained: bool = False, invalid: int = 0) -> GenerationResult:
    return GenerationResult(
        qid="q1",
        config_id="cfg",
        model="qwen3.5:4b",
        answer=answer,
        cited_chunk_ids=cited,
        abstained=abstained,
        invalid_citations=invalid,
    )


class QueueClient:
    """Returns the queued JSON payloads in order and records every request."""

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        content = self.payloads.pop(0)
        text = content if isinstance(content, str) else json.dumps(content)
        return SimpleNamespace(message=SimpleNamespace(content=text))


def verdicts(*flags):
    return {
        "verdicts": [
            {"id": i, "supported": s, "on_topic": t, "reason": ""} for i, (s, t) in enumerate(flags, 1)
        ]
    }


def test_split_statements_reads_citations_from_the_answer_text():
    parts = split_statements(
        "Ninety days' notice is needed [2]. Delaware law governs [1, 2]. It renews yearly."
    )
    assert [(p.text, p.cited) for p in parts] == [
        ("Ninety days' notice is needed", [2]),
        ("Delaware law governs", [1, 2]),
        ("It renews yearly", []),
    ]
    assert split_statements("Ninety days' notice. [2]")[0].cited == [2]  # stray citation joins its sentence


def test_split_statements_separates_clauses_that_cite_different_excerpts():
    answer = (
        "Names must be assigned within thirty (30) days [3], software returned within a reasonable time [4], "
        "and leases assigned on demand [6][8]. It also survives termination."
    )
    parts = split_statements(answer)
    assert [(p.text, p.cited) for p in parts] == [
        ("Names must be assigned within thirty (30) days", [3]),
        ("software returned within a reasonable time", [4]),
        ("leases assigned on demand", [6, 8]),
        ("It also survives termination", []),
    ]
    # an uncited sentence before a cited one stays uncited
    parts = split_statements("Delaware law applies. Notice is ninety days [2].")
    assert [(p.text, p.cited) for p in parts] == [
        ("Delaware law applies", []),
        ("Notice is ninety days", [2]),
    ]


def test_faithfulness_guard_ignores_judge_support_without_real_citations():
    statements = [
        Statement(text="a", cited=[2]),
        Statement(text="b", cited=[]),
        Statement(text="c", cited=[9]),
    ]
    judged = [Verdict(id=i, supported=True) for i in (1, 2, 3)]  # the judge says all three are supported
    assert faithfulness_score(statements, judged, CHUNKS) == pytest.approx(1 / 3)
    assert faithfulness_score([], [], CHUNKS) is None


def test_numbers_must_appear_in_the_cited_excerpt():
    excerpt = ["Terminable on ninety (90) days' notice under Section 12.7."]
    assert unsupported_numbers("Ninety days' notice is required", excerpt) == []  # no digits to check
    assert unsupported_numbers("90 days' notice under Section 12.7", excerpt) == []
    assert unsupported_numbers("60 days' notice as set out in Article 5.1", excerpt) == ["5.1", "60"]
    assert unsupported_numbers("One party may give 90 days' notice", excerpt) == []  # "one" is not a quantity
    assert unsupported_numbers("a $45,420.00 minimum", ["a minimum of $45,420 worth"]) == []
    assert unsupported_numbers("30 days", ["within thirty days"]) == []  # number words count in the excerpt
    statement, yes = Statement(text="Notice is 60 days", cited=[2]), Verdict(id=1, supported=True)
    assert (
        faithfulness_score([statement], [yes], CHUNKS) == 0.0
    )  # the judge said yes; the number check says no


def test_citation_validity():
    assert citation_validity(generation("x [1]", ["d1::section::00000"], invalid=1)) == 0.5
    assert citation_validity(generation("no citations", [])) == 0.0
    assert citation_validity(generation("", [], abstained=True)) is None


def test_score_answer_shows_each_statement_only_its_cited_excerpts():
    client = QueueClient(
        verdicts((True, True), (True, False)), {"correctness": "Correct", "reason": "matches"}
    )
    judge = OllamaJudge(JudgeConfig(), client=client)
    answer = "Ninety days' notice is needed [2]. Delaware law governs [1]."
    s = score_answer(
        question(), generation(answer, ["d1::section::00001", "d1::section::00000"]), CHUNKS, judge
    )
    assert (s.faithfulness, s.answer_relevance, s.answer_correctness) == (1.0, 0.5, 1.0)
    assert s.abstention_correct is True and s.citation_validity == 1.0

    prompt = client.calls[0]["messages"][1]["content"]
    first, second = prompt.split("---")
    assert "ninety (90)" in first and "Delaware" not in first  # statement 1 sees only excerpt [2]
    assert "Delaware" in second and "ninety (90)" not in second
    assert client.calls[0]["think"] is False and "[2]" not in client.calls[1]["messages"][1]["content"]
    assert json.loads(s.judge_rationale)["grade"]["correctness"] == "correct"


def test_the_judge_sees_the_sentence_window_the_generator_saw():
    window = Chunk(
        chunk_id="d1::sentence_window::00007",
        doc_id="d1",
        strategy="sentence_window",
        text="Either party may terminate.",
        context_text="Either party may terminate. Notice must be given ninety (90) days in advance.",
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=27,
        token_count=0,
    )
    client = QueueClient(verdicts((True, True)), {"correctness": "correct", "reason": ""})
    judge = OllamaJudge(JudgeConfig(), client=client)
    s = score_answer(question(), generation("Notice is 90 days [1].", [window.chunk_id]), [window], judge)
    assert "ninety (90) days in advance" in client.calls[0]["messages"][1]["content"]
    assert s.faithfulness == 1.0  # the number check reads the window too, not only the matched sentence


def test_uncited_answer_scores_zero_even_if_the_judge_approves():
    client = QueueClient(verdicts((True, True)), {"correctness": "correct", "reason": ""})
    s = score_answer(
        question(), generation("The warranty lasts 3 years.", []), CHUNKS, OllamaJudge(client=client)
    )
    assert s.faithfulness == 0.0 and s.citation_validity == 0.0
    assert "(no excerpts cited)" in client.calls[0]["messages"][1]["content"]


def test_abstention_paths_skip_the_llm():
    judge = OllamaJudge(JudgeConfig(), client=QueueClient())
    refused = score_answer(question(True), generation("", [], abstained=True), CHUNKS, judge)
    assert refused.abstention_correct is False and refused.answer_correctness == 0.0
    correct_refusal = score_answer(question(False), generation("", [], abstained=True), CHUNKS, judge)
    assert correct_refusal.abstention_correct is True and correct_refusal.answer_correctness is None
    assert judge.calls == 0


def test_missing_or_unparseable_judge_output_leaves_scores_empty():
    missing = QueueClient(verdicts((True, True)), "not json")  # 2 statements, only 1 verdict
    s = score_answer(
        question(), generation("A [1]. B [2].", ["d1::section::00000"]), CHUNKS, OllamaJudge(client=missing)
    )
    assert s.faithfulness is None and s.answer_relevance is None and s.answer_correctness is None


def test_cohen_kappa_and_bands():
    assert cohen_kappa(["y", "y", "n", "n"], ["y", "n", "n", "n"]) == pytest.approx(0.5)
    assert cohen_kappa(["y", "n"], ["y", "n"]) == 1.0
    assert kappa_band(0.73) == "substantial"
    assert kappa_band(0.1) == "slight"
    assert kappa_band(0.9) == "almost perfect"


def test_agreement_report_against_reference_labels():
    judged = {
        "q1": EvalScores(qid="q1", config_id="c", faithfulness=1.0, answer_correctness=1.0),
        "q2": EvalScores(qid="q2", config_id="c", faithfulness=0.5, answer_correctness=0.5),
        "q3": EvalScores(qid="q3", config_id="c", faithfulness=1.0, answer_correctness=0.0),
    }
    labels = {
        "q1": {"faithful": "yes", "correct": "yes"},
        "q2": {"faithful": "no", "correct": "no"},
        "q3": {"faithful": "yes", "correct": "no"},
    }
    assert judge_labels(judged["q2"]) == {"faithful": "no", "correct": "partial"}
    report = agreement_report(labels, judged)
    assert report["faithful"]["agreement"] == 1.0 and report["faithful"]["kappa"] == 1.0
    assert report["correct"]["n"] == 3 and report["correct"]["agreement"] == pytest.approx(0.667, abs=1e-3)
