import json
from types import SimpleNamespace

import pytest

from contract_rag.eval.agreement import agreement_report, cohen_kappa, judge_labels, kappa_band
from contract_rag.eval.judge import (
    Claim,
    JudgeConfig,
    OllamaJudge,
    citation_validity,
    faithfulness_score,
    score_answer,
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
        return SimpleNamespace(
            message=SimpleNamespace(content=content if isinstance(content, str) else json.dumps(content))
        )


def test_faithfulness_requires_real_citations():
    claims = [
        Claim(claim="90 days notice", cited=[2], supported=True),
        Claim(claim="Delaware law", cited=[], supported=True),  # judge says yes, but it cites nothing
        Claim(claim="made up", cited=[9], supported=True),  # cites an excerpt that doesn't exist
        Claim(claim="wrong", cited=[1], supported=False),
    ]
    assert faithfulness_score(claims, n_chunks=2) == 0.25
    assert faithfulness_score([], 2) is None


def test_citation_validity():
    assert citation_validity(generation("x [1]", ["d1::section::00000"], invalid=1)) == 0.5
    assert citation_validity(generation("no citations", [])) == 0.0
    assert citation_validity(generation("", [], abstained=True)) is None


def test_score_answer_normal_path():
    client = QueueClient(
        {"claims": [{"claim": "90 days", "cited": [2], "supported": True, "reason": "stated in [2]"}]},
        {"correctness": "Correct", "relevance": 4, "reason": "matches"},
    )
    judge = OllamaJudge(JudgeConfig(), client=client)
    s = score_answer(
        question(), generation("Ninety days' notice [2].", ["d1::section::00001"]), CHUNKS, judge
    )
    assert (s.faithfulness, s.answer_correctness, s.answer_relevance) == (1.0, 1.0, 0.75)
    assert s.abstention_correct is True and s.citation_validity == 1.0
    assert json.loads(s.judge_rationale)["grade"]["correctness"] == "correct"
    assert client.calls[0]["format"]["required"] == ["claims"] and client.calls[0]["think"] is False
    assert "[2] (no section, p.1)" in client.calls[0]["messages"][1]["content"]


def test_abstention_paths_skip_the_llm():
    judge = OllamaJudge(JudgeConfig(), client=QueueClient())
    refused = score_answer(question(True), generation("", [], abstained=True), CHUNKS, judge)
    assert refused.abstention_correct is False and refused.answer_correctness == 0.0
    correct_refusal = score_answer(question(False), generation("", [], abstained=True), CHUNKS, judge)
    assert correct_refusal.abstention_correct is True and correct_refusal.answer_correctness is None
    assert judge.calls == 0


def test_unanswerable_but_answered_is_flagged_and_checked_for_faithfulness():
    client = QueueClient(
        {"claims": [{"claim": "royalty 5%", "cited": [], "supported": False, "reason": "none"}]}
    )
    s = score_answer(
        question(False), generation("The royalty is 5%.", []), CHUNKS, OllamaJudge(client=client)
    )
    assert s.abstention_correct is False and s.faithfulness == 0.0 and s.answer_correctness is None


def test_unparseable_judge_output_leaves_scores_empty():
    client = QueueClient("not json", {"correctness": "maybe", "relevance": 9, "reason": ""})
    s = score_answer(
        question(), generation("Ninety days [2].", ["d1::section::00001"]), CHUNKS, OllamaJudge(client=client)
    )
    assert s.faithfulness is None and s.answer_correctness is None


def test_cohen_kappa_and_bands():
    assert cohen_kappa(["y", "y", "n", "n"], ["y", "n", "n", "n"]) == pytest.approx(0.5)
    assert cohen_kappa(["y", "n"], ["y", "n"]) == 1.0
    assert (
        kappa_band(0.73) == "substantial"
        and kappa_band(0.1) == "slight"
        and kappa_band(0.9) == "almost perfect"
    )


def test_agreement_report_against_human_labels():
    judged = {
        "q1": EvalScores(qid="q1", config_id="c", faithfulness=1.0, answer_correctness=1.0),
        "q2": EvalScores(qid="q2", config_id="c", faithfulness=0.5, answer_correctness=0.5),
        "q3": EvalScores(qid="q3", config_id="c", faithfulness=1.0, answer_correctness=0.0),
    }
    human = {
        "q1": {"faithful": "yes", "correct": "yes"},
        "q2": {"faithful": "no", "correct": "no"},
        "q3": {"faithful": "yes", "correct": "no"},
    }
    assert judge_labels(judged["q2"]) == {"faithful": "no", "correct": "partial"}
    report = agreement_report(human, judged)
    assert report["faithful"]["agreement"] == 1.0 and report["faithful"]["kappa"] == 1.0
    assert report["correct"]["n"] == 3 and report["correct"]["agreement"] == pytest.approx(0.667, abs=1e-3)
