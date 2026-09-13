import json
from types import SimpleNamespace

from contract_rag.generation.generator import GeneratorConfig, OllamaGenerator, parse_output
from contract_rag.generation.prompts import ABSTAIN_ANSWER, build_messages, format_context, parse_citations
from contract_rag.schemas import Chunk


def make_chunk(i: int, text: str, section: list[str] | None = None, pages=(1, 1)) -> Chunk:
    return Chunk(
        chunk_id=f"d1::section::{i:05d}",
        doc_id="d1",
        strategy="section",
        text=text,
        section_path=section or [],
        page_start=pages[0],
        page_end=pages[1],
        char_start=0,
        char_end=len(text),
        token_count=0,
    )


CHUNKS = [
    make_chunk(0, "This Agreement is governed by the laws of Delaware.", ["15. GOVERNING LAW"]),
    make_chunk(1, "Either party may terminate on 90 days' notice.", ["12. TERMINATION"], pages=(4, 5)),
    make_chunk(2, "Licensor's liability is capped at fees paid."),
]


class FakeClient:
    def __init__(self, content: str):
        self.content = content
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            message=SimpleNamespace(content=self.content), prompt_eval_count=321, eval_count=17
        )


def test_context_is_numbered_with_section_and_pages():
    ctx = format_context(CHUNKS)
    assert ctx.startswith("[1] (15. GOVERNING LAW, p.1)\nThis Agreement")
    assert "[2] (12. TERMINATION, pp.4-5)" in ctx
    assert "[3] (no section, p.1)" in ctx
    messages = build_messages("What law governs?", CHUNKS)
    assert messages[0]["role"] == "system" and messages[1]["content"].endswith("Question: What law governs?")


def test_parse_citations_maps_aliases_and_counts_invalid():
    cited, invalid = parse_citations("Delaware [1]. Terminable [2][9]. Also [1, 3].", CHUNKS)
    assert cited == ["d1::section::00000", "d1::section::00001", "d1::section::00002"]
    assert invalid == 1


def test_parse_output_handles_json_abstention_and_garbage():
    assert parse_output('{"answer": "Delaware [1].", "abstained": false}') == ("Delaware [1].", False)
    assert parse_output('{"answer": "", "abstained": true}') == (ABSTAIN_ANSWER, True)
    assert parse_output("not json [1]") == ("not json [1]", False)


def test_generator_builds_request_and_result():
    client = FakeClient(json.dumps({"answer": "Delaware law governs [1].", "abstained": False}))
    gen = OllamaGenerator(GeneratorConfig(model="qwen3.5:4b"), client=client)
    result = gen.generate("What law governs?", CHUNKS, qid="q1", config_id="cfg")

    call = client.calls[0]
    assert call["model"] == "qwen3.5:4b" and call["think"] is False
    assert call["format"]["required"] == ["answer", "abstained"]
    assert call["options"]["temperature"] == 0.0 and call["options"]["num_ctx"] == 8192

    assert result.answer == "Delaware law governs [1]."
    assert result.cited_chunk_ids == ["d1::section::00000"]
    assert not result.abstained and result.invalid_citations == 0
    assert (result.prompt_tokens, result.completion_tokens, result.cost_usd) == (321, 17, 0.0)


def test_think_flag_omitted_when_none():
    client = FakeClient('{"answer": "", "abstained": true}')
    result = OllamaGenerator(GeneratorConfig(think=None), client=client).generate("?", CHUNKS, "q2", "cfg")
    assert "think" not in client.calls[0]
    assert result.abstained and result.cited_chunk_ids == []


class TruncatingClient(FakeClient):
    def chat(self, **kwargs):
        response = super().chat(**kwargs)
        response.done_reason = "length"
        return response


def test_thinking_gets_bigger_budget_and_truncation_is_flagged():
    client = TruncatingClient("")
    result = OllamaGenerator(GeneratorConfig(think=True), client=client).generate("?", CHUNKS, "q3", "cfg")
    assert client.calls[0]["options"]["num_predict"] == 2048
    assert result.truncated and result.answer == "" and not result.abstained
