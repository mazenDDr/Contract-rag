from contract_rag.ingest.normalize import normalize_inline, normalize_lines, whitespace_equivalent


def test_normalize_lines_repairs_wrapped_hyphenation() -> None:
    assert normalize_lines(["The counter-", "party shall pay."]) == "The counterparty shall pay."
    assert normalize_lines(["A non-", "Disclosure heading"]) == "A non- Disclosure heading"


def test_whitespace_equivalent_ignores_only_whitespace() -> None:
    assert normalize_inline("  Some\u00a0text\n") == "Some text"
    assert whitespace_equivalent("a\n b", "a b")
    assert not whitespace_equivalent("a-b", "ab")
