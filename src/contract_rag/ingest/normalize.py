"""Text normalization helpers for layout-aware PDF parsing."""

from __future__ import annotations

import re
import unicodedata

_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_HYPHENATED_LINE = re.compile(r"(?P<prefix>\b[\w]{2,})-$")


def normalize_inline(text: str) -> str:
    """Normalize Unicode and horizontal whitespace without retaining line breaks."""
    normalized = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    return _HORIZONTAL_SPACE.sub(" ", normalized.replace("\n", " ")).strip()


def normalize_lines(lines: list[str]) -> str:
    """Join PDF lines and repair likely word-wrap hyphenation."""
    cleaned = [normalize_inline(line) for line in lines]
    cleaned = [line for line in cleaned if line]
    if not cleaned:
        return ""

    result = cleaned[0]
    for line in cleaned[1:]:
        match = _HYPHENATED_LINE.search(result)
        if match and line[0].islower():
            result = result[:-1] + line
        else:
            result += " " + line
    return result.strip()


def normalize_markdown(text: str) -> str:
    """Normalize table Markdown while retaining its row structure."""
    lines = [normalize_inline(line) for line in unicodedata.normalize("NFKC", text).splitlines()]
    return _BLANK_LINES.sub("\n\n", "\n".join(line for line in lines if line)).strip()


def whitespace_equivalent(left: str, right: str) -> bool:
    """Return whether two strings differ only in whitespace."""
    return " ".join(left.split()) == " ".join(right.split())
