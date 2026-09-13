"""Build the evaluation question set from CUAD annotations.

Every question is scoped to one contract (doc_id) and has one of four types:
- cuad_derived: one clause category the contract contains
- multi_span:   two related categories (or one spread-out category) whose evidence sits far apart,
                so a single chunk rarely holds the whole answer
- numeric:      a category whose evidence contains a number the answer must state
- unanswerable: a category the lawyers marked absent, asked as if it existed (tests abstention)

Contracts, not questions, are split into dev and test, so no contract appears in both splits.
Evidence is kept as verbatim text, and only spans that align to our parsed text are used.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from contract_rag.eval.labels import DocText
from contract_rag.schemas import Document, EvalQuestion, QuestionType

MIN_SPAN_ALNUM = 3  # drops redaction placeholders such as "[●]"
FAR_APART_CHARS = 1500
QTYPE_ORDER: tuple[QuestionType, ...] = ("multi_span", "numeric", "unanswerable", "cuad_derived")

_CATEGORY = re.compile(r'related to "([^"]+)"')
_WORD_NUM = (
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|sixty|ninety)"
)
_UNIT = r"(?:%|(?:percent|per\s*cent|days?|weeks?|months?|years?|hours?)\b)"
_DATE = (
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}"
)
# A real amount, duration or date ("$45,420", "60 days", "ninety (90) days", "two-year", "May 1, 2020"),
# not a section number such as "2.9.2".
_AMOUNT = re.compile(
    rf"\$\s?\d|\b\d[\d,.]*[\s-]*{_UNIT}|\b{_WORD_NUM}\b(?:[\s-]+\w+)?[\s-]*(?:\(\d+\)[\s-]*)?{_UNIT}|{_DATE}",
    re.IGNORECASE,
)
_REDACTED = re.compile(r"\[\s*(?:\*+|●)\s*\]")
_SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with"}

DERIVED = {
    "Governing Law": "Under {name}, which state's or country's law governs the agreement?",
    "Expiration Date": "When does the initial term of {name} end?",
    "Effective Date": "When does {name} take effect?",
    "Renewal Term": "Does {name} renew after its initial term, and if so for how long?",
    "Notice Period To Terminate Renewal": "How much notice must a party give to stop {name} from renewing?",
    "Anti-Assignment": "Can a party assign {name} to a third party without the other party's consent?",
    "Cap On Liability": "Is either party's liability limited under {name}, and to what?",
    "License Grant": "What license, if any, is granted under {name}, and to whom?",
    "Audit Rights": "Does either party have the right to audit the other's books or records under {name}?",
    "Exclusivity": "Does {name} set up any exclusive relationship between the parties?",
    "Insurance": "What insurance must a party maintain under {name}?",
    "Revenue/Profit Sharing": "Does {name} require one party to share revenue or profits with the other?",
    "Minimum Commitment": "Is there a minimum purchase, order or payment commitment under {name}?",
    "Post-Termination Services": "What obligations continue after {name} ends?",
    "Termination For Convenience": "Can a party end {name} early without cause, and on what notice?",
    "Change Of Control": "What happens under {name} if a party is acquired or undergoes a change of control?",
    "Non-Compete": "Does {name} restrict either party from competing, and for how long?",
    "Ip Ownership Assignment": "Under {name}, who owns intellectual property created for the other party?",
    "Warranty Duration": "How long do the warranties under {name} last?",
}

# Two categories whose evidence must sit far apart, or one category whose spans are spread out.
MULTI: dict[tuple[str, ...], str] = {
    (
        "Expiration Date",
        "Renewal Term",
    ): "How long is the initial term of {name}, and what happens when that term ends?",
    ("Termination For Convenience", "Post-Termination Services"): (
        "Can {name} be terminated without cause, and which obligations continue after termination?"
    ),
    ("Cap On Liability", "Uncapped Liability"): (
        "How is liability limited under {name}, and in which cases is it not limited?"
    ),
    ("Anti-Assignment", "Change Of Control"): (
        "Under {name}, what limits apply to assigning the agreement, including when a party changes control?"
    ),
    (
        "License Grant",
        "Non-Transferable License",
    ): "What license does {name} grant, and can the licensee transfer it?",
    (
        "Exclusivity",
        "Non-Compete",
    ): "What exclusivity and non-compete restrictions does {name} place on the parties?",
    ("Minimum Commitment", "Revenue/Profit Sharing"): (
        "What must be purchased or paid under {name}, including any minimums and revenue sharing?"
    ),
    ("Insurance",): "What insurance obligations does {name} impose, including coverage types and amounts?",
    ("Audit Rights",): "What audit or inspection rights does {name} give, and on what conditions?",
    ("Post-Termination Services",): "Which obligations under {name} survive or apply after it ends?",
}

NUMERIC = {
    "Notice Period To Terminate Renewal": "How many days' notice stops {name} from renewing?",
    "Renewal Term": "How long is each renewal period of {name}?",
    "Warranty Duration": "For how long is the warranty under {name} valid?",
    "Cap On Liability": "What is the maximum amount a party can be liable for under {name}?",
    "Minimum Commitment": "What minimum quantity or amount must be ordered or paid under {name}?",
    "Liquidated Damages": "What amount of liquidated damages does {name} specify?",
    "Revenue/Profit Sharing": "What percentage or amount of revenue or profit is shared under {name}?",
    "Insurance": "What minimum insurance coverage amounts does {name} require?",
    "Expiration Date": "On what date does the initial term of {name} expire?",
}

UNANSWERABLE = {
    "Non-Compete": "How long does the non-compete restriction in {name} last after termination?",
    "Liquidated Damages": "What amount of liquidated damages does {name} set for a breach?",
    "Minimum Commitment": "What minimum annual purchase does {name} require?",
    "Most Favored Nation": "What most-favored-customer pricing protection does {name} provide?",
    "Source Code Escrow": "Who acts as the source code escrow agent under {name}?",
    "Revenue/Profit Sharing": "What percentage of profits must be shared under {name}?",
    "Warranty Duration": "How long is the warranty period under {name}?",
    "Rofr/Rofo/Rofn": "What right of first refusal does {name} give, and to whom?",
    "Insurance": "What insurance coverage limits does {name} require?",
    "Non-Disparagement": "What restrictions on disparaging the other party does {name} contain?",
    "No-Solicit Of Employees": "How long does {name} bar a party from hiring the other's employees?",
    "Audit Rights": "How often may a party audit the other's records under {name}?",
}


# ---------- contract loading ----------


@dataclass
class Contract:
    doc_id: str
    title: str
    contract_type: str
    name: str  # e.g. "the Distribution Agreement between A and B"
    spans: dict[str, list[str]] = field(default_factory=dict)  # category -> aligned evidence, in doc order
    positions: dict[str, list[int]] = field(default_factory=dict)  # category -> char_start of each span
    present: set[str] = field(default_factory=set)  # categories with any CUAD evidence, aligned or not
    answers: dict[str, str] = field(default_factory=dict)  # category -> master_clauses.csv answer
    dropped_spans: int = 0


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def title_case(text: str) -> str:
    """Title-case names written mostly in capitals ("SUPPLY AGREEMENT" -> "Supply Agreement")."""
    letters = [c for c in text if c.isalpha()]
    if not letters or sum(c.isupper() for c in letters) / len(letters) < 0.8:
        return text
    words = text.lower().split()
    return " ".join(w if i and w in _SMALL_WORDS else w[:1].upper() + w[1:] for i, w in enumerate(words))


def parse_parties(raw: str) -> list[str]:
    """'Acme, Inc. ("Acme"); Beta LLC ("Beta")' -> ['Acme, Inc.', 'Beta LLC']"""
    parties: list[str] = []
    for part in raw.split(";"):
        part = re.sub(r"\([^)]*\)", "", part)
        part = re.sub(r"\s+", " ", part).strip(" ,;\"'“”")
        if part and part.lower() not in {p.lower() for p in parties}:
            parties.append(part)
    return parties


_PARTY_TAIL = re.compile(
    r"\s+(?:including|together with|and its|operating|doing business|d/b/a|acting|as successor)\b|,\s+an?\s",
    re.IGNORECASE,
)


def short_party(name: str, limit: int = 60) -> str:
    """Trim descriptive tails ("X Inc. including its successors and assigns" -> "X Inc.")."""
    head = _PARTY_TAIL.split(name, maxsplit=1)[0].strip(" ,")
    if len(head) > limit and "," in head:
        head = head.split(",", 1)[0].strip()
    return head


def agreement_name(document_name: str, parties: list[str], fallback: str) -> str:
    base = re.sub(r"\s*\([^)]*$", "", document_name.strip().strip(" .\"'“”"))  # unclosed "(t" fragments
    base = re.sub(r",\s*\w{1,2}$", "", base)  # CSV junk such as ", d"
    base = title_case(base) or fallback
    base = re.sub(r"^(?:the|this)\s+", "", base, flags=re.IGNORECASE)
    parties = [short_party(p) for p in parties]
    name = f"the {base}"
    if len(parties) >= 2:
        name += f" between {parties[0]} and {parties[1]}"
    elif parties:
        name += f" with {parties[0]}"
    return name


def _drop_contained(pairs: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Remove spans that are substrings of another span in the same category."""
    texts = [t for _, t in pairs]
    return [(p, t) for p, t in pairs if not any(t != o and t in o for o in texts)]


def load_contracts(
    cuad: dict[str, Any],
    csv_rows: Iterable[dict[str, str]],
    manifest: Iterable[dict[str, Any]],
    documents: dict[str, Document],
) -> list[Contract]:
    records = {_key(r["title"]): r for r in manifest}
    rows = {_key(Path(r["Filename"]).stem): r for r in csv_rows}
    contracts = []
    for article in cuad["data"]:
        record = records.get(_key(article["title"]))
        if record is None or record["doc_id"] not in documents:
            continue
        row = rows.get(_key(article["title"]), {})
        answers = {
            re.sub(r"\s*-\s*answer$", "", k, flags=re.IGNORECASE).strip(): re.sub(
                r"\s+", " ", v or ""
            ).strip()
            for k, v in row.items()
            if re.search(r"-\s*answer$", k, re.IGNORECASE)
        }
        contract = Contract(
            doc_id=record["doc_id"],
            title=record["title"],
            contract_type=record.get("contract_type", "Agreement"),
            name=agreement_name(
                answers.get("Document Name", ""),
                parse_parties(answers.get("Parties", "")),
                record.get("contract_type", "Agreement"),
            ),
            answers=answers,
        )
        doc_text = DocText(documents[record["doc_id"]])
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                match = _CATEGORY.search(qa["question"])
                if not match:
                    continue
                category = match.group(1)
                # CUAD spans carry the source's line breaks and runs of spaces; collapse them for readability
                # (alignment normalizes whitespace anyway, so this doesn't change which spans are found)
                texts = sorted({re.sub(r"\s+", " ", a["text"]).strip() for a in qa["answers"]} - {""})
                if texts:
                    contract.present.add(category)
                aligned = []
                for text in texts:
                    alignment = (
                        doc_text.locate(text) if sum(c.isalnum() for c in text) >= MIN_SPAN_ALNUM else None
                    )
                    if alignment is None:
                        contract.dropped_spans += 1
                    else:
                        aligned.append((alignment.char_start, text))
                aligned = _drop_contained(sorted(aligned))
                if aligned:
                    contract.positions[category] = [p for p, _ in aligned]
                    contract.spans[category] = [t for _, t in aligned]
        contracts.append(contract)
    return sorted(contracts, key=lambda c: c.doc_id)


# ---------- candidate generation ----------


@dataclass(frozen=True)
class Candidate:
    doc_id: str
    qtype: QuestionType
    key: str  # category, or "A + B" for multi-category questions
    categories: tuple[str, ...]
    question: str
    reference: str
    evidence: tuple[str, ...]
    contract_name: str = ""  # the "{name}" inserted into the question


def _informative(answer: str) -> bool:
    return answer.strip().lower() not in {"", "yes", "no"}


def reference_answer(contract: Contract, categories: Iterable[str]) -> str:
    """The spreadsheet's normalized answer where it has one, otherwise the evidence text itself."""
    categories = list(categories)
    parts = []
    for category in categories:
        answer = contract.answers.get(category, "")
        value = answer if _informative(answer) else " … ".join(contract.spans.get(category, []))
        parts.append(f"{category}: {value}" if len(categories) > 1 else value)
    return "; ".join(parts)[:800]


def _far_apart(first: list[int], second: list[int]) -> bool:
    return min(abs(a - b) for a in first for b in second) > FAR_APART_CHARS


def derived_candidates(c: Contract) -> Iterator[Candidate]:
    for category, template in DERIVED.items():
        if c.spans.get(category):
            yield Candidate(
                c.doc_id,
                "cuad_derived",
                category,
                (category,),
                template.format(name=c.name),
                reference_answer(c, [category]),
                tuple(c.spans[category]),
                c.name,
            )


def multi_candidates(c: Contract) -> Iterator[Candidate]:
    for categories, template in MULTI.items():
        if not all(c.spans.get(cat) for cat in categories):
            continue
        if len(categories) == 1:
            pos = c.positions[categories[0]]
            spread = len(pos) >= 2 and max(pos) - min(pos) > FAR_APART_CHARS
        else:
            spread = _far_apart(c.positions[categories[0]], c.positions[categories[1]])
        if spread:
            yield Candidate(
                c.doc_id,
                "multi_span",
                " + ".join(categories),
                categories,
                template.format(name=c.name),
                reference_answer(c, categories),
                tuple(s for cat in categories for s in c.spans[cat]),
                c.name,
            )


def numeric_candidates(c: Contract) -> Iterator[Candidate]:
    for category, template in NUMERIC.items():
        spans = c.spans.get(category, [])
        if any(_AMOUNT.search(s) and not _REDACTED.search(s) for s in spans):
            yield Candidate(
                c.doc_id,
                "numeric",
                category,
                (category,),
                template.format(name=c.name),
                reference_answer(c, [category]),
                tuple(spans),
                c.name,
            )


def unanswerable_candidates(c: Contract) -> Iterator[Candidate]:
    for category, template in UNANSWERABLE.items():
        absent = category not in c.present and c.answers.get(category, "").strip().lower() in {"", "no"}
        if absent:
            yield Candidate(
                c.doc_id, "unanswerable", category, (category,), template.format(name=c.name), "", (), c.name
            )


GENERATORS: dict[QuestionType, Callable[[Contract], Iterator[Candidate]]] = {
    "cuad_derived": derived_candidates,
    "multi_span": multi_candidates,
    "numeric": numeric_candidates,
    "unanswerable": unanswerable_candidates,
}


# ---------- selection ----------


def split_documents(doc_ids: Iterable[str], dev_fraction: float, seed: int) -> dict[str, str]:
    ids = sorted(doc_ids)
    random.Random(seed).shuffle(ids)
    n_dev = round(len(ids) * dev_fraction)
    return {doc_id: "dev" if i < n_dev else "test" for i, doc_id in enumerate(ids)}


def select(
    candidates: list[Candidate],
    quota: int,
    rng: random.Random,
    used_docs: Counter[str],
    used_pairs: set[tuple[str, str]],
    per_doc_cap: int,
) -> list[Candidate]:
    """Round-robin over categories so no single clause type dominates; respect per-document limits."""
    by_key: dict[str, list[Candidate]] = defaultdict(list)
    for cand in sorted(candidates, key=lambda x: (x.key, x.doc_id)):
        by_key[cand.key].append(cand)
    keys = sorted(by_key)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(by_key[key])
    picked: list[Candidate] = []
    while len(picked) < quota:
        progress = False
        for key in keys:
            if len(picked) >= quota:
                break
            while by_key[key]:
                cand = by_key[key].pop()
                pairs = {(cand.doc_id, cat) for cat in cand.categories}
                if used_docs[cand.doc_id] < per_doc_cap and not pairs & used_pairs:
                    picked.append(cand)
                    used_docs[cand.doc_id] += 1
                    used_pairs.update(pairs)
                    progress = True
                    break
        if not progress:
            break
    return picked


def build_questions(
    contracts: list[Contract],
    quotas: dict[str, dict[str, int]],
    seed: int,
    dev_fraction: float,
    per_doc_cap: int = 2,
    replace: dict[str, str] | None = None,
) -> tuple[list[EvalQuestion], dict[str, Any]]:
    """Select questions, then swap out reviewed-bad ones (`replace`: qid -> reason).

    A replacement keeps the qid, split and question type and comes from the same split, so every other
    question, including its qid, is unchanged; results already keyed by qid stay valid.
    """
    splits = split_documents((c.doc_id for c in contracts), dev_fraction, seed)
    rng = random.Random(seed)
    used_docs: Counter[str] = Counter()
    used_pairs: set[tuple[str, str]] = set()
    chosen: list[tuple[str, Candidate]] = []
    shortfalls = {}
    for split in ("dev", "test"):
        pool = [c for c in contracts if splits[c.doc_id] == split]
        for qtype in QTYPE_ORDER:
            quota = int(quotas[split].get(qtype, 0))
            candidates = [cand for c in pool for cand in GENERATORS[qtype](c)]
            picked = select(candidates, quota, rng, used_docs, used_pairs, per_doc_cap)
            if len(picked) < quota:
                shortfalls[f"{split}/{qtype}"] = quota - len(picked)
            chosen.extend((split, cand) for cand in picked)

    order = {q: i for i, q in enumerate(QTYPE_ORDER)}
    chosen.sort(key=lambda x: (x[0] != "dev", order[x[1].qtype], x[1].doc_id, x[1].key))
    numbered: list[list[Any]] = [
        [f"q{i:04d}", split, cand] for i, (split, cand) in enumerate(chosen, start=1)
    ]

    replaced = {}
    for qid, reason in sorted((replace or {}).items()):
        entry = next((e for e in numbered if e[0] == qid), None)
        if entry is None:
            raise ValueError(f"cannot replace unknown question {qid}")
        _, split, old = entry
        used_docs[old.doc_id] -= 1  # the old (doc, category) pair stays in used_pairs, so it can't come back
        pool = [c for c in contracts if splits[c.doc_id] == split]
        candidates = sorted(
            (
                cand
                for c in pool
                for cand in GENERATORS[old.qtype](c)
                if used_docs[cand.doc_id] < per_doc_cap
                and not {(cand.doc_id, cat) for cat in cand.categories} & used_pairs
            ),
            key=lambda x: (x.key, x.doc_id),
        )
        if not candidates:
            raise ValueError(f"no replacement available for {qid} ({split}/{old.qtype})")
        new = random.Random(f"{seed}:{qid}").choice(candidates)
        used_docs[new.doc_id] += 1
        used_pairs.update((new.doc_id, cat) for cat in new.categories)
        entry[2] = new
        replaced[qid] = {
            "reason": reason,
            "old": f"{old.doc_id} / {old.key}",
            "new": f"{new.doc_id} / {new.key}",
        }

    questions = [
        EvalQuestion(
            qid=qid,
            question=cand.question,
            doc_id=cand.doc_id,
            qtype=cand.qtype,
            category=cand.key,
            reference_answer=cand.reference,
            evidence_spans=list(cand.evidence),
            answerable=cand.qtype != "unanswerable",
            split=split,
            source="cuad",
            notes="replacement after review" if qid in replaced else "",
            contract_name=cand.contract_name or None,
        )
        for qid, split, cand in numbered
    ]
    report = {
        "questions": len(questions),
        "by_split_and_type": {
            s: dict(Counter(q.qtype for q in questions if q.split == s)) for s in ("dev", "test")
        },
        "categories": dict(Counter(q.category for q in questions).most_common()),
        "documents_used": len({q.doc_id for q in questions}),
        "documents_available": {s: sum(v == s for v in splits.values()) for s in ("dev", "test")},
        "shortfalls": shortfalls,
        "replaced": replaced,
        "evidence_spans_dropped_unaligned": sum(c.dropped_spans for c in contracts),
        "seed": seed,
    }
    return questions, report


# ---------- outputs ----------


def review_sheet(questions: list[EvalQuestion], titles: dict[str, str]) -> str:
    lines = [
        "# Evaluation questions: review sheet",
        "",
        "Generated by `scripts/build_questions.py`. For each question check that it is natural and",
        "unambiguous, and that the evidence really answers it (or, for unanswerable ones, that the",
        "contract plausibly lacks the clause). Tick the box, or note a fix.",
    ]
    for split in ("dev", "test"):
        for qtype in QTYPE_ORDER:
            group = [q for q in questions if q.split == split and q.qtype == qtype]
            if not group:
                continue
            lines += ["", f"## {split} · {qtype} ({len(group)})", ""]
            for q in group:
                lines.append(f"- [ ] **{q.qid}** · {q.category} · _{titles.get(q.doc_id or '', q.doc_id)}_")
                lines.append(f"  - **Q:** {q.question}")
                if q.answerable:
                    lines.append(f"  - **Reference:** {q.reference_answer[:300]}")
                    for span in q.evidence_spans[:3]:
                        snippet = re.sub(r"\s+", " ", span)[:220]
                        lines.append(f"  - Evidence: “{snippet}”")
                else:
                    lines.append("  - **Expected:** abstain; the lawyers found no such clause.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    cuad = json.loads(Path(cfg["cuad_json"]).read_text(encoding="utf-8"))
    with open(cfg["master_csv"], encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    manifest = [json.loads(line) for line in Path(cfg["manifest_path"]).read_text().splitlines() if line]
    lines = Path(cfg["documents_path"]).read_text(encoding="utf-8").splitlines()
    documents = {d.doc_id: d for d in (Document.model_validate_json(line) for line in lines if line)}

    contracts = load_contracts(cuad, csv_rows, manifest, documents)
    questions, report = build_questions(
        contracts,
        cfg["quotas"],
        int(cfg["seed"]),
        float(cfg["dev_fraction"]),
        int(cfg["per_doc_cap"]),
        replace=cfg.get("replace") or {},
    )
    out = Path(cfg["questions_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(q.model_dump_json() + "\n" for q in questions), encoding="utf-8")
    titles = {c.doc_id: c.name for c in contracts}
    Path(cfg["review_path"]).write_text(review_sheet(questions, titles), encoding="utf-8")
    Path(cfg["report_path"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
