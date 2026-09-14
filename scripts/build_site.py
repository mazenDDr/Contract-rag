"""Build the project pages from their templates and the runs:
site/template.html -> site/index.html (the one-page tour) and site/guide-template.html -> site/guide.html
(the field guide, which explains every part of the code).

Every number and example on the pages comes from the repository's runs:
- the story section follows one real question (q0078) through parsing, chunking, search, answering and
  judging, using the saved blocks, chunks, per-stage rankings, answer and judge verdicts;
- the exhibits are real answers from the final answer-quality run, each citation shown with the
  sentence of its excerpt that best matches the statement citing it;
- the guide adds every ablation configuration's test scores, the story question's numbered prompt
  excerpts, and one graded answer (q0031) whose statements the judge checked one by one.

Needs data/processed (parsed documents, blocks, chunks) and runs/matrix-v3.
Run: PYTHONPATH=src .venv/bin/python scripts/build_site.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from contract_rag.analysis.failures import content_words
from contract_rag.eval.judge import split_statements
from contract_rag.retrieval.config import load_chunks
from contract_rag.retrieval.pipeline import scoped_query

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/matrix-v3"
BASE = "fixed__bm25__nodense__none__norerank__k8__doc"  # the headline setup
HYBRID = "fixed__bm25__bge-base__w0.4__norerank__k8__doc"  # for its embedding ranking
STORY_QID = "q0078"
EXHIBITS = ["q0078", "q0046", "q0069", "q0060", "q0048"]
JUDGED_QID = "q0031"  # graded correct, yet one of its two citations points at the wrong excerpt
_SENTENCE = re.compile(r"(?<=[.;])\s+(?=[A-Z(\"“])")
_NUMBER = re.compile(r"\d[\d,]*")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def short_id(chunk_id: str) -> str:
    return chunk_id.split("::", 1)[1]


def best_sentence(chunk_text: str, statement: str) -> str:
    """The sentence of an excerpt that shares the most words (and numbers, weighted up) with a statement."""
    sentences = [s.strip() for s in _SENTENCE.split(re.sub(r"\s+", " ", chunk_text)) if len(s.strip()) > 25]
    words, numbers = content_words(statement), set(_NUMBER.findall(statement))
    return max(
        sentences, key=lambda s: len(content_words(s) & words) + 3 * len(set(_NUMBER.findall(s)) & numbers)
    )


def exhibits(questions, generations, scores, chunks) -> list[dict]:
    out = []
    for qid in EXHIBITS:
        record, score, q = generations[qid], scores[qid], questions[qid]
        answer, final = record["generation"]["answer"], record["final"]
        citations = []
        for n in sorted({n for st in split_statements(answer) for n in st.cited}):
            chunk = chunks[final[n - 1]]
            statements = [st.text for st in split_statements(answer) if n in st.cited]
            picked = list(dict.fromkeys(best_sentence(chunk.text, st) for st in statements))
            citations.append(
                {
                    "n": n,
                    "chunk_id": short_id(chunk.chunk_id),
                    "section": " > ".join(chunk.section_path[-2:]),
                    "page": chunk.page_start,
                    "sentence": " … ".join(s[:420] for s in picked),
                }
            )
        out.append(
            {
                "qid": qid,
                "category": q["category"],
                "answerable": q["answerable"],
                "contract": q.get("contract_name"),
                "question": q["question"],
                "answer": answer,
                "correctness": score["answer_correctness"],
                "faithfulness": score["faithfulness"],
                "abstention_correct": score["abstention_correct"],
                "citations": citations,
            }
        )
    return out


def story(questions, generations, scores, all_chunks) -> dict:
    q = questions[STORY_QID]
    doc = next(d for d in read_jsonl(ROOT / "data/processed/documents.jsonl") if d["doc_id"] == q["doc_id"])
    text, evidence = doc["full_text"], q["evidence_spans"][0]
    ev_at = text.find(evidence[:60])
    ev_end = ev_at + len(evidence)
    lo, hi = ev_at - 700, ev_end + 500
    blocks = [b for b in read_jsonl(ROOT / "data/processed/blocks.jsonl") if b["doc_id"] == q["doc_id"]]
    chunks = {}
    for strategy, loaded in all_chunks.items():
        mine = [c for c in loaded if c.doc_id == q["doc_id"]]
        near = [c for c in mine if c.char_end > lo and c.char_start < hi]
        chunks[strategy] = {
            "count": len(mine),
            "near": [
                {
                    "id": short_id(c.chunk_id),
                    "at": [c.char_start, c.char_end],
                    "has_evidence": c.char_start <= ev_at < c.char_end or ev_at <= c.char_start < ev_end,
                }
                for c in near
            ],
        }
    retrieval = {(r["config_id"], r["qid"]): r for r in read_jsonl(RUN / "retrieval.jsonl")}

    def ranked(config: str, stage: str) -> list[dict]:
        rows = retrieval[(config, STORY_QID)]["stages"].get(stage, [])
        return [
            {"id": short_id(r["chunk_id"]), "rank": r["rank"], "score": round(r["score"], 3)} for r in rows
        ]

    record, detail = generations[STORY_QID], json.loads(scores[STORY_QID]["judge_rationale"])
    return {
        "qid": STORY_QID,
        "pages": doc["num_pages"],
        "question": q["question"],
        "contract_name": q["contract_name"],
        "scoped_question": scoped_query(q["question"], q["contract_name"], True),
        "evidence_at": [ev_at, ev_end],
        "window_at": [lo, hi],
        "window": text[lo:hi],
        "blocks": [
            {"type": b["block_type"], "page": b["page"], "at": [b["char_start"], b["char_end"]]}
            for b in blocks
        ],
        "chunks": chunks,
        "bm25": ranked(BASE, "bm25"),
        "dense_bge_base": ranked(HYBRID, "dense"),
        "final": [short_id(c) for c in record["final"]],
        "answer": record["generation"]["answer"],
        "latency_ms": round(record["generation"]["latency_ms"]),
        "statements": detail["statements"],
        "grade": detail["grade"],
    }


def grid() -> list[list]:
    """Every configuration of the retrieval ablation with its test recall@8, MRR and estimated latency."""
    summary = json.loads((ROOT / "runs/ablation-v1/summary.json").read_text(encoding="utf-8"))
    return [
        [
            key,
            cfg["description"],
            round(cfg["test"]["context_recall"]["mean"], 3),
            round(cfg["test"]["mrr"]["mean"], 3),
            cfg["est_latency_ms"],
        ]
        for key, cfg in summary["configs"].items()
    ]


def judged(questions, generations, scores, chunks) -> dict:
    """One graded answer with its statements, verdicts, grade and the excerpts it cites."""
    q, record, score = questions[JUDGED_QID], generations[JUDGED_QID], scores[JUDGED_QID]
    detail, final = json.loads(score["judge_rationale"]), record["final"]
    cited = sorted({n for st in detail["statements"] for n in st["cited"]})
    return {
        "qid": JUDGED_QID,
        "question": q["question"],
        "reference": q["reference_answer"],
        "answer": record["generation"]["answer"],
        "statements": detail["statements"],
        "grade": detail["grade"],
        "excerpts": {
            str(n): {
                "id": short_id(final[n - 1]),
                "page": chunks[final[n - 1]].page_start,
                "text": re.sub(r"\s+", " ", chunks[final[n - 1]].text).strip()[:520],
            }
            for n in cited
        },
        "scores": {
            k: score[k]
            for k in ("faithfulness", "citation_validity", "answer_relevance", "answer_correctness")
        },
    }


def prompt(generations, chunks) -> dict:
    """The excerpts the story question's prompt numbered [1]..[8], with the call's token counts."""
    record = generations[STORY_QID]
    gen = record["generation"]

    def pages(c) -> str:
        return f"p.{c.page_start}" if c.page_start == c.page_end else f"pp.{c.page_start}-{c.page_end}"

    return {
        "excerpts": [
            {
                "n": i,
                "id": short_id(cid),
                "pages": pages(chunks[cid]),
                "section": (chunks[cid].section_path or ["no section"])[-1],
            }
            for i, cid in enumerate(record["final"], start=1)
        ],
        "prompt_tokens": gen["prompt_tokens"],
        "completion_tokens": gen["completion_tokens"],
        "latency_ms": round(gen["latency_ms"]),
    }


def tree() -> list[list]:
    """Every source, test, config and deploy file with its line count, for the guide's repository map."""
    patterns = ("src/contract_rag/**/*.py", "scripts/*.py", "ui/*.py", "tests/*.py", "configs/*.yaml")
    files = [p for pattern in patterns for p in sorted(ROOT.glob(pattern)) if p.name != "__init__.py"]
    files += sorted((ROOT / "deploy/space").iterdir()) + [ROOT / "Dockerfile"]
    return [
        [p.relative_to(ROOT).as_posix(), len(p.read_text(encoding="utf-8").splitlines())]
        for p in files
        if p.is_file()
    ]


def render(template: str, out: str, data: dict) -> None:
    html = (ROOT / template).read_text(encoding="utf-8")
    for marker, value in data.items():
        if marker not in html:
            raise SystemExit(f"placeholder {marker} missing from {template}")
        html = html.replace(marker, json.dumps(value, ensure_ascii=False).replace("</", "<\\/"))
    (ROOT / out).write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html):,} bytes)")


def main() -> None:
    questions = {q["qid"]: q for q in read_jsonl(ROOT / "data/eval/questions.jsonl")}
    generations = {g["qid"]: g for g in read_jsonl(RUN / "generation.jsonl") if g["config_id"] == BASE}
    scores = {s["qid"]: s for s in read_jsonl(RUN / "scores.jsonl") if s["config_id"] == BASE}
    all_chunks = {
        s: load_chunks(ROOT / f"data/processed/chunks/{s}.jsonl")
        for s in ("fixed", "sentence_window", "section")
    }
    fixed = {c.chunk_id: c for c in all_chunks["fixed"]}
    the_story = story(questions, generations, scores, all_chunks)
    render(
        "site/template.html",
        "site/index.html",
        {
            "/*__STORY__*/null": the_story,
            "/*__EXAMPLES__*/null": exhibits(questions, generations, scores, fixed),
        },
    )
    render(
        "site/guide-template.html",
        "site/guide.html",
        {
            "/*__STORY__*/null": the_story,
            "/*__GRID__*/null": grid(),
            "/*__JUDGED__*/null": judged(questions, generations, scores, fixed),
            "/*__PROMPT__*/null": prompt(generations, fixed),
            "/*__TREE__*/null": tree(),
        },
    )


if __name__ == "__main__":
    main()
