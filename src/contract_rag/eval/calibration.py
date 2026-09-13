"""Judge calibration: answer dev questions from a fixed context, judge them, and compare with human labels.

Retrieval is deliberately taken out of the loop. Each question gets the parsed blocks that contain its
evidence plus distractor blocks from the same contract, so the answers exercise only the generator and judge.
The export for human labelling contains no judge verdicts, so labels are given blind.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from contract_rag.eval.agreement import agreement_report
from contract_rag.eval.judge import JudgeConfig, OllamaJudge, score_answer
from contract_rag.eval.labels import DocText, overlap
from contract_rag.eval.retrieval_metrics import aggregate
from contract_rag.generation.generator import GeneratorConfig, OllamaGenerator
from contract_rag.schemas import Block, Chunk, Document, EvalQuestion, EvalScores, GenerationResult

MAX_CHUNK_CHARS = 2500
MAX_EVIDENCE_BLOCKS = 6


def oracle_chunks(
    question: EvalQuestion, doc: Document, blocks: list[Block], rng: random.Random, n_distractors: int = 3
) -> list[Chunk]:
    """Blocks overlapping the gold evidence, plus random same-contract distractors, in shuffled order."""
    text = DocText(doc)
    located = [al for span in question.evidence_spans if (al := text.locate(span)) is not None]
    evidence = [
        b
        for b in blocks
        if any(overlap(b.char_start, b.char_end, a.char_start, a.char_end) > 0 for a in located)
    ][:MAX_EVIDENCE_BLOCKS]
    chosen = {b.block_id for b in evidence}
    pool = [b for b in blocks if b.block_id not in chosen and b.block_type != "heading" and len(b.text) >= 80]
    k = n_distractors if question.answerable else n_distractors + 2
    picked = evidence + rng.sample(pool, min(k, len(pool)))
    rng.shuffle(picked)
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}::oracle::{i:05d}",
            doc_id=doc.doc_id,
            strategy="section",
            text=b.text[:MAX_CHUNK_CHARS],
            section_path=b.section_path,
            page_start=b.page,
            page_end=b.page,
            char_start=b.char_start,
            char_end=b.char_start + len(b.text[:MAX_CHUNK_CHARS]),
            token_count=0,
        )
        for i, b in enumerate(picked)
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return (
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if path.exists()
        else []
    )


def _append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def label_item(question: EvalQuestion, chunks: list[Chunk], gen: GenerationResult) -> dict[str, Any]:
    """What a human labeller sees: no judge output, so the labels are blind."""
    return {
        "qid": question.qid,
        "qtype": question.qtype,
        "question": question.question,
        "reference": question.reference_answer,
        "evidence": question.evidence_spans,
        "excerpts": [
            {"n": i, "section": " > ".join(c.section_path[-2:]), "page": c.page_start, "text": c.text}
            for i, c in enumerate(chunks, start=1)
        ],
        "answer": gen.answer,
        "abstained": gen.abstained,
        "truncated": gen.truncated,
    }


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    questions = [
        EvalQuestion.model_validate_json(line)
        for line in Path(cfg["questions_path"]).read_text(encoding="utf-8").splitlines()
        if line
    ]
    questions = [q for q in questions if q.split == cfg["split"]][: int(cfg["limit"])]
    docs = {
        d.doc_id: d
        for d in (
            Document.model_validate_json(line)
            for line in Path(cfg["documents_path"]).read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    blocks: dict[str, list[Block]] = defaultdict(list)
    wanted = {q.doc_id for q in questions}
    for line in Path(cfg["blocks_path"]).read_text(encoding="utf-8").splitlines():
        if line and json.loads(line)["doc_id"] in wanted:
            block = Block.model_validate_json(line)
            blocks[block.doc_id].append(block)

    # 1) generate (resumable): all answers first, so only one model is loaded at a time
    answers_path = out / "answers.jsonl"
    done = {r["qid"]: r for r in _read_jsonl(answers_path)}
    generator = OllamaGenerator(GeneratorConfig(**cfg.get("generator", {})))
    for q in questions:
        if q.qid in done:
            continue
        rng = random.Random(f"{cfg['seed']}:{q.qid}")
        chunks = oracle_chunks(q, docs[q.doc_id], blocks[q.doc_id], rng, int(cfg["n_distractors"]))
        gen = generator.generate(q.question, chunks, q.qid, "oracle-context")
        done[q.qid] = {
            "qid": q.qid,
            "chunks": [c.model_dump() for c in chunks],
            "generation": gen.model_dump(),
        }
        _append(answers_path, done[q.qid])
        print(f"generated {q.qid} ({gen.latency_ms / 1000:.1f}s)")

    # 2) judge (resumable)
    judge_path = out / "judge.jsonl"
    judged = {r["qid"]: EvalScores.model_validate(r) for r in _read_jsonl(judge_path)}
    judge = OllamaJudge(JudgeConfig(**cfg.get("judge", {})))
    for q in questions:
        if q.qid in judged:
            continue
        rec = done[q.qid]
        chunks = [Chunk.model_validate(c) for c in rec["chunks"]]
        judged[q.qid] = score_answer(q, GenerationResult.model_validate(rec["generation"]), chunks, judge)
        _append(judge_path, judged[q.qid].model_dump())
        print(f"judged {q.qid}")

    # 3) blind labelling export + summary
    items = [
        label_item(
            q,
            [Chunk.model_validate(c) for c in done[q.qid]["chunks"]],
            GenerationResult.model_validate(done[q.qid]["generation"]),
        )
        for q in questions
    ]
    (out / "label_items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    gens = [GenerationResult.model_validate(done[q.qid]["generation"]) for q in questions]
    summary = {
        "questions": len(questions),
        "judge_model": judge.model,
        "generator_model": gens[0].model if gens else None,
        "judge_scores": {
            k: round(v["mean"], 3) for k, v in aggregate(judged[q.qid] for q in questions).items()
        },
        "abstained": sum(g.abstained for g in gens),
        "truncated": sum(g.truncated for g in gens),
        "judge_seconds_this_run": round(judge.latency_ms / 1000, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def agreement(cfg: dict[str, Any]) -> dict[str, Any]:
    labels = {r["qid"]: r for r in _read_jsonl(Path(cfg["labels_path"]))}
    judged = {
        r["qid"]: EvalScores.model_validate(r) for r in _read_jsonl(Path(cfg["out_dir"]) / "judge.jsonl")
    }
    report = agreement_report(labels, judged)
    (Path(cfg["out_dir"]) / "agreement.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/judge.yaml"))
    parser.add_argument("--agreement", action="store_true", help="compare the judge with human labels")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(json.dumps(agreement(cfg) if args.agreement else run(cfg), indent=2))


if __name__ == "__main__":
    main()
