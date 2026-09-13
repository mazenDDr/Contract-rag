"""Full answer evaluation for the retrieval configurations chosen on dev: retrieve -> generate -> judge.

Three phases keep one model busy at a time (retrieval models, then the generator, then the judge).
Identical inputs are never recomputed: an answer is reused when two configurations hand the generator the
same chunks for a question, and a judgement when the answer and chunks are the same. Every phase appends to
disk as it goes, so an interrupted run resumes. Results are reported on the held-out test split.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from contract_rag.eval.judge import RUBRIC, JudgeConfig, OllamaJudge, score_answer
from contract_rag.eval.labels import build_labels
from contract_rag.eval.retrieval_metrics import aggregate, paired_difference, score_retrieval
from contract_rag.generation.generator import GeneratorConfig, OllamaGenerator
from contract_rag.retrieval.config import IndexConfig, load_chunks
from contract_rag.retrieval.pipeline import RetrievalResources, scoped_query
from contract_rag.schemas import (
    Chunk,
    Document,
    EvalQuestion,
    EvalScores,
    GenerationResult,
    RetrievalConfig,
    RetrievalResult,
)

ANSWER_METRICS = (
    "context_recall",
    "mrr",
    "faithfulness",
    "citation_validity",
    "answer_relevance",
    "answer_correctness",
    "abstention_accuracy",
)


class MatrixConfig(BaseModel):
    ablation_summary: Path = Path("runs/ablation-v1/summary.json")
    questions_path: Path = Path("data/eval/questions.jsonl")
    documents_path: Path = Path("data/processed/documents.jsonl")
    retrieval_config: Path = Path("configs/retrieval.yaml")
    runs_dir: Path = Path("runs")
    report_path: Path = Path("docs/answer_quality.md")
    split: str = "test"
    bm25_baselines: bool = True
    select_top: int | None = None  # keep only the top N dev picks (None = all of them)
    extra_configs: list[str] = Field(default_factory=list)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    n_boot: int = 2000
    seed: int = 0
    # Ollama's server grows by gigabytes over hundreds of requests; unloading the model resets it
    reload_every: int = 25
    # re-grade an earlier run: copy its retrieval and answers, and reuse its statement verdicts so only the
    # correctness grading runs again
    reuse_verdicts_from: Path | None = None


def parse_config_key(key: str) -> tuple[RetrievalConfig, str]:
    """Inverse of the ablation's config_key: (RetrievalConfig, query_mode)."""
    parts = key.split("__")
    raw_query = parts[-1] == "rawq"
    if raw_query:
        parts = parts[:-1]
    chunking, sparse, dense, fusion, reranker, k_final, scope = parts
    rc = RetrievalConfig(
        chunking=chunking,  # type: ignore[arg-type]
        sparse=sparse == "bm25",
        dense_model=None if dense == "nodense" else dense,
        fusion="rrf" if fusion == "rrf" else "weighted" if fusion.startswith("w") else "none",
        fusion_alpha=float(fusion[1:]) if fusion.startswith("w") else 0.5,
        reranker=None if reranker == "norerank" else reranker,
        k_final=int(k_final.removeprefix("k")),
        doc_filter=scope == "doc",
    )
    return rc, "raw" if raw_query or scope == "corpus" else "scoped"


def choose_configs(
    summary: dict[str, Any], bm25_baselines: bool, extra: Sequence[str], select_top: int | None = None
) -> list[str]:
    """Top dev-selected configurations, the best one per chunking, then a BM25-only baseline per chunking."""
    keys = list(summary["selected_on_dev"])[:select_top] + list(summary.get("best_per_chunking", {}).values())
    if bm25_baselines:
        for chunking in sorted({parse_config_key(k)[0].chunking for k in keys}):
            baseline = f"{chunking}__bm25__nodense__none__norerank__k8__doc"
            if baseline in summary["configs"]:
                keys.append(baseline)
    keys += list(extra)
    return list(dict.fromkeys(keys))


def _unload(component: Any) -> None:
    """Ask Ollama to drop a model from memory now instead of after its 5-minute keep-alive, so the
    generator and the judge never sit in RAM together on a 24 GB laptop."""
    client = getattr(component, "client", None)
    if hasattr(client, "generate"):
        with contextlib.suppress(Exception):  # an optimisation; never fail the run over it
            client.generate(model=component.model, prompt="", keep_alive=0)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def _append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(
    cfg: MatrixConfig,
    repo_root: Path,
    run_dir: Path,
    resources_factory: Callable[[IndexConfig, Path], RetrievalResources] = RetrievalResources,
    generator_factory: Callable[[GeneratorConfig], Any] = OllamaGenerator,
    judge_factory: Callable[[JudgeConfig], Any] = OllamaJudge,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(json.loads(cfg.model_dump_json()), sort_keys=False))
    index_cfg = IndexConfig.model_validate(yaml.safe_load((repo_root / cfg.retrieval_config).read_text()))
    summary = json.loads((repo_root / cfg.ablation_summary).read_text())
    keys = choose_configs(summary, cfg.bm25_baselines, cfg.extra_configs, cfg.select_top)
    parsed = {k: parse_config_key(k) for k in keys}
    lines = (repo_root / cfg.questions_path).read_text().splitlines()
    questions = [
        q for q in (EvalQuestion.model_validate_json(x) for x in lines if x.strip()) if q.split == cfg.split
    ]
    doc_lines = (repo_root / cfg.documents_path).read_text().splitlines()
    docs = {d.doc_id: d for d in (Document.model_validate_json(x) for x in doc_lines if x.strip())}
    chunkings = sorted({rc.chunking for rc, _ in parsed.values()})
    chunks: dict[str, dict[str, Chunk]] = {}
    labels = {}
    for chunking in chunkings:
        loaded = load_chunks(repo_root / index_cfg.chunks_dir / f"{chunking}.jsonl")
        chunks[chunking] = {c.chunk_id: c for c in loaded}
        labels[chunking] = {lab.qid: lab for lab in build_labels(questions, docs, loaded, chunking)}  # type: ignore[arg-type]
    log(f"{len(keys)} configurations x {len(questions)} {cfg.split} questions")

    prior: dict[tuple[str, str], str | None] = {}
    if cfg.reuse_verdicts_from is not None:
        source = repo_root / cfg.reuse_verdicts_from
        for name in ("retrieval.jsonl", "generation.jsonl"):
            if not (run_dir / name).exists():
                shutil.copyfile(source / name, run_dir / name)
        prior = {
            (s["config_id"], s["qid"]): s["judge_rationale"] for s in _read_jsonl(source / "scores.jsonl")
        }
        log(f"re-grading {source.name}: retrieval and answers copied, {len(prior)} verdict sets to reuse")

    # 1) retrieval (resumable)
    retrieval_path = run_dir / "retrieval.jsonl"
    retrieved = {
        (r["config_id"], r["qid"]): RetrievalResult.model_validate(r) for r in _read_jsonl(retrieval_path)
    }
    resources = resources_factory(index_cfg, repo_root)
    try:
        for key, (rc, mode) in parsed.items():
            todo = [q for q in questions if (key, q.qid) not in retrieved]
            if not todo:  # already retrieved: don't load models or open the index
                continue
            pipe = resources.pipeline(rc)
            for q in todo:
                query = (
                    scoped_query(q.question, q.contract_name, rc.doc_filter)
                    if mode == "scoped"
                    else q.question
                )
                result = pipe.run(query, qid=q.qid, doc_id=q.doc_id)
                result.config_id = key
                retrieved[(key, q.qid)] = result
                _append(retrieval_path, result.model_dump())
            log(f"retrieved: {key}")
    finally:
        resources.close()

    # 2) generation, reused across configurations that retrieved identical chunks (resumable)
    generation_path = run_dir / "generation.jsonl"
    generated = {(g["config_id"], g["qid"]): g for g in _read_jsonl(generation_path)}
    by_context = {(g["qid"], tuple(g["final"])): g["generation"] for g in generated.values()}
    generator = generator_factory(cfg.generator)
    reused = fresh = 0
    for key, (rc, _) in parsed.items():
        for q in questions:
            if (key, q.qid) in generated:
                continue
            final = [c.chunk_id for c in retrieved[(key, q.qid)].final]
            context = (q.qid, tuple(final))
            if context in by_context:
                gen = GenerationResult.model_validate({**by_context[context], "config_id": key})
                reused += 1
            else:
                gen = generator.generate(q.question, [chunks[rc.chunking][c] for c in final], q.qid, key)
                by_context[context] = gen.model_dump()
                fresh += 1
                if fresh % cfg.reload_every == 0:
                    _unload(generator)
            record = {"config_id": key, "qid": q.qid, "final": final, "generation": gen.model_dump()}
            generated[(key, q.qid)] = record
            _append(generation_path, record)
        log(f"generated: {key}")
    _unload(generator)

    # 3) judging, reused for identical (answer, chunks) (resumable)
    scores_path = run_dir / "scores.jsonl"
    scored = {(s["config_id"], s["qid"]): EvalScores.model_validate(s) for s in _read_jsonl(scores_path)}
    judged_context: dict[tuple, EvalScores] = {}
    judge = judge_factory(cfg.judge)
    by_qid = {q.qid: q for q in questions}
    for key, (rc, _) in parsed.items():
        for q in questions:
            if (key, q.qid) in scored:
                continue
            record = generated[(key, q.qid)]
            gen = GenerationResult.model_validate(record["generation"])
            context = (q.qid, gen.answer, gen.abstained, tuple(record["final"]))
            if context not in judged_context:
                final_chunks = [chunks[rc.chunking][c] for c in record["final"]]
                judged_context[context] = score_answer(
                    by_qid[q.qid], gen, final_chunks, judge, prior.get((key, q.qid))
                )
                if len(judged_context) % cfg.reload_every == 0:
                    _unload(judge)
            answer = judged_context[context]
            retrieval = score_retrieval(retrieved[(key, q.qid)], labels[rc.chunking][q.qid])
            merged = retrieval.model_copy(
                update={
                    "config_id": key,
                    **{
                        f: getattr(answer, f)
                        for f in (
                            "faithfulness",
                            "citation_validity",
                            "answer_relevance",
                            "answer_correctness",
                            "abstention_correct",
                            "judge_model",
                            "judge_rationale",
                        )
                    },
                }
            )
            scored[(key, q.qid)] = merged
            _append(scores_path, merged.model_dump())
        log(f"judged: {key}")
    _unload(judge)

    report = summarize(cfg, keys, parsed, summary, scored, retrieved, generated, questions)
    report["meta"] = {
        "configs": len(keys),
        "questions": len(questions),
        "split": cfg.split,
        "generations_reused": reused,
        # this invocation only; a resumed run spans several sessions (and possibly machines)
        "session_minutes": round((time.perf_counter() - started) / 60, 1),
        "generator": cfg.generator.model,
        "judge": cfg.judge.model,
        "grader": RUBRIC,
        "verdicts_reused_from": str(cfg.reuse_verdicts_from) if cfg.reuse_verdicts_from else None,
    }
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    text = render_report(run_dir.name, report)
    (run_dir / "report.md").write_text(text)
    (repo_root / cfg.report_path).parent.mkdir(parents=True, exist_ok=True)
    (repo_root / cfg.report_path).write_text(text)
    return report


def summarize(cfg, keys, parsed, ablation, scored, retrieved, generated, questions) -> dict[str, Any]:
    table = {}
    for key in keys:
        scores = [scored[(key, q.qid)] for q in questions]
        agg = aggregate(scores, n_boot=cfg.n_boot, seed=cfg.seed)
        gen_ms = [generated[(key, q.qid)]["generation"]["latency_ms"] for q in questions]
        ret_ms = [retrieved[(key, q.qid)].latency_ms.get("total", 0.0) for q in questions]
        description = ablation["configs"].get(key, {}).get("description", key)
        table[key] = {
            "description": description,
            "baseline": key.endswith("__bm25__nodense__none__norerank__k8__doc"),
            **{m: {k: round(v, 4) for k, v in agg[m].items()} for m in ANSWER_METRICS if m in agg},
            "median_retrieval_ms": round(statistics.median(ret_ms), 1),
            "median_generation_ms": round(statistics.median(gen_ms), 1),
        }
    comparisons = []
    for key in keys:
        rc = parsed[key][0]
        baseline = f"{rc.chunking}__bm25__nodense__none__norerank__k8__doc"
        if key == baseline or baseline not in table:
            continue
        for metric in ("answer_correctness", "faithfulness"):
            d = paired_difference(
                [scored[(baseline, q.qid)] for q in questions],
                [scored[(key, q.qid)] for q in questions],
                metric,
                n_boot=cfg.n_boot,
                seed=cfg.seed,
            )
            comparisons.append(
                {
                    "candidate": table[key]["description"],
                    "baseline": table[baseline]["description"],
                    "metric": metric,
                    **{k: round(v, 4) for k, v in d.items()},
                    "significant": d["ci_low"] > 0 or d["ci_high"] < 0,
                }
            )
    return {"configs": table, "comparisons": comparisons}


def _cell(entry: dict[str, Any], metric: str, ci: bool = False) -> str:
    m = entry.get(metric)
    if not m:
        return "–"
    return f"{m['mean']:.2f} [{m['ci_low']:.2f}, {m['ci_high']:.2f}]" if ci else f"{m['mean']:.2f}"


def render_report(run_id: str, report: dict[str, Any]) -> str:
    meta, table = report["meta"], report["configs"]
    lines = [
        "# Answer quality",
        "",
        f"Run `{run_id}` · {meta['configs']} retrieval configurations × {meta['questions']} {meta['split']} "
        f"questions · generator `{meta['generator']}`, judge `{meta['judge']}` "
        f"(grading rubric `{meta.get('grader', 'v4')}`).",
        "",
        *(
            [
                f"Re-graded from `{meta['verdicts_reused_from']}`: same retrieval and answers; statement "
                "verdicts reused, correctness graded again.",
                "",
            ]
            if meta.get("verdicts_reused_from")
            else []
        ),
        "Configurations were selected on dev in the retrieval ablation; BM25-only baselines are added. "
        "Judge scores are best used to compare configurations (calibration: faithfulness κ 0.57 against "
        "reference labels). Brackets are 95% bootstrap intervals.",
        "",
        "| Configuration | R@8 | Correctness | Faithfulness | Citation validity | Abstention accuracy "
        "| Relevance | Retrieval ms | Generation ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in sorted(table.values(), key=lambda entry: -entry["answer_correctness"]["mean"]):
        name = e["description"] + (" *(baseline)*" if e["baseline"] else "")
        lines.append(
            f"| {name} | {_cell(e, 'context_recall')} | {_cell(e, 'answer_correctness', True)} | "
            f"{_cell(e, 'faithfulness', True)} | {_cell(e, 'citation_validity')} | "
            f"{_cell(e, 'abstention_accuracy')} | {_cell(e, 'answer_relevance')} | "
            f"{e['median_retrieval_ms']:.0f} | {e['median_generation_ms']:.0f} |"
        )
    lines += [
        "",
        "## Against the BM25 baseline (paired, same questions)",
        "",
        "| Candidate | Metric | Δ | 95% CI | Significant |",
        "|---|---|---|---|---|",
    ]
    for d in report["comparisons"]:
        lines.append(
            f"| {d['candidate']} | {d['metric']} | {d['mean_diff']:+.3f} | "
            f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] | {'yes' if d['significant'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/matrix.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/matrix-v1"))
    args = parser.parse_args()
    cfg = MatrixConfig.model_validate(yaml.safe_load(args.config.read_text()))
    report = run(cfg, Path.cwd().resolve(), args.run_dir)
    print(json.dumps(report["meta"], indent=2))


if __name__ == "__main__":
    main()
