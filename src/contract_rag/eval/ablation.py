"""Retrieval ablation: every chunking x retriever x reranker x scope combination, scored on the eval set.

Retrieval metrics need no LLM, so the whole grid runs. Three rules keep it affordable and honest:
- identical retrieval calls and (query, passage) reranker scores are computed once and reused;
- latency comes only from uncached calls, reported per component (BM25, each embedding model, each
  reranker);
- weighted-fusion alpha is tuned on dev questions, configs are selected on dev, and results are reported
  on test.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from contract_rag.eval.labels import QuestionLabels, alignment_report, build_labels
from contract_rag.eval.retrieval_metrics import aggregate, paired_difference, score_retrieval
from contract_rag.retrieval.config import IndexConfig, load_chunks
from contract_rag.retrieval.embed import Encoder, SentenceTransformerEncoder
from contract_rag.retrieval.pipeline import RetrievalPipeline, RetrievalResources, scoped_query
from contract_rag.retrieval.rerank import CrossEncoderReranker, PairScorer
from contract_rag.schemas import Document, EvalQuestion, EvalScores, RetrievalConfig, RetrievedChunk

KS = (1, 3, 5, 8, 10, 20)
REPORTED = ("recall@1", "recall@5", "context_recall", "mrr", "ndcg_at_10", "context_precision")


class AblationConfig(BaseModel):
    questions_path: Path = Path("data/eval/questions.jsonl")
    documents_path: Path = Path("data/processed/documents.jsonl")
    retrieval_config: Path = Path("configs/retrieval.yaml")
    runs_dir: Path = Path("runs")
    report_path: Path = Path("docs/ablations.md")
    chunkings: list[str] = Field(default_factory=lambda: ["fixed", "sentence_window", "section"])
    dense_models: list[str] = Field(default_factory=lambda: ["bge-small", "bge-base", "e5-base"])
    rerankers: list[str] = Field(default_factory=lambda: ["minilm", "bge-reranker-base"])
    alpha_grid: list[float] = Field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    scopes: list[str] = Field(default_factory=lambda: ["doc", "corpus"])
    raw_query_controls: bool = True
    k_candidates: int = 50
    k_final: int = 8
    n_boot: int = 2000
    seed: int = 0
    select_top: int = 8


# ---------- caching wrappers ----------


class CachedRetriever:
    """Memoizes retrieve(); records the latency of real (uncached) calls under `component`."""

    def __init__(self, inner: Any, component: str, timings: dict[str, list[float]]):
        self.inner, self.component, self.timings = inner, component, timings
        self.cache: dict[tuple, list[RetrievedChunk]] = {}

    def retrieve(self, query: str, k: int, doc_id: str | None = None) -> list[RetrievedChunk]:
        key = (query, k, doc_id)
        if key not in self.cache:
            start = time.perf_counter()
            self.cache[key] = self.inner.retrieve(query, k, doc_id)
            self.timings[self.component].append((time.perf_counter() - start) * 1000)
        return self.cache[key]


class CachingScorer:
    """Scores each (query, passage) pair once; records per-pair latency of real calls."""

    def __init__(self, inner: PairScorer, component: str, timings: dict[str, list[float]]):
        self.inner, self.component, self.timings = inner, component, timings
        self.cache: dict[tuple[str, str], float] = {}
        self.pairs_scored = 0

    def predict(self, sentences: list[tuple[str, str]], **kwargs: Any) -> list[float]:
        keys = [(q, hashlib.sha1(t.encode()).hexdigest()) for q, t in sentences]
        missing: dict[tuple[str, str], tuple[str, str]] = {}
        for key, pair in zip(keys, sentences, strict=True):
            if key not in self.cache:
                missing.setdefault(key, pair)
        if missing:
            start = time.perf_counter()
            scores = self.inner.predict(list(missing.values()), **kwargs)
            per_pair = (time.perf_counter() - start) * 1000 / len(missing)
            self.timings[self.component].extend([per_pair] * len(missing))
            self.cache.update(zip(missing, (float(s) for s in scores), strict=True))
            self.pairs_scored += len(missing)
        return [self.cache[k] for k in keys]


def _cross_encoder(name: str, device: str, max_length: int) -> PairScorer:
    import torch
    from sentence_transformers import CrossEncoder

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    return CrossEncoder(name, device=device, max_length=max_length)


# ---------- the grid ----------


def describe(rc: RetrievalConfig, query_mode: str = "scoped") -> str:
    if rc.sparse and rc.dense_model:
        fusion = "RRF" if rc.fusion == "rrf" else f"weighted a={rc.fusion_alpha:g}"
        retrieval = f"BM25+{rc.dense_model} ({fusion})"
    else:
        retrieval = "BM25" if rc.sparse else rc.dense_model
    parts = [rc.chunking, retrieval, rc.reranker or "no rerank", "contract" if rc.doc_filter else "corpus"]
    if rc.doc_filter and query_mode == "raw":
        parts.append("name kept in query")
    return " · ".join(parts)


def config_key(rc: RetrievalConfig, query_mode: str) -> str:
    return rc.config_id() + ("__rawq" if rc.doc_filter and query_mode == "raw" else "")


def retrieval_variants(dense_models: Sequence[str], alphas: dict[tuple[str, str], float], chunking: str):
    variants: list[dict[str, Any]] = [{"sparse": True}]
    variants += [{"sparse": False, "dense_model": m} for m in dense_models]
    for m in dense_models:
        variants.append({"sparse": True, "dense_model": m, "fusion": "rrf"})
        variants.append(
            {"sparse": True, "dense_model": m, "fusion": "weighted", "fusion_alpha": alphas[(chunking, m)]}
        )
    return variants


def build_grid(
    cfg: AblationConfig, alphas: dict[tuple[str, str], float]
) -> list[tuple[RetrievalConfig, str]]:
    """(RetrievalConfig, query_mode). Contract scope uses the scoped query; corpus scope needs the name."""
    common = {"k_candidates": cfg.k_candidates, "k_final": cfg.k_final}
    items = []
    for chunking in cfg.chunkings:
        for variant in retrieval_variants(cfg.dense_models, alphas, chunking):
            for reranker in [None, *cfg.rerankers]:
                for scope in cfg.scopes:
                    rc = RetrievalConfig(
                        chunking=chunking, reranker=reranker, doc_filter=scope == "doc", **variant, **common
                    )
                    items.append((rc, "scoped" if scope == "doc" else "raw"))
    if cfg.raw_query_controls and "doc" in cfg.scopes:
        dense = cfg.dense_models[-1]
        controls = [{"sparse": True}, {"sparse": False, "dense_model": dense}]
        controls.append({"sparse": True, "dense_model": dense, "fusion": "rrf"})
        for chunking in cfg.chunkings:
            for variant in controls:
                items.append(
                    (RetrievalConfig(chunking=chunking, doc_filter=True, **variant, **common), "raw")
                )
    return items


# ---------- running ----------


class AblationRunner:
    def __init__(
        self,
        index_config: IndexConfig,
        repo_root: Path,
        encoder_factory: Callable[[str, str], Encoder] = SentenceTransformerEncoder,
        scorer_factory: Callable[[str, str, int], PairScorer] = _cross_encoder,
    ):
        self.timings: dict[str, list[float]] = defaultdict(list)
        self.scorer_factory = scorer_factory
        self.scorers: dict[str, CachingScorer] = {}
        self.resources = RetrievalResources(
            index_config, repo_root, encoder_factory=encoder_factory, reranker_factory=self._make_reranker
        )
        self._retrievers: dict[tuple[str, ...], CachedRetriever] = {}
        self.names = {cfg.name: key for key, cfg in index_config.rerankers.items()}

    def _make_reranker(self, name: str, device: str = "auto", batch_size: int = 32, max_length: int = 512):
        scorer = CachingScorer(
            self.scorer_factory(name, device, max_length), f"rerank:{self.names[name]}", self.timings
        )
        self.scorers[self.names[name]] = scorer
        return CrossEncoderReranker(name, batch_size=batch_size, scorer=scorer)

    def _retriever(self, kind: str, strategy: str, model: str | None = None) -> CachedRetriever:
        key = (kind, strategy, model or "")
        if key not in self._retrievers:
            inner = self.resources.bm25(strategy) if kind == "bm25" else self.resources.dense(strategy, model)
            self._retrievers[key] = CachedRetriever(
                inner, "bm25" if kind == "bm25" else f"dense:{model}", self.timings
            )
        return self._retrievers[key]

    def pipeline(self, rc: RetrievalConfig) -> RetrievalPipeline:
        return RetrievalPipeline(
            rc,
            bm25=self._retriever("bm25", rc.chunking) if rc.sparse else None,
            dense=self._retriever("dense", rc.chunking, rc.dense_model) if rc.dense_model else None,
            reranker=self.resources.reranker(rc.reranker) if rc.reranker else None,
            chunk_text=self.resources.chunk_text(rc.chunking) if rc.reranker else None,
        )

    def evaluate(
        self,
        rc: RetrievalConfig,
        query_mode: str,
        questions: Sequence[EvalQuestion],
        labels: dict[str, QuestionLabels],
    ) -> tuple[list[EvalScores], list[dict[str, Any]]]:
        pipe, key = self.pipeline(rc), config_key(rc, query_mode)
        scores, records = [], []
        for q in questions:
            query = (
                scoped_query(q.question, q.contract_name, rc.doc_filter)
                if query_mode == "scoped"
                else q.question
            )
            result = pipe.run(query, qid=q.qid, doc_id=q.doc_id)
            result.config_id = key
            scores.append(score_retrieval(result, labels[q.qid], ks=KS))
            records.append(
                {
                    "qid": q.qid,
                    "config_id": key,
                    "final": [c.chunk_id for c in result.final],
                    "stages": {
                        stage: [c.chunk_id for c in items[:20]] for stage, items in result.stages.items()
                    },
                }
            )
        return scores, records

    def tune_alphas(
        self,
        cfg: AblationConfig,
        dev: Sequence[EvalQuestion],
        labels: dict[str, dict[str, QuestionLabels]],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Best weighted-fusion alpha per (chunking, model), on dev questions only (MRR, then R@8)."""
        tuned = {}
        for chunking in cfg.chunkings:
            for model in cfg.dense_models:
                results = {}
                for alpha in cfg.alpha_grid:
                    rc = RetrievalConfig(
                        chunking=chunking,
                        sparse=True,
                        dense_model=model,
                        fusion="weighted",
                        fusion_alpha=alpha,
                        k_candidates=cfg.k_candidates,
                        k_final=cfg.k_final,
                    )
                    agg = aggregate(self.evaluate(rc, "scoped", dev, labels[chunking])[0], n_boot=1)
                    results[alpha] = (round(agg["mrr"]["mean"], 4), round(agg["context_recall"]["mean"], 4))
                best = max(cfg.alpha_grid, key=lambda a: (results[a], -a))
                tuned[(chunking, model)] = {
                    "alpha": best,
                    "dev_mrr_recall8": {str(a): v for a, v in results.items()},
                }
        return tuned

    def close(self) -> None:
        self.resources.close()


# ---------- summarising ----------


def _metrics(scores: Iterable[EvalScores], n_boot: int, seed: int) -> dict[str, dict[str, float]]:
    agg = aggregate(scores, n_boot=n_boot, seed=seed)
    return {m: {k: round(v, 4) for k, v in agg[m].items()} for m in REPORTED if m in agg}


def component_latency(timings: dict[str, list[float]], k_candidates: int) -> dict[str, float]:
    """Median milliseconds per query for each component (rerankers: median per pair x candidates)."""
    out = {}
    for component, values in sorted(timings.items()):
        median = statistics.median(values) if values else 0.0
        out[component] = round(median * k_candidates if component.startswith("rerank:") else median, 2)
    return out


def estimated_latency(rc: RetrievalConfig, latency: dict[str, float]) -> float:
    total = latency.get("bm25", 0.0) if rc.sparse else 0.0
    total += latency.get(f"dense:{rc.dense_model}", 0.0) if rc.dense_model else 0.0
    total += latency.get(f"rerank:{rc.reranker}", 0.0) if rc.reranker else 0.0
    return round(total, 1)


def summarize(
    cfg: AblationConfig,
    items: list[tuple[RetrievalConfig, str]],
    scores: dict[str, list[EvalScores]],
    splits: dict[str, str],
    latency: dict[str, float],
) -> dict[str, Any]:
    by_key = {config_key(rc, mode): (rc, mode) for rc, mode in items}
    table = {}
    for key, (rc, mode) in by_key.items():
        table[key] = {
            "description": describe(rc, mode),
            "scope": "contract" if rc.doc_filter else "corpus",
            "query": mode,
            "est_latency_ms": estimated_latency(rc, latency),
            **{
                split: _metrics((s for s in scores[key] if splits[s.qid] == split), cfg.n_boot, cfg.seed)
                for split in ("dev", "test")
            },
        }

    def dev_rank(key: str) -> tuple[float, float]:
        return (table[key]["dev"]["context_recall"]["mean"], table[key]["dev"]["mrr"]["mean"])

    signatures: dict[str, tuple] = {}

    def dev_signature(key: str) -> tuple:
        if key not in signatures:
            dev = sorted((s for s in scores[key] if splits[s.qid] == "dev"), key=lambda s: s.qid)
            signatures[key] = tuple((s.qid, s.context_recall, s.mrr) for s in dev)
        return signatures[key]

    # A configuration that scores the same as a selected one on every dev question is folded into it:
    # under contract scope a reranker often sees the whole contract, so the first stage stops mattering.
    scoped = [k for k, (rc, mode) in by_key.items() if rc.doc_filter and mode == "scoped"]
    selected: list[str] = []
    for key in sorted(scoped, key=dev_rank, reverse=True):
        twin = next((k for k in selected if dev_signature(k) == dev_signature(key)), None)
        if twin is not None:
            table[twin].setdefault("tied_on_dev", []).append(key)
        elif len(selected) < cfg.select_top:
            selected.append(key)

    def test_scores(key: str) -> list[EvalScores]:
        return [s for s in scores[key] if splits[s.qid] == "test"]

    def best(keys: Iterable[str]) -> str:
        return max(keys, key=dev_rank)

    def diff(name: str, base: str, cand: str, metric: str) -> dict[str, Any]:
        d = paired_difference(test_scores(base), test_scores(cand), metric, n_boot=cfg.n_boot, seed=cfg.seed)
        return {
            "comparison": name,
            "baseline": table[base]["description"],
            "candidate": table[cand]["description"],
            "metric": metric,
            **{k: round(v, 4) for k, v in d.items()},
            "significant": d["ci_low"] > 0 or d["ci_high"] < 0,
        }

    comparisons = []
    for chunking in cfg.chunkings:
        pool = {k: by_key[k][0] for k in scoped if by_key[k][0].chunking == chunking}
        bm25 = next(k for k, rc in pool.items() if rc.sparse and not rc.dense_model and not rc.reranker)
        dense = best(k for k, rc in pool.items() if not rc.sparse and not rc.reranker)
        hybrid = best(k for k, rc in pool.items() if rc.sparse and rc.dense_model and not rc.reranker)
        for metric in ("mrr", "context_recall"):
            comparisons.append(diff(f"{chunking}: best dense vs BM25", bm25, dense, metric))
            comparisons.append(diff(f"{chunking}: best hybrid vs BM25", bm25, hybrid, metric))
        base_rc = pool[hybrid]
        for reranker in cfg.rerankers:
            with_rr = next(
                k
                for k, rc in pool.items()
                if rc.model_copy(update={"reranker": None}).config_id() == base_rc.config_id()
                and rc.reranker == reranker
            )
            for metric in ("mrr", "context_recall"):
                comparisons.append(diff(f"{chunking}: + {reranker} on best hybrid", hybrid, with_rr, metric))
        for key, (rc, mode) in by_key.items():
            if mode == "raw" and rc.doc_filter and rc.chunking == chunking:
                scoped_twin = rc.config_id()
                comparisons.append(
                    diff(f"{chunking}: drop contract name ({describe(rc)})", key, scoped_twin, "mrr")
                )

    per_chunking = {c: best(k for k in scoped if by_key[k][0].chunking == c) for c in cfg.chunkings}
    return {
        "selected_on_dev": selected,
        "best_per_chunking": per_chunking,
        "comparisons": comparisons,
        "configs": table,
        "latency_ms": latency,
    }


def _fmt(m: dict[str, float], ci: bool = False) -> str:
    return f"{m['mean']:.2f} [{m['ci_low']:.2f}, {m['ci_high']:.2f}]" if ci else f"{m['mean']:.2f}"


def render_report(run_id: str, summary: dict[str, Any], meta: dict[str, Any]) -> str:
    t = summary["configs"]
    lines = [
        "# Retrieval ablations",
        "",
        f"Run `{run_id}` · {meta['questions']} answerable questions "
        f"({meta['dev']} dev / {meta['test']} test) · "
        f"{meta['configs']} configurations · {meta['minutes']:.0f} min on an Apple M4 Pro.",
        "",
        "Recall is *evidence recall*: the share of lawyer-highlighted evidence spans with a relevant chunk "
        "in the top k. R@8 is what the generator receives. MRR is 1/rank of the first relevant chunk. "
        "Brackets are 95% bootstrap intervals over questions. Configurations are **selected on dev and "
        "reported on test**.",
        "",
        "## Configurations selected on dev",
        "",
        "| # | Configuration | dev R@8 | dev MRR | test R@1 | test R@5 | test R@8 | test MRR "
        "| est. ms/query | ties on dev |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, key in enumerate(summary["selected_on_dev"], start=1):
        c = t[key]
        lines.append(
            f"| {i} | {c['description']} | {_fmt(c['dev']['context_recall'])} | {_fmt(c['dev']['mrr'])} | "
            f"{_fmt(c['test']['recall@1'])} | {_fmt(c['test']['recall@5'])} | "
            f"{_fmt(c['test']['context_recall'], True)} | {_fmt(c['test']['mrr'], True)} | "
            f"{c['est_latency_ms']:.0f} | {len(c.get('tied_on_dev', []))} |"
        )
    lines += [
        "",
        "*Ties on dev*: configurations with the same recall and MRR as this one on every dev question. They "
        "are folded into it rather than taking a slot.",
    ]
    if "candidate_pool" in meta:
        lines += ["", "## Candidate pool vs contract size", ""]
        for chunking, pool in meta["candidate_pool"].items():
            lines.append(
                f"- **{chunking}**: median {pool['median_chunks']} chunks per contract; "
                f"{pool['share_within_candidates']:.0%} of contracts fit within the {pool['k_candidates']} "
                "candidates a reranker scores"
            )
        lines += [
            "",
            "When the whole contract fits in the candidate pool, a contract-scoped reranker scores every "
            "chunk and its output no longer depends on which retriever produced the candidates.",
        ]
    lines += [
        "",
        "## What each component contributes (test split, paired difference, 95% CI)",
        "",
        "| Comparison | Metric | Δ (candidate − baseline) | 95% CI | Significant |",
        "|---|---|---|---|---|",
    ]
    for d in summary["comparisons"]:
        lines.append(
            f"| {d['comparison']} | {d['metric']} | {d['mean_diff']:+.3f} | "
            f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] | {'yes' if d['significant'] else 'no'} |"
        )
    lines += ["", "## Best configuration per chunking strategy (chosen on dev, test scores)", ""]
    lines += ["| Chunking | Configuration | test R@1 | test R@8 | test MRR |", "|---|---|---|---|---|"]
    for chunking, key in summary["best_per_chunking"].items():
        c = t[key]
        lines.append(
            f"| {chunking} | {c['description']} | {_fmt(c['test']['recall@1'])} | "
            f"{_fmt(c['test']['context_recall'], True)} | {_fmt(c['test']['mrr'], True)} |"
        )
    lines += ["", "## Latency per component (median ms per query, uncached calls)", ""]
    lines += ["| Component | ms |", "|---|---|"]
    lines += [f"| {k} | {v:.1f} |" for k, v in summary["latency_ms"].items()]
    lines += ["", "## Weighted-fusion alpha (tuned on dev: MRR, then R@8)", ""]
    lines += ["| Chunking · model | alpha |", "|---|---|"]
    lines += [f"| {k} | {v['alpha']:g} |" for k, v in meta["alphas"].items()]
    lines += ["", "## Answer-key quality per chunking", ""]
    for chunking, report in meta["label_quality"].items():
        lines.append(
            f"- **{chunking}**: aligned {report['aligned_rate']:.1%}, "
            f"uncovered spans {report.get('uncovered_spans', 0)}"
        )
    lines += ["", "## Full grid (test split)", ""]
    lines += [
        "| Configuration | R@1 | R@5 | R@8 | MRR | nDCG@10 | est. ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in sorted(t, key=lambda k: (t[k]["scope"], -t[k]["test"]["mrr"]["mean"])):
        c = t[key]["test"]
        lines.append(
            f"| {t[key]['description']} | {_fmt(c['recall@1'])} | {_fmt(c['recall@5'])} | "
            f"{_fmt(c['context_recall'])} | {_fmt(c['mrr'])} | {_fmt(c['ndcg_at_10'])} | "
            f"{t[key]['est_latency_ms']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def run(
    cfg: AblationConfig,
    repo_root: Path,
    run_dir: Path | None = None,
    encoder_factory: Callable[[str, str], Encoder] = SentenceTransformerEncoder,
    scorer_factory: Callable[[str, str, int], PairScorer] = _cross_encoder,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    started = time.perf_counter()
    index_cfg = IndexConfig.model_validate(yaml.safe_load((repo_root / cfg.retrieval_config).read_text()))
    question_lines = (repo_root / cfg.questions_path).read_text(encoding="utf-8").splitlines()
    questions = [
        q for q in (EvalQuestion.model_validate_json(x) for x in question_lines if x.strip()) if q.answerable
    ]
    doc_lines = (repo_root / cfg.documents_path).read_text(encoding="utf-8").splitlines()
    docs = {d.doc_id: d for d in (Document.model_validate_json(x) for x in doc_lines if x.strip())}
    splits = {q.qid: q.split for q in questions}
    dev = [q for q in questions if q.split == "dev"]
    labels, label_quality, candidate_pool = {}, {}, {}
    for chunking in cfg.chunkings:
        chunks = load_chunks(repo_root / index_cfg.chunks_dir / f"{chunking}.jsonl")
        built = build_labels(questions, docs, chunks, chunking)  # type: ignore[arg-type]
        labels[chunking] = {lab.qid: lab for lab in built}
        label_quality[chunking] = alignment_report(built)
        per_doc: dict[str, int] = defaultdict(int)
        for c in chunks:
            per_doc[c.doc_id] += 1
        sizes = sorted(per_doc.values())
        candidate_pool[chunking] = {
            "median_chunks": sizes[len(sizes) // 2],
            "share_within_candidates": sum(n <= cfg.k_candidates for n in sizes) / len(sizes),
            "k_candidates": cfg.k_candidates,
        }

    config_hash = hashlib.sha1(cfg.model_dump_json().encode()).hexdigest()[:8]
    run_dir = run_dir or repo_root / cfg.runs_dir / f"{datetime.now():%Y%m%d-%H%M}_{config_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(json.loads(cfg.model_dump_json()), sort_keys=False))
    scores_path = run_dir / "scores.jsonl"
    summary_path = run_dir / "summary.json"
    # a finished run that is re-summarized keeps the grid's compute time, latency and pair counts
    prior = json.loads(summary_path.read_text()) if summary_path.exists() else None
    prior_minutes = prior["meta"]["minutes"] if prior else 0.0
    evaluated = 0
    done: dict[str, list[EvalScores]] = defaultdict(list)
    if scores_path.exists():  # resume
        for line in scores_path.read_text().splitlines():
            s = EvalScores.model_validate_json(line)
            done[s.config_id].append(s)

    runner = AblationRunner(index_cfg, repo_root, encoder_factory, scorer_factory)
    try:
        alphas = runner.tune_alphas(cfg, dev, labels)
        log("alpha tuned on dev: " + ", ".join(f"{c}/{m}={v['alpha']:g}" for (c, m), v in alphas.items()))
        items = build_grid(cfg, {k: v["alpha"] for k, v in alphas.items()})
        log(f"{len(items)} configurations x {len(questions)} questions")
        with scores_path.open("a") as sfile, gzip.open(run_dir / "retrieval.jsonl.gz", "at") as rfile:
            for i, (rc, mode) in enumerate(items, start=1):
                key = config_key(rc, mode)
                if len(done.get(key, [])) == len(questions):
                    continue
                t0 = time.perf_counter()
                scores, records = runner.evaluate(rc, mode, questions, labels[rc.chunking])
                evaluated += 1
                done[key] = scores
                sfile.write("".join(s.model_dump_json() + "\n" for s in scores))
                rfile.write("".join(json.dumps(r) + "\n" for r in records))
                sfile.flush()
                log(f"[{i}/{len(items)}] {describe(rc, mode)}: {time.perf_counter() - t0:.1f}s")
        if prior and not evaluated:  # nothing new ran; alpha tuning alone gives partial, colder timings
            latency, pairs = prior["latency_ms"], prior["meta"]["reranker_pairs_scored"]
        else:
            latency = component_latency(runner.timings, cfg.k_candidates)
            pairs = {k: s.pairs_scored for k, s in runner.scorers.items()}
    finally:
        runner.close()

    summary = summarize(cfg, items, done, splits, latency)
    meta = {
        "questions": len(questions),
        "dev": len(dev),
        "test": len(questions) - len(dev),
        "configs": len(items),
        "minutes": prior_minutes
        + ((time.perf_counter() - started) / 60 if evaluated or not prior_minutes else 0),
        "alphas": {f"{c} · {m}": v for (c, m), v in alphas.items()},
        "label_quality": label_quality,
        "candidate_pool": candidate_pool,
        "reranker_pairs_scored": pairs,
    }
    summary["meta"] = meta
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    report = render_report(run_dir.name, summary, meta)
    (run_dir / "report.md").write_text(report)
    (repo_root / cfg.report_path).parent.mkdir(parents=True, exist_ok=True)
    (repo_root / cfg.report_path).write_text(report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/ablation.yaml"))
    parser.add_argument("--run-dir", type=Path, help="resume an interrupted run in this directory")
    args = parser.parse_args()
    cfg = AblationConfig.model_validate(yaml.safe_load(args.config.read_text()))
    summary = run(cfg, Path.cwd().resolve(), args.run_dir)
    print(json.dumps({"selected_on_dev": summary["selected_on_dev"], "meta": summary["meta"]}, indent=2))


if __name__ == "__main__":
    main()
