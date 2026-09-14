<p align="center">
  <img src="docs/assets/hero.svg" width="100%" alt="A contract page with the termination clause highlighted, the question asked about it, the model's answer citing excerpt [5], and a CORRECT stamp from the grader.">
</p>

<h1 align="center">contract-rag</h1>

<p align="center">
  <b>Ask a question about a real commercial contract. Get the clause, with a citation, or an honest "the contract doesn't say".</b><br>
  Retrieval-augmented question answering over 100 real contracts, built with small open models that run on a laptop, and measured end to end.
</p>

<p align="center">
  <a href="https://mazenddr.github.io/Contract-rag/"><b>Field guide</b></a> (every part explained, with animations)
  &nbsp;·&nbsp;
  <a href="https://mazenddr.github.io/Contract-rag/tour/"><b>The tour</b></a> (one question, start to finish)
  &nbsp;·&nbsp;
  <a href="https://mazenddr.github.io/Contract-rag/demo/"><b>Try it</b></a> (70 recorded answers, graded)
  &nbsp;·&nbsp;
  <a href="docs/blog.md"><b>The story</b></a> (what 560 graded answers taught me)
</p>

---

## Results at a glance

| | |
|---|---|
| **Search** | Fixed 512-token chunks + BM25 inside the chosen contract: **recall@8 0.85, MRR 0.73** on held-out test questions. Chosen from 189 setups on the dev split. |
| **Answers** | 560 answers graded claim by claim. The served setup scores **correctness 0.75** [0.64, 0.84], citation validity 0.87 and abstention accuracy 0.94. |
| **Grader** | Checked against reference labels before being trusted: **κ 0.60** for correctness and 0.57 for faithfulness. |
| **Failures** | Every wrong answer is traced to the stage that broke: **62% fail in retrieval**, before the model sees the evidence. |
| **Cost** | **$0.** Every model is open-weight and runs locally: `qwen3.5:4b` answers, `gemma4:12b` grades. |

## How it works

<p align="center"><img src="docs/assets/pipeline.svg" width="100%" alt="The pipeline: parse, chunk, search, answer, grade, diagnose. Each stage writes a file the next one reads."></p>

1. **Parse.** PyMuPDF reads every PDF with its layout: headings, tables as Markdown, pages, and repeated headers and footers dropped. Every block keeps its exact character offsets. 100 contracts and 1,580 pages, parsed in 67 s with 0 failures. [→ guide](https://mazenddr.github.io/Contract-rag/#a-parse)
2. **Chunk.** Each contract is cut three ways, so the cut can be measured rather than guessed: fixed 512-token windows, sections at headings, and single sentences with ±3 sentences of context. [→ guide](https://mazenddr.github.io/Contract-rag/#a-chunk)
3. **Search.** BM25 and three embedding models (bge-small, bge-base, e5-base in a local Qdrant), with RRF and weighted fusion and two cross-encoder rerankers, all switchable. Search runs inside the question's contract, with the contract's name taken out of the query. [→ guide](https://mazenddr.github.io/Contract-rag/#a-search)
4. **Answer.** The 8 excerpts are numbered [1]…[8]. A local `qwen3.5:4b` must end every sentence with the excerpt it used, or set `abstained`. Each [n] is mapped back to a chunk id, a page and a section. [→ guide](https://mazenddr.github.io/Contract-rag/#a-answer)
5. **Grade.** `gemma4:12b` checks each statement against *only the excerpts that statement cites*, code checks every number, and a second call grades correctness against the lawyers' reference. [→ guide](https://mazenddr.github.io/Contract-rag/#a-grade)
6. **Diagnose.** Every failed answer is walked back through the saved rankings to the first stage that went wrong. [→ guide](https://mazenddr.github.io/Contract-rag/#a-fail)

## What the experiments found

### Where you cut the text decides what search can find

<p align="center"><img src="docs/assets/chunking.svg" width="100%" alt="The same passage cut three ways; fixed chunks overlap, and the clause sits in the overlap."></p>

Fixed chunks won search. A contract is only about 13 of them long, so the right one is usually near the top. Smaller chunks give *more faithful* answers (0.83–0.87 against 0.62), though, because each excerpt holds one provision instead of several that a small model can mix up.

### The simplest search won, and one line mattered most

<p align="center"><img src="docs/assets/search.svg" width="100%" alt="Differences in MRR per component with 95% intervals: embeddings −0.243, hybrid −0.020, MiniLM +0.006, bge reranker −0.148, dropping the contract name +0.396."></p>

Inside one contract, the contract's own name only matches the title page and the signature block. Removing it from the query was worth **+0.40 MRR**, more than any model. Embeddings, fusion and rerankers didn't beat plain BM25 on these questions: contracts repeat their defined terms word for word. Every comparison is paired, question by question, with a bootstrap interval. Full tables: [`docs/ablations.md`](docs/ablations.md).

### Answers are graded one statement at a time

<p align="center"><img src="docs/assets/grading.svg" width="100%" alt="A real answer split into two statements: the first cites the preamble and is not supported; the second is supported. Faithfulness 0.50, correctness: correct."></p>

A single "rate this answer" score would hide the sentence that's wrong. Here the answer is right, but its first citation points at the wrong excerpt. The grader was checked against reference labels before its scores were used (`runs/judge-calibration-v5`). A rubric that looked more rigorous only reached κ 0.18 and was rejected.

| Setup (70 test questions) | Recall@8 | Correctness | Faithfulness | Citation validity |
|---|---|---|---|---|
| **fixed · BM25 · in contract** (served) | 0.85 | **0.75** [0.64, 0.84] | 0.62 | 0.87 |
| fixed · BM25 + bge-base, weighted | 0.84 | 0.74 [0.63, 0.83] | 0.68 | 0.94 |
| sentence window · BM25 + bge-small, weighted | 0.58 | 0.74 [0.64, 0.83] | 0.87 | 1.00 |
| section · BM25 | 0.78 | 0.71 [0.60, 0.81] | 0.85 | 0.97 |

All 8 setups: [`docs/answer_quality.md`](docs/answer_quality.md).

### Why answers fail

<p align="center"><img src="docs/assets/failures.svg" width="100%" alt="Failures by first failing stage: partial evidence 68, ranking miss 40, unsupported claims 33, misread evidence 25, and smaller categories."></p>

The first surprise was the grader itself. It faulted true details found elsewhere in the contract, and it read a JSON flag instead of the answer's words. Fixing it cut failures from 282 to 185 **without changing a single answer**. What's left is mostly retrieval:
- multi-part questions, where one half of the question dominates the search;
- vocabulary gaps ("restrict from competing" against "shall not enter into an agreement with…").

20 cases were read by hand. The taxonomy and root causes are in [`docs/failure_taxonomy.md`](docs/failure_taxonomy.md).

## Try it

**Online, instantly:** [recorded answers](https://mazenddr.github.io/Contract-rag/demo/). Pick any of the 70 test questions to see exactly what the system answered during the evaluation: its citations with the matched sentence highlighted, its grade, each statement's check and, for the failures, where it went wrong. Nothing runs a model, so it's free and instant.

**Live, on your machine:** the all-in-one image carries the model, the API and the page. Build the data first ([below](#run-it-yourself)), then:

```bash
PYTHONPATH=src .venv/bin/python scripts/push_space.py     # assembles build/space/
docker build -t contract-rag-space build/space
docker run -p 7860:7860 -e APP_API_KEY=choose-a-key contract-rag-space
# the page: http://localhost:7860    the API docs: http://localhost:7860/api/docs

curl -s -H "X-API-Key: choose-a-key" -H "Content-Type: application/json" \
     http://localhost:7860/api/ask \
     -d '{"doc_id": "kubient-inc-07-02-2020-ex-10-14-master-services-agreement-part1",
          "question": "Can a party end the agreement early without cause, and on what notice?"}'
```

In a container the model runs on CPU, which takes 45 s to 3 minutes per answer depending on the cores; with Ollama on a laptop GPU it takes about 10 s. The response includes the answer, `abstained`, each citation's page, section and text, token counts, and the search/answer time split ([`docs/api.md`](docs/api.md)). [Why the 4B model and not a faster, smaller one](docs/deployment.md): the 2B cited its sources only 14% of the time.

## Run it yourself

Needs Python 3.11 and [Ollama](https://ollama.com). Everything runs locally; the laptop used was an Apple M4 Pro with 24 GB.

```bash
make setup
ollama pull qwen3.5:4b && ollama pull gemma4:12b
export PYTHONPATH=src

.venv/bin/python scripts/download_cuad.py      # CUAD → manifests, 100-contract subset, test fixture
.venv/bin/python scripts/parse_cuad.py         # PDFs → documents + blocks
.venv/bin/python scripts/chunk_cuad.py         # three chunk files
.venv/bin/python scripts/build_indexes.py      # BM25 + vectors (embeddings cached)
.venv/bin/python scripts/build_questions.py    # the 100-question exam
.venv/bin/python scripts/run_ablation.py       # 189 search setups → docs/ablations.md
.venv/bin/python scripts/run_matrix.py --config configs/matrix.yaml --run-dir runs/matrix-v1
.venv/bin/python scripts/analyze_failures.py --run-dir runs/matrix-v3 --out runs/failure-analysis-v3

APP_API_KEY=choose-a-key .venv/bin/python scripts/serve_api.py
API_URL=http://127.0.0.1:8000 APP_API_KEY=choose-a-key streamlit run ui/streamlit_app.py

make check                                     # ruff + 94 tests on a 3-contract fixture, no network
```

## Repository layout

```
src/contract_rag/
  schemas.py      shared data shapes: Document → Block → Chunk → RetrievalResult → GenerationResult → EvalScores
  ingest/         download CUAD, layout-aware PDF parsing
  chunking/       fixed, section and sentence-window chunkers
  retrieval/      BM25, embeddings + Qdrant, fusion, rerankers, the pipeline
  generation/     citation-forcing prompt, Ollama client
  eval/           questions, answer key, retrieval metrics, ablation, grader, calibration, matrix runs
  analysis/       failure triage
  api/            FastAPI service
ui/               Streamlit page
configs/          every setting, in YAML
data/eval/        the committed exam, grader labels and hand-review labels
runs/*/           the committed reports and summaries behind every number here
deploy/space/     the all-in-one image: model + API + page
site/             the field guide, the tour and the recorded answers (GitHub Pages)
```

A map of every file with what it does is in the [field guide](https://mazenddr.github.io/Contract-rag/#a-tree).

## At scale

This project answers questions about one named contract at a time, over 100 contracts. Here is what would change at 100× that, and what the measurements already say about it:

- **Search without a contract name.** Scoping to one contract is why plain BM25 wins: the median contract is 13 chunks. Searching all 100 contracts instead drops fixed-chunk BM25 recall@8 from 0.85 to 0.49 (the corpus-scope rows in `docs/ablations.md`). Across thousands of contracts, embeddings, metadata filters (party, contract type, date) and a reranker would start earning their cost, and the ablation harness can measure that with one config change.
- **Indexes.** BM25 would move to OpenSearch or Elasticsearch, and vectors to a Qdrant server with the same `doc_id` payload index used here. Embeddings are already cached by (model, text hash), so re-ingesting is incremental.
- **Ingestion.** Parsing runs at about 0.04 s per page with a per-document cache, so it parallelises across workers without changes. A million pages is about 12 CPU-hours.
- **Serving.** Nearly all answer time is the model reading about 4,400 prompt tokens: about 10 s on a laptop GPU, and minutes on a CPU. A GPU inference server with batching and prefix caching, streamed answers, and a queue in place of the service's single lock are the next steps. So is caching answers to repeated questions.
- **Evaluation as a gate.** The harness is config-driven and resumable. The 16-minute search ablation could run on every change, a sampled answer matrix nightly, and grader agreement could be tracked on a growing labelled set.
- **What to fix first.** The failure analysis ranks it:
  - split multi-part questions into sub-searches (partial evidence caused 40 of the 53 multi-part failures);
  - rewrite queries into the contract's own vocabulary;
  - verify each citation after generation.

## Built with

PyMuPDF · tiktoken · NLTK · bm25s + PyStemmer · sentence-transformers (bge, e5) · Qdrant · cross-encoders · Ollama (`qwen3.5:4b`, `gemma4:12b`) · FastAPI · Streamlit · Docker

## License

The code is released under the [MIT License](LICENSE). The contracts and annotations it evaluates on are [CUAD v1](https://www.atticusprojectai.org/cuad) by The Atticus Project, licensed separately under CC BY 4.0.
