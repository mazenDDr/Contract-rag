# Failure triage

Source run `matrix-v3` · 560 answers · 185 not fully correct (wrong, partial, refused when answerable, or answered when unanswerable).

Each failure is traced through the pipeline and assigned the first stage that failed (see `src/contract_rag/analysis/failures.py`).

## Failures by category

| Category | Stage | Count | Share of failures |
|---|---|---|---|
| partial_evidence | retrieval | 68 | 37% |
| ranking_miss | retrieval | 40 | 22% |
| unsupported_claims | generation | 33 | 18% |
| misread_evidence | generation | 25 | 14% |
| answered_unanswerable | generation | 8 | 4% |
| refused_with_evidence | generation | 4 | 2% |
| retrieval_miss | retrieval | 4 | 2% |
| reranker_demotion | retrieval | 3 | 2% |

Correct answers where none of the labelled evidence reached the generator: 1 (the answer came from text the labels don't mark).

## By configuration

| Configuration | recall in final list | evidence the generator saw | retrieval_miss | ranking_miss | reranker_demotion | partial_evidence | refused_with_evidence | unsupported_claims | misread_evidence | answered_unanswerable |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed · BM25 · no rerank · contract | 0.85 | 0.85 | 0 | 4 | 0 | 5 | 0 | 9 | 2 | 1 |
| fixed · BM25 · minilm · contract | 0.80 | 0.80 | 0 | 2 | 3 | 9 | 0 | 7 | 2 | 1 |
| fixed · BM25+bge-small (weighted a=0.4) · no rerank · contract | 0.84 | 0.84 | 0 | 5 | 0 | 8 | 0 | 5 | 2 | 1 |
| fixed · BM25+bge-base (weighted a=0.4) · no rerank · contract | 0.84 | 0.84 | 0 | 4 | 0 | 6 | 0 | 5 | 6 | 1 |
| sentence_window · BM25+bge-small (weighted a=0.4) · no rerank · contract | 0.58 | 0.75 | 0 | 6 | 0 | 12 | 0 | 0 | 4 | 1 |
| section · BM25+bge-base (RRF) · no rerank · contract | 0.74 | 0.74 | 0 | 7 | 0 | 12 | 3 | 0 | 3 | 1 |
| section · BM25 · no rerank · contract | 0.78 | 0.78 | 1 | 7 | 0 | 7 | 0 | 2 | 4 | 1 |
| sentence_window · BM25 · no rerank · contract | 0.52 | 0.71 | 3 | 5 | 0 | 9 | 1 | 5 | 2 | 1 |

## By question type

| Type | retrieval_miss | ranking_miss | reranker_demotion | partial_evidence | refused_with_evidence | unsupported_claims | misread_evidence | answered_unanswerable |
|---|---|---|---|---|---|---|---|---|
| cuad_derived | 3 | 30 | 2 | 25 | 3 | 30 | 23 | 0 |
| multi_span | 0 | 7 | 1 | 40 | 0 | 3 | 2 | 0 |
| numeric | 1 | 3 | 0 | 3 | 1 | 0 | 0 | 0 |
| unanswerable | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |

## Hardest questions (mean correctness across configurations <= 0.25)

| Question | Type | Mean correctness | Categories |
|---|---|---|---|
| q0036 | multi_span | 0.00 | partial_evidence 5, ranking_miss 3 |
| q0080 | cuad_derived | 0.00 | ranking_miss 6, retrieval_miss 2 |
| q0089 | cuad_derived | 0.00 | ranking_miss 4, unsupported_claims 3, retrieval_miss 1 |
| q0032 | multi_span | 0.12 | partial_evidence 7, unsupported_claims 1 |
| q0074 | cuad_derived | 0.12 | unsupported_claims 3, ranking_miss 2, reranker_demotion 1, correct 1, partial_evidence 1 |
| q0077 | cuad_derived | 0.19 | ranking_miss 7, correct 1 |
| q0035 | multi_span | 0.25 | partial_evidence 8 |
| q0086 | cuad_derived | 0.25 | unsupported_claims 3, misread_evidence 3, correct 2 |

## Checks on patterns from the hand review (all rows)

- **Grader strictness:** of 152 answers not graded correct, 36 (24%) give a reason citing details outside the highlighted clause or the reference.
- **Abstention flag:** 2 of 8 answers to unanswerable questions say in their text that the information isn't there, without setting the abstention flag; 4 of 42 refusals on answerable questions explain a redacted value.
- **Vocabulary gap:** 0.34 of a question's content words appear in evidence that reached the generator (n=996), against 0.12 in evidence that didn't (n=284).
- **Preamble:** 8 questions have evidence in a contract's first fixed-size chunk; their retrieval-failure rate is 23% against 25% for the rest.
