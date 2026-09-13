# Failure taxonomy

**Why do answers go wrong, and at which stage of the pipeline?**

This analysis covers the answer-quality run `runs/matrix-v1`: 8 retrieval configurations × 70 test questions = 560 answers (`qwen3.5:4b` answering, `gemma4:12b` judging). 282 of those answers are not fully correct: wrong, partial, refused when the question was answerable, or answered when it wasn't.

The analysis has three layers:

1. **Automatic triage of all 282 failures.** Each is traced through the saved pipeline data to the first stage that failed.
2. **Hand review of 20 cases:**
   - the 16 hardest questions (mean correctness ≤ 0.25 across configurations),
   - the 2 unanswerable questions answered most often,
   - 2 "misread" answers on otherwise easy questions, as a check on the grader.
3. **Checks over all 560 rows**, testing whether patterns seen in the hand review hold at scale.

To reproduce:

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_failures.py --run-dir runs/matrix-v1
```

This writes `runs/failure-analysis-v1/{triage.jsonl, summary.json, report.md}`. The hand labels are in `data/eval/failure_review.jsonl`.

## Headline findings

1. **Half of the failures happen before the model sees the evidence.** The single biggest cause is multi-part questions that are only half retrieved: 90 of 282 failures (32%) are *partial evidence*, and on multi-part questions that is 45 of 64 failures. One half of the question dominates the search, and the clause for the other half ranks 14th–30th.
2. **Ranking misses are a vocabulary gap.** On average, 34% of a question's content words appear in evidence that reached the generator, against 12% in evidence that didn't. Contracts rarely use the words people ask with: "restrict from competing" meets "shall not enter into an agreement with Futurestep".
3. **With the evidence in hand, the 4B model still misreads.** It inverts who holds a right, merges two provisions, cites the wrong excerpt number for a correct fact, or fills a missing duration with a number from an unrelated excerpt.
4. **The grader is too strict, so reported correctness is a lower bound.**
   - Of 221 answers not graded correct, 128 (58%) are faulted for details "not present in the provided clause", details that are usually true elsewhere in the contract.
   - In 8 of the 20 hand-read cases, the pipeline's score was wrong or too harsh.
   - On re-reading, mean correctness of the 18 answerable hand-read cases rises from 0.14 to 0.39. That is a small, hand-picked, hard sample, so it only shows the direction.
5. **The abstention metric mostly measures a flag.** 13 of the 19 "answered an unanswerable question" failures actually say "the agreement does not specify…" in the text, but leave `abstained` false. Of 42 refusals on answerable questions, 4 were correct reports that the value is redacted (`[***]`).
6. **Sentence-window retrieval is better than its recall suggests.** The evidence the generator actually saw (0.71–0.75) is far above recall@8 on the matched sentence (0.52–0.58), which explains why sentence-window had the best correctness despite the worst recall.

## 1. Automatic triage

Each failed answer is checked in pipeline order, and the first "no" names it:

| Order | Check | If no |
|---|---|---|
| 1 | Is the evidence inside some chunk? | `lost_in_chunking` |
| 2 | Did BM25 or dense rank a chunk holding it? | `retrieval_miss` |
| 3 | Did any of it reach the generator? | `reranker_demotion` if it was in the final cut before the reranker; else `ranking_miss` |
| 4 | Did *all* of it reach the generator? | `partial_evidence` |
| 5 | Did the model answer? | `refused_with_evidence` |
| 6 | Is every statement supported by its citations? | `unsupported_claims`; else `misread_evidence` |
| - | Unanswerable question, not refused | `answered_unanswerable` |

"Reached the generator" is measured on the text the generator was shown (the sentence window when there is one), using the same overlap rule as the relevance labels.

### Failures by category

| Category | Stage | Count | Share |
|---|---|---|---|
| partial_evidence | retrieval | 90 | 32% |
| misread_evidence | generation | 69 | 24% |
| unsupported_claims | generation | 47 | 17% |
| ranking_miss | retrieval | 41 | 15% |
| answered_unanswerable | generation | 19 | 7% |
| refused_with_evidence | generation | 9 | 3% |
| retrieval_miss | retrieval | 4 | 1% |
| reranker_demotion | retrieval | 3 | 1% |
| lost_in_chunking | retrieval | 0 | 0% |

Retrieval stages account for 138 failures and generation stages for 144. No evidence was lost in parsing or chunking. Retrieval misses and reranker demotions are rare because contract-scoped search ranks within a single contract, which has few chunks.

### By configuration

| Configuration | Recall in final list | Evidence the generator saw | Partial | Ranking miss | Unsupported | Misread | Answered unanswerable |
|---|---|---|---|---|---|---|---|
| fixed · BM25 | 0.85 | 0.85 | 8 | 4 | 11 | 7 | 3 |
| fixed · BM25 + MiniLM | 0.80 | 0.80 | 14 | 2 (+3 demoted) | 7 | 6 | 3 |
| fixed · BM25+bge-small weighted | 0.84 | 0.84 | 11 | 5 | 7 | 9 | 3 |
| fixed · BM25+bge-base weighted | 0.84 | 0.84 | 10 | 4 | 6 | 10 | 3 |
| section · BM25+bge-base RRF | 0.74 | 0.74 | 13 | 7 | 4 | 10 | 3 |
| section · BM25 | 0.78 | 0.78 | 8 | 7 | 6 | 12 | 1 |
| sentence_window · BM25+bge-small weighted | 0.58 | **0.75** | 14 | 6 | 0 | 10 | 2 |
| sentence_window · BM25 | 0.52 | **0.71** | 12 | 6 | 6 | 5 | 1 |

Fixed-chunk configurations have the most unsupported claims: 512-token excerpts hold several provisions, and the model mixes them up. The full per-question breakdown is in `runs/failure-analysis-v1/report.md`.

### By question type

Multi-part questions fail mostly by partial evidence (45 of 64 failures). Single-clause questions fail across the board, but most often after the evidence arrived: misread 61, unsupported 43, partial 41 and ranking miss 31, out of 187. Numeric questions rarely fail (12).

## 2. Hand review: root causes

The automatic category names the first stage that failed; reading the case names the *cause*. Each of the 20 cases gets one root cause, plus secondary causes where more than one thing went wrong.

| Root cause | Stage | Cases | Example | Fix to try |
|---|---|---|---|---|
| **Multi-part query imbalance** | retrieval | 3 (+1 secondary) | q0036: exclusivity + non-compete. The grant clause ranked 8th; the 3-year non-compete ranked 15th | Search once per sub-question and merge; a larger k for multi-part questions |
| **Vocabulary gap** | retrieval | 3 | q0089: "end early without cause" vs "Shipper decides to terminate … prior to the commencement of transportation service" (ranked 12th) | Rewrite the query into contract language; add category synonyms |
| **Answer in the preamble or definitions** | retrieval | 2 | q0077: the effective date is a bare date on page 1; "when does it take effect" shares no words with it | Not supported at scale (section 4); keep as anecdote |
| **Needle in a long chunk** | generation | 2 (+1 secondary) | q0086: the 3-month sell-off right sat in the top-ranked 512-token chunk, after an indemnity survival clause the model reported instead | Smaller chunks; ask for every relevant provision |
| **Misreading** (inversion, conflation) | generation | 2 (+1 secondary) | q0074: "XIMAGE will give title" became "XIMAGE retains ownership" | A stronger generator; one claim per citation |
| **Invented specifics** | generation | secondary in 3 | q0098: "two years" for the restricted period, a number from an unrelated excerpt, so the number check passed it | Refuse when a duration isn't in the text; check numbers against the clause they describe |
| **Citation drift** | generation | secondary in 3 | q0081: the right fact from excerpt [2], cited as [3] | Verify each citation after generation; reject excerpt numbers that don't exist ([9] of 8) |
| **Topic substitution** | generation | 1 | q0057: no non-compete exists; all 8 configurations answered with the surviving confidentiality duties instead | Tell the model to name what is absent |
| **Abstention flag mismatch** | evaluation | 1 (+1 secondary) | q0050: "The agreement does not contain a specific warranty period…" with `abstained: false` | Derive the flag from the text, or tighten the schema |
| **Grader too strict or wrong** | evaluation | 4 (+2 secondary) | q0031: the grader says the answer "incorrectly states" a fact that it does state | Grade key facts and contradictions only; don't penalize supported extra detail |
| **Question-set issue** | evaluation | 2 | q0092: the answer is redacted ([***]), and the model said so. q0038: a template joined a perpetual term with a maintenance renewal term | Drop redacted-answer questions; don't combine unrelated CUAD categories |

## 3. The 20 cases

All use fixed chunks + BM25 (contract scope), except case 19 (sentence window + BM25/bge-small). "Score fair?" records whether the pipeline's score matches a careful reading.

| # | Question | Type | Automatic | Root cause | Score fair? | What happened |
|---|---|---|---|---|---|---|
| 1 | q0035 | multi_span | partial_evidence | multi-part imbalance | no (fair ≈ 0.5) | The transfer policy ranked 1st; the licence grant 21st–25th. The gist is right, but citations drift and the grader faulted extra detail |
| 2 | q0036 | multi_span | partial_evidence | multi-part imbalance | yes | The non-compete was never retrieved; the contract has both an "exclusive right" and a "Non-Exclusive Grant", and the answer picked the wrong one |
| 3 | q0080 | cuad_derived | ranking_miss | vocabulary gap | yes | The "non-compete" never says compete; it ranked 21st. The model rightly refused |
| 4 | q0089 | cuad_derived | ranking_miss | vocabulary gap | yes | The clause ranked 12th; the model then invented force-majeure and notice terms instead of refusing |
| 5 | q0082 | cuad_derived | misread_evidence | grader too strict | no (≈ 0.5) | The key fact is right, plus true detail; marked incorrect for "details not in the clause" |
| 6 | q0032 | multi_span | partial_evidence | needle in long chunk | no (≈ 0.5) | The return-of-property duty was in the top chunk and went unused; a generic survival-clause answer |
| 7 | q0074 | cuad_derived | unsupported_claims | misreading | yes | Ownership inverted; the verdict is right, though the grader's stated reason is wrong |
| 8 | q0077 | cuad_derived | ranking_miss | preamble / definitions | yes | The date is in the preamble; the answer "defined in the first paragraph" is useless |
| 9 | q0081 | cuad_derived | unsupported_claims | grader too strict | no (≈ 0.5) | Mostly right; cites the wrong excerpt; faulted for true contract details |
| 10 | q0096 | cuad_derived | unsupported_claims | misreading | yes | Merges two post-term rights; cites excerpt [9], which doesn't exist |
| 11 | q0040 | multi_span | partial_evidence | multi-part imbalance | yes | Servicer review retrieved; Custodian access ranked 14th; the main claim is uncited |
| 12 | q0086 | cuad_derived | unsupported_claims | needle in long chunk | yes | The key sentence was in chunk #1; the model reported a different provision |
| 13 | q0087 | cuad_derived | ranking_miss | vocabulary gap | yes | "Term" appears everywhere; the clause ranked 36th; the model rightly refused |
| 14 | q0098 | cuad_derived | partial_evidence | preamble / definitions | yes | The restricted-period definition was not retrieved; "two years" is borrowed from another excerpt |
| 15 | q0066 | cuad_derived | unsupported_claims | grader too strict | no (≈ 0.5) | Right in substance (ENERGOUS owns the product IP); faulted for true details |
| 16 | q0092 | cuad_derived | refused_with_evidence | question-set issue | no (≈ 1.0) | Correctly reported the warranty period as redacted; scored 0 as a refusal |
| 17 | q0057 | unanswerable | answered_unanswerable | topic substitution | yes | Answered about confidentiality instead of saying there is no non-compete |
| 18 | q0050 | unanswerable | answered_unanswerable | abstention flag mismatch | no | Said there is no warranty period, but the flag was off |
| 19 | q0031 | multi_span | misread_evidence | grader too strict | no (≈ 1.0) | The grader misread a correct answer |
| 20 | q0038 | multi_span | misread_evidence | question-set issue | yes | A confusing combined question; missed the renewal clause at rank 8 |

## 4. Do the patterns hold beyond 20 cases?

Four patterns from the hand review were counted over all 560 answers (`checks` in `analysis/failures.py`):

| Pattern | Measurement | Result | Verdict |
|---|---|---|---|
| Grader faults true extra detail | Non-correct grades whose reason says a detail is not in the clause or reference | 128 of 221 (58%) | **Holds.** It is a regex on the grader's wording, so an estimate |
| Abstention flag mismatch | Answers to unanswerable questions whose text says the information is missing | 13 of 19 | **Holds** |
| Vocabulary gap | Share of a question's content words (stemmed) found in the evidence | 0.34 delivered (n=996) vs 0.12 missed (n=284) | **Holds** |
| Dates and terms hide in the preamble | Retrieval-failure rate for questions whose evidence is in a contract's first chunk | 30% (8 questions) vs 29% | **Does not hold**: a story from two cases |

## 5. What to fix, in order

1. **Fix the grader first.** Until it stops penalizing true extra detail, no generation improvement can be measured reliably. Grade key facts and contradictions only, then re-judge; no new generation is needed. Expect correctness to rise, most of all where the evidence was delivered.
2. **Make abstention unambiguous.** Derive `abstained` from the answer text, or require it whenever the answer says the information is absent. This fixes most of the 19 "answered unanswerable" rows at no model cost.
3. **Decompose multi-part questions.** Search once per sub-question and merge the results. This targets the largest category (partial evidence, 90 rows).
4. **Close the vocabulary gap.** Rewrite each query into contract language with the local model (for example, expected clause wording per CUAD category), then search with both the original and the rewrite.
5. **Tighten generation.** Use smaller excerpts on fixed chunking (section and sentence-window answers are 0.84–0.88 faithful against 0.54–0.70). Require one claim per citation, and reject citations to excerpts that don't exist.
6. **Clean the question set.** Drop questions whose reference answer is redacted, and don't pair unrelated CUAD categories in one question.

## Limitations

- **Hand review:** one reviewer, 20 cases, chosen from the hardest questions, so the category frequencies describe the hard tail, not the whole set. The root causes are judgement calls, and every label is recorded with its reasoning in `data/eval/failure_review.jsonl`.
- **First failing stage only:** the automatic triage names the first stage that failed and hides compound failures, such as a ranking miss followed by an invented answer.
- **Regex estimates:** the grader-strictness and abstention checks match patterns in model-written text, so they estimate rather than count exactly.
- **Evidence labels:** these are CUAD's highlights. Answers can legitimately use other text; here no correct answer lacked labelled evidence, so the labels were not the limiting factor.
