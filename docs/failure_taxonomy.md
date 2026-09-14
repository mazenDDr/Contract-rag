# Failure taxonomy

**Why do answers go wrong, and at which stage of the pipeline?**

This analysis covers the answer-quality run: 8 retrieval configurations × 70 test questions = 560 answers (`qwen3.5:4b` answering, `gemma4:12b` judging). It went through two gradings of the **same answers**:

| Run | Grading | Answers not fully correct | Mean correctness |
|---|---|---|---|
| `runs/matrix-v1` | The first grader | 282 | 0.59 |
| `runs/matrix-v3` | Fixed grader: `contradiction-only-v5` rubric, and abstention on unanswerable questions judged by the answer text | **185** | **0.71** |

The fixes came out of this analysis (sections 4–5). They were validated against reference labels before being adopted: correctness κ went from 0.48 to 0.60, see `runs/judge-calibration-v5/report.md`. Every table below uses the final grading unless marked otherwise.

The analysis has three layers:

1. **Automatic triage of every failure.** Each is traced through the saved pipeline data to the first stage that failed.
2. **Hand review of 20 cases:**
   - the 16 hardest questions under the first grading (mean correctness ≤ 0.25 across configurations),
   - the 2 unanswerable questions answered most often,
   - 2 "misread" answers on otherwise easy questions, as a check on the grader.
3. **Checks over all 560 rows**, testing whether patterns seen in the hand review hold at scale.

To reproduce:

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_failures.py --run-dir runs/matrix-v1 --out runs/failure-analysis-v1
PYTHONPATH=src .venv/bin/python scripts/analyze_failures.py --run-dir runs/matrix-v3 --out runs/failure-analysis-v3
```

The hand labels are in `data/eval/failure_review.jsonl`.

## Headline findings

1. **The first grader was part of the problem.**
   - 58% of its non-correct grades faulted details "not present in the provided clause", details that were usually true elsewhere in the contract.
   - Its abstention metric mostly scored a JSON flag: 13 of 19 "answered an unanswerable question" answers actually said the information was absent.
   - Fixing both and re-grading the same answers raised mean correctness from 0.59 to 0.71 and cut failures from 282 to 185, without changing a single answer.
2. **With grading fixed, retrieval is the main problem.** 115 of 185 failures (62%) happen before the model sees the evidence:
   - **Partial evidence** (68) comes mostly from multi-part questions, where one half of the question dominates the search. It accounts for 40 of the 53 multi-part failures.
   - **Ranking misses** (40) are a vocabulary gap. 34% of a question's content words appear in evidence that reached the generator, against 12% in evidence that didn't. "Restrict from competing" meets "shall not enter into an agreement with Futurestep".
3. **With the evidence in hand, the 4B model still makes mistakes in 58 answers** (33 unsupported claims, 25 misread). It inverts who holds a right, merges two provisions, cites the wrong excerpt number for a correct fact, or fills a missing duration with a number from an unrelated excerpt.
4. **Sentence-window retrieval is better than its recall suggests.** The evidence the generator actually saw (0.71–0.75) is far above recall@8 on the matched sentence (0.52–0.58). Sentence-window answers match the best fixed-chunk correctness (0.74 vs 0.75).
5. **The fixed grader is unbiased on the hardest cases, but not exact.** On the 18 answerable hand-read cases, my re-reading gives a mean of 0.36. The first grader gave 0.14; the final one gives 0.42. It matches the re-reading exactly on 12 of 18 (11 before): too lenient on 4 (it misses an unanswered part, or a wrong duration that the reference doesn't mention) and too strict on 2.

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

"Reached the generator" is measured on the text the generator was shown (the sentence window when there is one), using the same overlap rule as the relevance labels. "Refused" follows the scores' definition: the flag, or, on unanswerable questions, an answer that says the contract doesn't cover it.

### Failures by category

| Category | Stage | First grading | Final | Share (final) |
|---|---|---|---|---|
| partial_evidence | retrieval | 90 | 68 | 37% |
| ranking_miss | retrieval | 41 | 40 | 22% |
| unsupported_claims | generation | 47 | 33 | 18% |
| misread_evidence | generation | 69 | 25 | 14% |
| answered_unanswerable | generation | 19 | 8 | 4% |
| refused_with_evidence | generation | 9 | 4 | 2% |
| retrieval_miss | retrieval | 4 | 4 | 2% |
| reranker_demotion | retrieval | 3 | 3 | 2% |
| lost_in_chunking | retrieval | 0 | 0 | 0% |
| **total** | | **282** | **185** | |

"Misread evidence" shrank the most (69 → 25): most of it had been the grader, not the model. No evidence was lost in parsing or chunking. Retrieval misses and reranker demotions are rare because contract-scoped search ranks within a single contract, which has few chunks.

### By configuration

| Configuration | Recall in final list | Evidence the generator saw | Partial | Ranking miss | Unsupported | Misread | Answered unanswerable |
|---|---|---|---|---|---|---|---|
| fixed · BM25 | 0.85 | 0.85 | 5 | 4 | 9 | 2 | 1 |
| fixed · BM25 + MiniLM | 0.80 | 0.80 | 9 | 2 (+3 demoted) | 7 | 2 | 1 |
| fixed · BM25+bge-small weighted | 0.84 | 0.84 | 8 | 5 | 5 | 2 | 1 |
| fixed · BM25+bge-base weighted | 0.84 | 0.84 | 6 | 4 | 5 | 6 | 1 |
| section · BM25+bge-base RRF | 0.74 | 0.74 | 12 | 7 | 0 | 3 | 1 |
| section · BM25 | 0.78 | 0.78 | 7 | 7 | 2 | 4 | 1 |
| sentence_window · BM25+bge-small weighted | 0.58 | **0.75** | 12 | 6 | 0 | 4 | 1 |
| sentence_window · BM25 | 0.52 | **0.71** | 9 | 5 | 5 | 2 | 1 |

Fixed-chunk configurations have the most unsupported claims: 512-token excerpts hold several provisions, and the model mixes them up. Full tables are in `runs/failure-analysis-v3/report.md`.

### By question type

Multi-part questions fail mostly by partial evidence: 40 of 53 failures. Single-clause questions (116 failures) split between retrieval (ranking miss 30, partial 25) and generation (unsupported 30, misread 23). Numeric questions rarely fail (8).

## 2. Hand review: root causes

The automatic category names the first stage that failed; reading the case names the *cause*. Each of the 20 cases gets one root cause, plus secondary causes where more than one thing went wrong. The review was done on the first grading. The "evaluation" rows are what the grading fixes addressed.

| Root cause | Stage | Cases | Example | Fix |
|---|---|---|---|---|
| **Multi-part query imbalance** | retrieval | 3 (+1 secondary) | q0036: exclusivity + non-compete. The grant clause ranked 8th; the 3-year non-compete ranked 15th | Search once per sub-question and merge; a larger k for multi-part questions |
| **Vocabulary gap** | retrieval | 3 | q0089: "end early without cause" vs "Shipper decides to terminate … prior to the commencement of transportation service" (ranked 12th) | Rewrite the query into contract language; add category synonyms |
| **Answer in the preamble or definitions** | retrieval | 2 | q0077: the effective date is a bare date on page 1 | Not supported at scale (section 4) |
| **Needle in a long chunk** | generation | 2 (+1 secondary) | q0086: the 3-month sell-off right sat in the top-ranked 512-token chunk, after an indemnity survival clause the model reported instead | Smaller chunks; ask for every relevant provision |
| **Misreading** (inversion, conflation) | generation | 2 (+1 secondary) | q0074: "XIMAGE will give title" became "XIMAGE retains ownership" | A stronger generator; one claim per citation |
| **Invented specifics** | generation | secondary in 3 | q0098: "two years" for the restricted period, a number from an unrelated excerpt, so the number check passed it | Refuse when a duration isn't in the text; check numbers against the clause they describe |
| **Citation drift** | generation | secondary in 3 | q0081: the right fact from excerpt [2], cited as [3] | Verify each citation after generation; reject excerpt numbers that don't exist ([9] of 8) |
| **Topic substitution** | generation | 1 | q0057: no non-compete exists; all 8 configurations answered with the surviving confidentiality duties instead | Tell the model to name what is absent |
| **Abstention flag mismatch** | evaluation | 1 (+1 secondary) | q0050: "The agreement does not contain a specific warranty period…" with `abstained: false` | **Fixed**: on unanswerable questions the text decides |
| **Grader too strict or wrong** | evaluation | 4 (+2 secondary) | q0031: the grader says the answer "incorrectly states" a fact that it does state | **Fixed**: contradiction-only rubric (κ 0.60) |
| **Question-set issue** | evaluation | 2 | q0092: the answer is redacted ([***]), and the model said so. q0038: a template joined a perpetual term with a maintenance renewal term | **Partly fixed**: cited content in a refusal is now graded. Still to do: drop redacted-answer questions |

## 3. The 20 cases

All use fixed chunks + BM25 (contract scope), except case 19 (sentence window + BM25/bge-small).

- **"Fair"** is the score a careful reading supports.
- **v1** and **v3** are the first and the final grading's scores.
- **Automatic (v1 → v3)** is the triage category under each grading.

| # | Question | Type | Automatic (v1 → v3) | Root cause | Fair | v1 | v3 | What happened |
|---|---|---|---|---|---|---|---|---|
| 1 | q0035 | multi_span | partial_evidence | multi-part imbalance | 0.5 | 0 | 0.5 | The transfer policy ranked 1st; the licence grant 21st–25th. The gist is right; the citations drift |
| 2 | q0036 | multi_span | partial_evidence | multi-part imbalance | 0 | 0 | 0 | The non-compete was never retrieved; the answer picked the contract's "Non-Exclusive Grant" over its "exclusive right" |
| 3 | q0080 | cuad_derived | ranking_miss | vocabulary gap | 0 | 0 | 0 | The "non-compete" never says compete; it ranked 21st. The model rightly refused |
| 4 | q0089 | cuad_derived | ranking_miss | vocabulary gap | 0 | 0 | 0 | Ranked 12th; the model then invented termination terms instead of refusing |
| 5 | q0082 | cuad_derived | misread_evidence | grader too strict | 0.5 | 0 | 0.5 | The key fact is right, plus true detail; v1 faulted the detail |
| 6 | q0032 | multi_span | partial_evidence | needle in long chunk | 0.5 | 0 | 0 | The return-of-property duty was in the top chunk and went unused. Both gradings are harsher than fair |
| 7 | q0074 | cuad_derived | unsupported_claims | misreading | 0 | 0 | 0 | Ownership inverted |
| 8 | q0077 | cuad_derived | ranking_miss | preamble / definitions | 0 | 0 | 0 | The date is in the preamble; the answer says only "defined in the first paragraph" |
| 9 | q0081 | cuad_derived | unsupported_claims | grader too strict | 0.5 | 0 | 0.5 | Mostly right; cites the wrong excerpt |
| 10 | q0096 | cuad_derived | unsupported_claims | misreading | 0.5 | 0.5 | 0.5 | Merges two post-term rights; cites excerpt [9], which doesn't exist |
| 11 | q0040 | multi_span | partial_evidence → correct | multi-part imbalance | 0.5 | 0.5 | 1.0 | Custodian access was never retrieved; **v3 is too lenient** |
| 12 | q0086 | cuad_derived | unsupported_claims | needle in long chunk | 0 | 0 | 0 | The key sentence was in chunk #1; the model reported a different provision |
| 13 | q0087 | cuad_derived | ranking_miss | vocabulary gap | 0 | 0 | 0 | "Term" appears everywhere; the clause ranked 36th; the model rightly refused |
| 14 | q0098 | cuad_derived | partial_evidence → correct | preamble / definitions | 0.5 | 0.5 | 1.0 | Invents "two years"; **v3 is too lenient**, though faithfulness still flags the claim |
| 15 | q0066 | cuad_derived | unsupported_claims → correct | grader too strict | 0.5 | 0 | 1.0 | Right in substance; fair could be argued as 0.5 or 1 |
| 16 | q0092 | cuad_derived | refused_with_evidence → correct | question-set issue | 1.0 | 0 | 1.0 | Correctly reported the warranty period as redacted |
| 17 | q0057 | unanswerable | answered_unanswerable | topic substitution | - | wrong | wrong | Answered about confidentiality instead of saying there is no non-compete |
| 18 | q0050 | unanswerable | answered_unanswerable → correct | abstention flag mismatch | - | wrong | right | Said there is no warranty period, with the flag off |
| 19 | q0031 | multi_span | misread_evidence | grader too strict | 1.0 | 0.5 | 0.5 | The grader misreads a correct answer, in both gradings |
| 20 | q0038 | multi_span | misread_evidence → correct | question-set issue | 0.5 | 0.5 | 1.0 | A confusing combined question; missed the renewal clause. **v3 is too lenient** |

## 4. Do the patterns hold beyond 20 cases?

Patterns from the hand review, counted over all 560 answers (`checks` in `analysis/failures.py`):

| Pattern | Measurement | First grading | Final | Verdict |
|---|---|---|---|---|
| Grader faults true extra detail | Non-correct grades whose reason says a detail is not in the clause or reference | 128 of 221 (58%) | 36 of 152 (24%) | **Held, and fixed.** A regex on the grader's wording, so an estimate |
| Abstention flag mismatch | "Answered unanswerable" rows whose text says the information is missing | 13 of 19 | 2 of 8 | **Held, and fixed** |
| Vocabulary gap | Share of a question's content words (stemmed) found in the evidence | 0.34 delivered (n=996) vs 0.12 missed (n=284) | same | **Holds** |
| Dates and terms hide in the preamble | Retrieval-failure rate for questions whose evidence is in a contract's first chunk | 30% vs 29% (8 questions) | 23% vs 25% | **Does not hold**: a story from two cases |

## 5. What was fixed, and what to fix next

**Fixed in grading, with the answers unchanged:**
- **The correctness rubric.** Extra details now count only when they contradict the reference or the highlighted clauses, and the answer must reach the reference's conclusion. A more "mechanical" key-facts rubric was tried first and rejected: κ 0.18, because it lost the answer's conclusion and missed wrong details inside a fact.
- **Abstention on unanswerable questions.** An answer whose first sentence says the contract doesn't cover the question counts as a refusal. On answerable questions the flag still decides, because "does not specify X; instead…" usually introduces the real answer.
- **Refusals with cited content** are graded on that content.

**Next, in order of expected impact:**
1. **Decompose multi-part questions.** Search once per sub-question and merge the results. This targets the largest category (partial evidence, 68).
2. **Close the vocabulary gap.** Rewrite each query into contract language with the local model, then search with both the original and the rewrite. This targets ranking misses (40).
3. **Tighten generation** (58 failures):
   - Use smaller excerpts on fixed chunking (section and sentence-window answers are 0.83–0.87 faithful against 0.55–0.71).
   - Require one claim per citation.
   - Reject citations to excerpts that don't exist.
   - Fix the abstention flag at the source, in the prompt, instead of only in scoring.
4. **Clean the question set.** Drop questions whose reference answer is redacted, and don't pair unrelated CUAD categories in one question.
5. **Watch the grader's leniency.** It now misses an unanswered sub-question or an invented duration that the reference doesn't cover (cases 11, 14, 20). Faithfulness still flags invented details, so read correctness and faithfulness together.

## Limitations

- **Hand review:** one reviewer, 20 cases, chosen from the hardest questions under the first grading, so the category frequencies describe the hard tail, not the whole set. The root causes are judgement calls, and every label is recorded with its reasoning in `data/eval/failure_review.jsonl`.
- **Rubric tuning:** the correctness rubric was chosen on the same 24 reference labels it is scored against, and those labels come from a stronger LLM, not a person. A fresh labelled set would be the clean confirmation.
- **First failing stage only:** the automatic triage names the first stage that failed and hides compound failures, such as a ranking miss followed by an invented answer.
- **Regex estimates:** the grader-strictness and abstention checks match patterns in model-written text, so they estimate rather than count exactly.
- **Where the runs happened:** judging ran partly on an RTX 5060 Ti (matrix-v1) and partly on the Mac (the sentence-window re-judge and the v5 re-grade), with identical `gemma4:12b` weights (`4eb23ef187e2`).
