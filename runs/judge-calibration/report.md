# Judge calibration

**Question:** can a small local LLM judge be trusted to score answer faithfulness and correctness?

**Setup.**
- **Questions:** the 30 dev questions.
- **Answers:** `qwen3.5:4b` answers each one from a fixed context: the parsed blocks containing the gold evidence, plus 3 distractor blocks from the same contract. Retrieval is deliberately out of the loop.
- **Judge:** `gemma4:12b` (thinking off) scores every answer.
- **Reference labels:** faithful yes/no and correct yes/partial/no, for the 24 answers where the model did not decline. They were produced by a **stronger frontier LLM, not a human**, and assigned blind, without seeing the judge's verdicts. The labels live in `data/eval/judge_labels.jsonl` with severity notes.

## Iterations

| Version | Change | Faithful: agreement · κ | Correct: κ | What it showed |
|---|---|---|---|---|
| v1 | Judge splits the answer into claims and lists the excerpt each claim cites | 77% · 0.41 | 0.51 | An answer with **no citations** scored 1.0, because the guard trusted the judge's own citation list. Relevance was 5/5 for every answer. |
| v2 | Code splits the answer into sentences and reads citations from the answer text. Each sentence is judged against only the excerpts it cites. Relevance = share of on-topic statements. | 76% · 0.35 | 0.43 | The uncited answer is fixed. But one sentence citing three excerpts was judged against all three together, and an error in one clause passed. |
| v3 | Split into citation-bounded clauses | 76% · 0.35 | 0.43 | The contradicted clause is now isolated with its own excerpt, and the model **still** accepts it. The bottleneck is the model, not the structure. |
| v3 + thinking | `think=true`, 6,000-token budget | – | – | Stopped after 12m42s without finishing the first answer. At least 80× slower than without thinking, so infeasible for the ablation grid. |
| **v4 (final)** | + every number in a statement (amount, days, date, section number) must appear in its cited excerpts | **81% · 0.50** | 0.43 | Catches invented section numbers such as "Article 5.1", with no false flags. |
| v4, 24 labels | Same judge; 2 replaced dev questions now answered and labelled | **83% · 0.57** | 0.48 | The judge also catches a reversed license grant (q0005). |

κ is Cohen's kappa, which is agreement corrected for chance. All v1–v4 rows use the same 22 labels, so they are comparable. The last row adds 2 labels.

## Final judge on the 30 answers

| Faithfulness | Citation validity | Relevance | Correctness | Abstention accuracy | Judge time |
|---|---|---|---|---|---|
| 0.891 | 0.962 | 0.989 | 0.712 | 0.933 | 325 s (≈11 s/answer) |

## Remaining misses (judge says faithful; reference says no)

- **q0002** (major): the answer says "within a reasonable time" where the excerpt says "immediately".
- **q0014** (major): the claims are not stated in the cited excerpts.
- **q0025** (minor): adds the unit "days" to a redacted period.
- **q0028** (minor): adds "from the date of execution".

None of these involve a number, so the deterministic check cannot catch them.

## How to use this judge

- **Use for comparisons:** faithfulness is suitable for *comparing* configurations. The judge is consistent, catches uncited and number-level fabrications, and runs in about 11 s per answer.
- **Not an absolute measure:** it misses subtle contradictions, so an absolute "faithfulness = 0.89" overstates quality.
- **In failure analysis:** re-check the worst cases with the stronger reference grader.
- **Relevance** remains near-saturated (0.99) and is not informative here.

## Limitations

- **Small sample:** 22–24 labelled answers, so the κ estimates are imprecise.
- **LLM reference labels:** the labels come from an LLM, not a human.
- **Designed on the same items:** v2–v4 were designed after inspecting disagreements on these same items. The changes are structural (code-side parsing and a numeric rule), not item-specific prompt tuning, but a fresh labelled set would give an unbiased estimate.
