# Judge calibration: correctness rubric v5

This run judges the same 30 dev answers (generated from oracle context) as the v4 calibration and compares them with the same 24 reference labels. Only the correctness grading changed. The statement check and its prompt are identical, so faithfulness agreement is unchanged.

| Correctness rubric | Correctness agreement | Correctness κ | Faithfulness κ | Mean correctness (30) |
|---|---|---|---|---|
| v4: holistic, faults details not in the highlighted clause | 75% | 0.48 | 0.57 | 0.71 |
| Key facts: facts listed per answer, verdict computed in code (rejected) | 67% | 0.18 | 0.57 | 0.79 |
| **contradiction-only-v5 (adopted)** | **83%** | **0.60** | 0.57 | 0.79 |

v5 confusion (reference label → judge): yes→yes 15, partial→partial 5, yes→partial 2, partial→yes 2.

## Why the change

In the answer-quality run, 58% of non-correct grades faulted details "not present in the provided clause". In the hand review those details were usually true elsewhere in the contract (`docs/failure_taxonomy.md`). v5 keeps the v4 grader and adds two rules:

- Extra details count against an answer only when they contradict the reference or the highlighted clauses. Whether they are supported by the contract is left to the statement check.
- The answer must reach the reference's conclusion.

## Why the key-facts rubric was rejected

It looked more principled (list the reference's key facts, mark each one, compute the grade in code), but it agreed worse. Reading its eight disagreements with the reference labels showed three problems:

1. **It loses the conclusion.** Answers with every fact right but the opposite yes/no conclusion ("there is no revenue sharing") were marked correct.
2. **It ignores wrong details inside a fact.** "Within a reasonable time" where the clause says "immediately", or a licence granted in the wrong direction, still counted as stated.
3. **Its fact lists are unstable.** Sometimes it added facts the question didn't ask for ("consent not unreasonably withheld"), which was too strict. Sometimes it listed only what the answer mentioned, which was too lenient.

Its results are kept in `runs/judge-calibration-v5-keyfacts/`.

## Other changes in v5 (no effect on these labels)

- **Abstention by text:** an answer whose first sentence says the contract does not cover the question counts as an abstention, even when its flag is off.
- **Flagged content is graded:** a flagged refusal that reports cited content, such as a redacted value, is graded like any other answer.

## Caveats

- **Small sample:** 24 labels, so κ has a wide interval, and moving by a few labels changes it by about 0.1.
- **Labels from an LLM:** the reference labels were written blind by a stronger LLM, not by a person.
- **Tuning on this set:** the rubric was chosen by comparing variants on this same set of labels. A fresh labelled set would be the clean confirmation.
