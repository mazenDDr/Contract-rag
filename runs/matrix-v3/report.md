# Answer quality

Run `matrix-v3` · 8 retrieval configurations × 70 test questions · generator `qwen3.5:4b`, judge `gemma4:12b` (grading rubric `contradiction-only-v5`).

Re-graded from `runs/matrix-v2`: same retrieval and answers; its statement verdicts are reused, and so are grades made under the same rubric.

Configurations were selected on dev in the retrieval ablation; BM25-only baselines are added. Judge scores are best used to compare configurations (agreement with 24 reference labels: faithfulness κ 0.57, correctness κ 0.60 with the v5 rubric; see `runs/judge-calibration-v5/report.md`). Brackets are 95% bootstrap intervals.

| Configuration | R@8 | Correctness | Faithfulness | Citation validity | Abstention accuracy | Relevance | Retrieval ms | Generation ms |
|---|---|---|---|---|---|---|---|---|
| fixed · BM25 · no rerank · contract *(baseline)* | 0.85 | 0.75 [0.64, 0.84] | 0.62 [0.52, 0.73] | 0.87 | 0.94 | 0.97 | 0 | 9320 |
| fixed · BM25+bge-base (weighted a=0.4) · no rerank · contract | 0.84 | 0.74 [0.63, 0.83] | 0.68 [0.58, 0.77] | 0.94 | 0.91 | 0.98 | 17 | 9363 |
| sentence_window · BM25+bge-small (weighted a=0.4) · no rerank · contract | 0.58 | 0.74 [0.64, 0.83] | 0.87 [0.80, 0.93] | 1.00 | 0.94 | 0.98 | 85 | 7031 |
| fixed · BM25+bge-small (weighted a=0.4) · no rerank · contract | 0.84 | 0.73 [0.62, 0.83] | 0.71 [0.61, 0.82] | 0.92 | 0.91 | 0.98 | 15 | 9337 |
| section · BM25 · no rerank · contract *(baseline)* | 0.78 | 0.71 [0.60, 0.81] | 0.85 [0.77, 0.91] | 0.97 | 0.91 | 0.95 | 0 | 5647 |
| fixed · BM25 · minilm · contract | 0.80 | 0.70 [0.59, 0.80] | 0.55 [0.43, 0.66] | 0.86 | 0.91 | 0.97 | 80 | 10136 |
| sentence_window · BM25 · no rerank · contract *(baseline)* | 0.52 | 0.67 [0.57, 0.77] | 0.83 [0.75, 0.91] | 0.96 | 0.86 | 0.98 | 0 | 6828 |
| section · BM25+bge-base (RRF) · no rerank · contract | 0.74 | 0.65 [0.54, 0.76] | 0.86 [0.78, 0.92] | 0.98 | 0.89 | 0.96 | 32 | 5607 |

## Against the BM25 baseline (paired, same questions)

| Candidate | Metric | Δ | 95% CI | Significant |
|---|---|---|---|---|
| fixed · BM25 · minilm · contract | answer_correctness | -0.042 | [-0.144, +0.051] | no |
| fixed · BM25 · minilm · contract | faithfulness | -0.083 | [-0.201, +0.030] | no |
| fixed · BM25+bge-small (weighted a=0.4) · no rerank · contract | answer_correctness | -0.017 | [-0.085, +0.051] | no |
| fixed · BM25+bge-small (weighted a=0.4) · no rerank · contract | faithfulness | +0.090 | [+0.025, +0.163] | yes |
| fixed · BM25+bge-base (weighted a=0.4) · no rerank · contract | answer_correctness | -0.009 | [-0.076, +0.059] | no |
| fixed · BM25+bge-base (weighted a=0.4) · no rerank · contract | faithfulness | +0.047 | [-0.047, +0.145] | no |
| sentence_window · BM25+bge-small (weighted a=0.4) · no rerank · contract | answer_correctness | +0.068 | [+0.000, +0.136] | no |
| sentence_window · BM25+bge-small (weighted a=0.4) · no rerank · contract | faithfulness | +0.046 | [-0.037, +0.122] | no |
| section · BM25+bge-base (RRF) · no rerank · contract | answer_correctness | -0.059 | [-0.152, +0.034] | no |
| section · BM25+bge-base (RRF) · no rerank · contract | faithfulness | +0.012 | [-0.068, +0.097] | no |
