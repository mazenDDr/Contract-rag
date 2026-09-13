# Answer quality

Run `matrix-v1` · 8 retrieval configurations × 70 test questions · generator `qwen3.5:4b`, judge `gemma4:12b` · 51 min.

Configurations were selected on dev in the retrieval ablation; BM25-only baselines are added. Judge scores are best used to compare configurations (calibration: faithfulness κ 0.57 against reference labels). Brackets are 95% bootstrap intervals.

| Configuration | R@8 | Correctness | Faithfulness | Citation validity | Abstention accuracy | Relevance | Retrieval ms | Generation ms |
|---|---|---|---|---|---|---|---|---|
| sentence_window · BM25+bge-small (weighted a=0.4) · no rerank · contract | 0.58 | 0.63 [0.53, 0.73] | 0.53 [0.42, 0.62] | 1.00 | 0.93 | 0.98 | 85 | 7031 |
| fixed · BM25 · no rerank · contract *(baseline)* | 0.85 | 0.61 [0.50, 0.70] | 0.61 [0.50, 0.71] | 0.87 | 0.91 | 0.98 | 0 | 9320 |
| fixed · BM25 · minilm · contract | 0.80 | 0.61 [0.50, 0.70] | 0.54 [0.42, 0.66] | 0.86 | 0.89 | 0.97 | 80 | 10136 |
| fixed · BM25+bge-base (weighted a=0.4) · no rerank · contract | 0.84 | 0.61 [0.50, 0.70] | 0.66 [0.56, 0.77] | 0.94 | 0.89 | 0.99 | 17 | 9363 |
| fixed · BM25+bge-small (weighted a=0.4) · no rerank · contract | 0.84 | 0.59 [0.48, 0.69] | 0.70 [0.59, 0.80] | 0.92 | 0.89 | 0.98 | 15 | 9337 |
| section · BM25 · no rerank · contract *(baseline)* | 0.78 | 0.58 [0.47, 0.68] | 0.84 [0.76, 0.91] | 0.97 | 0.91 | 0.97 | 0 | 5647 |
| section · BM25+bge-base (RRF) · no rerank · contract | 0.74 | 0.53 [0.43, 0.64] | 0.88 [0.82, 0.94] | 0.98 | 0.86 | 0.98 | 32 | 5607 |
| sentence_window · BM25 · no rerank · contract *(baseline)* | 0.52 | 0.51 [0.40, 0.62] | 0.46 [0.36, 0.56] | 0.96 | 0.86 | 0.99 | 0 | 6828 |

## Against the BM25 baseline (paired, same questions)

| Candidate | Metric | Δ | 95% CI | Significant |
|---|---|---|---|---|
| fixed · BM25 · minilm · contract | answer_correctness | +0.000 | [-0.085, +0.076] | no |
| fixed · BM25 · minilm · contract | faithfulness | -0.073 | [-0.194, +0.046] | no |
| fixed · BM25+bge-small (weighted a=0.4) · no rerank · contract | answer_correctness | -0.017 | [-0.085, +0.059] | no |
| fixed · BM25+bge-small (weighted a=0.4) · no rerank · contract | faithfulness | +0.109 | [+0.045, +0.180] | yes |
| fixed · BM25+bge-base (weighted a=0.4) · no rerank · contract | answer_correctness | +0.000 | [-0.076, +0.076] | no |
| fixed · BM25+bge-base (weighted a=0.4) · no rerank · contract | faithfulness | +0.061 | [-0.050, +0.168] | no |
| sentence_window · BM25+bge-small (weighted a=0.4) · no rerank · contract | answer_correctness | +0.119 | [+0.034, +0.212] | yes |
| sentence_window · BM25+bge-small (weighted a=0.4) · no rerank · contract | faithfulness | +0.057 | [-0.040, +0.160] | no |
| section · BM25+bge-base (RRF) · no rerank · contract | answer_correctness | -0.051 | [-0.127, +0.025] | no |
| section · BM25+bge-base (RRF) · no rerank · contract | faithfulness | +0.007 | [-0.087, +0.102] | no |
