# Answer quality

Run `matrix-deploy-0.8b` · 1 retrieval configurations × 70 test questions · generator `qwen3.5:0.8b`, judge `gemma4:12b` (grading rubric `contradiction-only-v5`).

Configurations were selected on dev in the retrieval ablation; BM25-only baselines are added. Judge scores are best used to compare configurations (agreement with 24 reference labels: faithfulness κ 0.57, correctness κ 0.60 with the v5 rubric; see `runs/judge-calibration-v5/report.md`). Brackets are 95% bootstrap intervals.

| Configuration | R@8 | Correctness | Faithfulness | Citation validity | Abstention accuracy | Relevance | Retrieval ms | Generation ms |
|---|---|---|---|---|---|---|---|---|
| fixed · BM25 · no rerank · contract *(baseline)* | 0.85 | 0.32 [0.22, 0.42] | 0.03 [0.00, 0.09] | 0.06 | 0.80 | 0.97 | 0 | 1769 |

## Against the BM25 baseline (paired, same questions)

| Candidate | Metric | Δ | 95% CI | Significant |
|---|---|---|---|---|
