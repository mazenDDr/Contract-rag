# Retrieval ablations

Run `ablation-v1` · 85 answerable questions (26 dev / 59 test) · 189 configurations · 16 min on an Apple M4 Pro.

Recall is *evidence recall*: the share of lawyer-highlighted evidence spans with a relevant chunk in the top k. R@8 is what the generator receives. MRR is 1/rank of the first relevant chunk. Brackets are 95% bootstrap intervals over questions. Configurations are **selected on dev and reported on test**.

## Configurations selected on dev

| # | Configuration | dev R@8 | dev MRR | test R@1 | test R@5 | test R@8 | test MRR | est. ms/query | ties on dev |
|---|---|---|---|---|---|---|---|---|---|
| 1 | fixed · BM25 · no rerank · contract | 0.90 | 0.79 | 0.46 | 0.77 | 0.85 [0.76, 0.91] | 0.73 [0.64, 0.82] | 0 | 0 |
| 2 | fixed · BM25 · minilm · contract | 0.90 | 0.70 | 0.44 | 0.72 | 0.80 [0.72, 0.88] | 0.72 [0.62, 0.81] | 176 | 9 |
| 3 | fixed · BM25+bge-small (weighted a=0.4) · no rerank · contract | 0.90 | 0.81 | 0.45 | 0.79 | 0.84 [0.75, 0.91] | 0.71 [0.62, 0.81] | 15 | 0 |
| 4 | fixed · BM25+bge-base (weighted a=0.4) · no rerank · contract | 0.90 | 0.80 | 0.40 | 0.79 | 0.84 [0.76, 0.91] | 0.69 [0.59, 0.78] | 20 | 0 |
| 5 | fixed · BM25+bge-small (RRF) · no rerank · contract | 0.90 | 0.77 | 0.36 | 0.77 | 0.81 [0.73, 0.89] | 0.64 [0.55, 0.74] | 15 | 0 |
| 6 | fixed · BM25+e5-base (weighted a=0.4) · no rerank · contract | 0.87 | 0.81 | 0.44 | 0.79 | 0.84 [0.76, 0.91] | 0.72 [0.63, 0.81] | 22 | 0 |
| 7 | fixed · BM25+e5-base (RRF) · no rerank · contract | 0.87 | 0.73 | 0.39 | 0.77 | 0.85 [0.78, 0.92] | 0.67 [0.58, 0.76] | 22 | 0 |
| 8 | fixed · BM25+bge-base (RRF) · no rerank · contract | 0.86 | 0.76 | 0.41 | 0.74 | 0.85 [0.77, 0.91] | 0.70 [0.60, 0.79] | 20 | 0 |

*Ties on dev*: configurations with the same recall and MRR as this one on every dev question. They are folded into it rather than taking a slot.

## Candidate pool vs contract size

- **fixed**: median 13 chunks per contract; 90% of contracts fit within the 50 candidates a reranker scores
- **sentence_window**: median 148 chunks per contract; 15% of contracts fit within the 50 candidates a reranker scores
- **section**: median 39 chunks per contract; 62% of contracts fit within the 50 candidates a reranker scores

When the whole contract fits in the candidate pool, a contract-scoped reranker scores every chunk and its output no longer depends on which retriever produced the candidates.

## What each component contributes (test split, paired difference, 95% CI)

| Comparison | Metric | Δ (candidate − baseline) | 95% CI | Significant |
|---|---|---|---|---|
| fixed: best dense vs BM25 | mrr | -0.243 | [-0.336, -0.151] | yes |
| fixed: best hybrid vs BM25 | mrr | -0.020 | [-0.046, +0.004] | no |
| fixed: best dense vs BM25 | context_recall | -0.118 | [-0.214, -0.018] | yes |
| fixed: best hybrid vs BM25 | context_recall | -0.009 | [-0.062, +0.045] | no |
| fixed: + minilm on best hybrid | mrr | +0.006 | [-0.076, +0.088] | no |
| fixed: + minilm on best hybrid | context_recall | -0.049 | [-0.100, -0.009] | yes |
| fixed: + bge-reranker-base on best hybrid | mrr | -0.148 | [-0.245, -0.058] | yes |
| fixed: + bge-reranker-base on best hybrid | context_recall | -0.038 | [-0.122, +0.046] | no |
| fixed: drop contract name (fixed · BM25 · no rerank · contract) | mrr | +0.396 | [+0.287, +0.501] | yes |
| fixed: drop contract name (fixed · e5-base · no rerank · contract) | mrr | +0.083 | [-0.035, +0.199] | no |
| fixed: drop contract name (fixed · BM25+e5-base (RRF) · no rerank · contract) | mrr | +0.295 | [+0.179, +0.410] | yes |
| sentence_window: best dense vs BM25 | mrr | +0.012 | [-0.099, +0.118] | no |
| sentence_window: best hybrid vs BM25 | mrr | +0.002 | [-0.050, +0.052] | no |
| sentence_window: best dense vs BM25 | context_recall | +0.016 | [-0.086, +0.117] | no |
| sentence_window: best hybrid vs BM25 | context_recall | +0.060 | [-0.004, +0.131] | no |
| sentence_window: + minilm on best hybrid | mrr | +0.026 | [-0.049, +0.100] | no |
| sentence_window: + minilm on best hybrid | context_recall | -0.042 | [-0.119, +0.033] | no |
| sentence_window: + bge-reranker-base on best hybrid | mrr | -0.113 | [-0.207, -0.023] | yes |
| sentence_window: + bge-reranker-base on best hybrid | context_recall | -0.167 | [-0.259, -0.084] | yes |
| sentence_window: drop contract name (sentence_window · BM25 · no rerank · contract) | mrr | +0.401 | [+0.291, +0.504] | yes |
| sentence_window: drop contract name (sentence_window · e5-base · no rerank · contract) | mrr | +0.338 | [+0.202, +0.467] | yes |
| sentence_window: drop contract name (sentence_window · BM25+e5-base (RRF) · no rerank · contract) | mrr | +0.426 | [+0.303, +0.543] | yes |
| section: best dense vs BM25 | mrr | -0.114 | [-0.243, +0.017] | no |
| section: best hybrid vs BM25 | mrr | +0.005 | [-0.088, +0.096] | no |
| section: best dense vs BM25 | context_recall | -0.168 | [-0.279, -0.064] | yes |
| section: best hybrid vs BM25 | context_recall | -0.038 | [-0.122, +0.047] | no |
| section: + minilm on best hybrid | mrr | -0.080 | [-0.158, -0.004] | yes |
| section: + minilm on best hybrid | context_recall | -0.003 | [-0.093, +0.087] | no |
| section: + bge-reranker-base on best hybrid | mrr | -0.175 | [-0.289, -0.068] | yes |
| section: + bge-reranker-base on best hybrid | context_recall | -0.039 | [-0.138, +0.060] | no |
| section: drop contract name (section · BM25 · no rerank · contract) | mrr | +0.484 | [+0.369, +0.591] | yes |
| section: drop contract name (section · e5-base · no rerank · contract) | mrr | +0.299 | [+0.178, +0.421] | yes |
| section: drop contract name (section · BM25+e5-base (RRF) · no rerank · contract) | mrr | +0.425 | [+0.302, +0.543] | yes |

## Best configuration per chunking strategy (chosen on dev, test scores)

| Chunking | Configuration | test R@1 | test R@8 | test MRR |
|---|---|---|---|---|
| fixed | fixed · BM25 · no rerank · contract | 0.46 | 0.85 [0.76, 0.91] | 0.73 [0.64, 0.82] |
| sentence_window | sentence_window · BM25+bge-small (weighted a=0.4) · no rerank · contract | 0.17 | 0.58 [0.47, 0.68] | 0.46 [0.36, 0.56] |
| section | section · BM25+bge-base (RRF) · no rerank · contract | 0.36 | 0.74 [0.64, 0.82] | 0.68 [0.58, 0.77] |

## Latency per component (median ms per query, uncached calls)

| Component | ms |
|---|---|
| bm25 | 0.1 |
| dense:bge-base | 20.1 |
| dense:bge-small | 14.5 |
| dense:e5-base | 21.4 |
| rerank:bge-reranker-base | 1012.4 |
| rerank:minilm | 175.5 |

## Weighted-fusion alpha (tuned on dev: MRR, then R@8)

| Chunking · model | alpha |
|---|---|
| fixed · bge-small | 0.4 |
| fixed · bge-base | 0.4 |
| fixed · e5-base | 0.4 |
| sentence_window · bge-small | 0.4 |
| sentence_window · bge-base | 0.6 |
| sentence_window · e5-base | 0.4 |
| section · bge-small | 0.2 |
| section · bge-base | 0.4 |
| section · e5-base | 0.2 |

## Answer-key quality per chunking

- **fixed**: aligned 100.0%, uncovered spans 0
- **sentence_window**: aligned 100.0%, uncovered spans 0
- **section**: aligned 100.0%, uncovered spans 0

## Full grid (test split)

| Configuration | R@1 | R@5 | R@8 | MRR | nDCG@10 | est. ms |
|---|---|---|---|---|---|---|
| section · BM25+bge-base (weighted a=0.4) · no rerank · contract | 0.42 | 0.74 | 0.77 | 0.73 | 0.68 | 20 |
| fixed · BM25 · no rerank · contract | 0.46 | 0.77 | 0.85 | 0.73 | 0.71 | 0 |
| fixed · BM25+e5-base (weighted a=0.4) · no rerank · contract | 0.44 | 0.79 | 0.84 | 0.72 | 0.70 | 22 |
| fixed · BM25 · minilm · contract | 0.44 | 0.72 | 0.80 | 0.72 | 0.67 | 176 |
| fixed · bge-base · minilm · contract | 0.44 | 0.72 | 0.80 | 0.72 | 0.67 | 196 |
| fixed · e5-base · minilm · contract | 0.44 | 0.72 | 0.80 | 0.72 | 0.67 | 197 |
| fixed · BM25+bge-base (RRF) · minilm · contract | 0.44 | 0.72 | 0.80 | 0.72 | 0.67 | 196 |
| fixed · BM25+bge-base (weighted a=0.4) · minilm · contract | 0.44 | 0.72 | 0.80 | 0.72 | 0.67 | 196 |
| fixed · BM25+e5-base (RRF) · minilm · contract | 0.44 | 0.72 | 0.80 | 0.72 | 0.67 | 197 |
| fixed · BM25+e5-base (weighted a=0.4) · minilm · contract | 0.44 | 0.72 | 0.80 | 0.72 | 0.67 | 197 |
| fixed · bge-small · minilm · contract | 0.44 | 0.72 | 0.79 | 0.72 | 0.66 | 190 |
| fixed · BM25+bge-small (RRF) · minilm · contract | 0.44 | 0.72 | 0.79 | 0.72 | 0.66 | 190 |
| fixed · BM25+bge-small (weighted a=0.4) · minilm · contract | 0.44 | 0.72 | 0.79 | 0.72 | 0.66 | 190 |
| fixed · BM25+bge-small (weighted a=0.4) · no rerank · contract | 0.45 | 0.79 | 0.84 | 0.71 | 0.70 | 15 |
| section · BM25+e5-base (weighted a=0.2) · no rerank · contract | 0.41 | 0.69 | 0.82 | 0.70 | 0.68 | 22 |
| section · BM25+bge-small (weighted a=0.2) · no rerank · contract | 0.40 | 0.69 | 0.78 | 0.70 | 0.66 | 15 |
| fixed · BM25+bge-base (RRF) · no rerank · contract | 0.41 | 0.74 | 0.85 | 0.70 | 0.66 | 20 |
| fixed · BM25+bge-base (weighted a=0.4) · no rerank · contract | 0.40 | 0.79 | 0.84 | 0.69 | 0.68 | 20 |
| section · BM25+bge-base (RRF) · no rerank · contract | 0.36 | 0.68 | 0.74 | 0.68 | 0.63 | 20 |
| section · BM25 · no rerank · contract | 0.39 | 0.66 | 0.78 | 0.67 | 0.65 | 0 |
| fixed · BM25+e5-base (RRF) · no rerank · contract | 0.39 | 0.77 | 0.85 | 0.67 | 0.67 | 22 |
| fixed · BM25+bge-small (RRF) · no rerank · contract | 0.36 | 0.77 | 0.81 | 0.64 | 0.64 | 15 |
| section · BM25+bge-small (RRF) · no rerank · contract | 0.30 | 0.69 | 0.78 | 0.64 | 0.60 | 15 |
| section · BM25+e5-base (RRF) · no rerank · contract | 0.28 | 0.69 | 0.80 | 0.64 | 0.62 | 22 |
| section · BM25 · minilm · contract | 0.30 | 0.68 | 0.73 | 0.60 | 0.58 | 176 |
| section · bge-small · minilm · contract | 0.29 | 0.68 | 0.73 | 0.60 | 0.58 | 190 |
| section · bge-base · minilm · contract | 0.29 | 0.68 | 0.73 | 0.60 | 0.58 | 196 |
| section · e5-base · minilm · contract | 0.29 | 0.68 | 0.73 | 0.60 | 0.58 | 197 |
| section · BM25+bge-small (RRF) · minilm · contract | 0.29 | 0.68 | 0.73 | 0.60 | 0.58 | 190 |
| section · BM25+bge-small (weighted a=0.2) · minilm · contract | 0.29 | 0.68 | 0.73 | 0.60 | 0.58 | 190 |
| section · BM25+bge-base (RRF) · minilm · contract | 0.29 | 0.68 | 0.73 | 0.60 | 0.58 | 196 |
| section · BM25+bge-base (weighted a=0.4) · minilm · contract | 0.29 | 0.68 | 0.73 | 0.60 | 0.58 | 196 |
| section · BM25+e5-base (RRF) · minilm · contract | 0.29 | 0.68 | 0.73 | 0.60 | 0.58 | 197 |
| section · BM25+e5-base (weighted a=0.2) · minilm · contract | 0.29 | 0.68 | 0.73 | 0.60 | 0.58 | 197 |
| fixed · bge-base · bge-reranker-base · contract | 0.29 | 0.70 | 0.80 | 0.56 | 0.58 | 1032 |
| fixed · BM25 · bge-reranker-base · contract | 0.29 | 0.70 | 0.79 | 0.56 | 0.58 | 1012 |
| fixed · bge-small · bge-reranker-base · contract | 0.29 | 0.70 | 0.80 | 0.56 | 0.58 | 1027 |
| fixed · e5-base · bge-reranker-base · contract | 0.29 | 0.70 | 0.79 | 0.56 | 0.58 | 1034 |
| fixed · BM25+bge-small (RRF) · bge-reranker-base · contract | 0.29 | 0.70 | 0.80 | 0.56 | 0.58 | 1027 |
| fixed · BM25+bge-small (weighted a=0.4) · bge-reranker-base · contract | 0.29 | 0.70 | 0.80 | 0.56 | 0.58 | 1027 |
| fixed · BM25+bge-base (RRF) · bge-reranker-base · contract | 0.29 | 0.70 | 0.80 | 0.56 | 0.58 | 1033 |
| fixed · BM25+bge-base (weighted a=0.4) · bge-reranker-base · contract | 0.29 | 0.70 | 0.80 | 0.56 | 0.58 | 1033 |
| fixed · BM25+e5-base (RRF) · bge-reranker-base · contract | 0.29 | 0.70 | 0.79 | 0.56 | 0.58 | 1034 |
| fixed · BM25+e5-base (weighted a=0.4) · bge-reranker-base · contract | 0.29 | 0.70 | 0.79 | 0.56 | 0.58 | 1034 |
| section · bge-base · no rerank · contract | 0.28 | 0.55 | 0.61 | 0.56 | 0.50 | 20 |
| section · bge-small · no rerank · contract | 0.28 | 0.54 | 0.62 | 0.55 | 0.50 | 14 |
| sentence_window · BM25+e5-base (weighted a=0.4) · no rerank · contract | 0.23 | 0.52 | 0.60 | 0.54 | 0.48 | 22 |
| section · e5-base · no rerank · contract | 0.27 | 0.53 | 0.62 | 0.53 | 0.50 | 21 |
| sentence_window · BM25+bge-base (RRF) · no rerank · contract | 0.22 | 0.54 | 0.60 | 0.52 | 0.47 | 20 |
| sentence_window · BM25+e5-base (RRF) · no rerank · contract | 0.21 | 0.54 | 0.60 | 0.51 | 0.47 | 22 |
| section · bge-base · bge-reranker-base · contract | 0.26 | 0.59 | 0.70 | 0.51 | 0.52 | 1032 |
| sentence_window · BM25+bge-base (weighted a=0.6) · no rerank · contract | 0.21 | 0.51 | 0.58 | 0.51 | 0.47 | 20 |
| section · BM25 · bge-reranker-base · contract | 0.26 | 0.61 | 0.70 | 0.50 | 0.52 | 1012 |
| section · BM25+e5-base (RRF) · bge-reranker-base · contract | 0.26 | 0.59 | 0.70 | 0.50 | 0.51 | 1034 |
| section · e5-base · bge-reranker-base · contract | 0.26 | 0.59 | 0.71 | 0.50 | 0.52 | 1034 |
| section · bge-small · bge-reranker-base · contract | 0.26 | 0.59 | 0.70 | 0.50 | 0.51 | 1027 |
| section · BM25+bge-base (RRF) · bge-reranker-base · contract | 0.26 | 0.59 | 0.70 | 0.50 | 0.51 | 1033 |
| section · BM25+bge-base (weighted a=0.4) · bge-reranker-base · contract | 0.26 | 0.59 | 0.70 | 0.50 | 0.51 | 1033 |
| fixed · bge-base · no rerank · contract | 0.26 | 0.58 | 0.72 | 0.50 | 0.50 | 20 |
| section · BM25+bge-small (RRF) · bge-reranker-base · contract | 0.26 | 0.59 | 0.70 | 0.50 | 0.51 | 1027 |
| section · BM25+bge-small (weighted a=0.2) · bge-reranker-base · contract | 0.26 | 0.59 | 0.70 | 0.50 | 0.51 | 1027 |
| section · BM25+e5-base (weighted a=0.2) · bge-reranker-base · contract | 0.26 | 0.59 | 0.70 | 0.50 | 0.51 | 1034 |
| sentence_window · bge-base · minilm · contract | 0.24 | 0.48 | 0.54 | 0.49 | 0.44 | 196 |
| sentence_window · BM25+bge-base (weighted a=0.6) · minilm · contract | 0.24 | 0.46 | 0.54 | 0.49 | 0.44 | 196 |
| sentence_window · BM25+bge-small (RRF) · minilm · contract | 0.24 | 0.46 | 0.54 | 0.49 | 0.44 | 190 |
| sentence_window · BM25+bge-small (weighted a=0.4) · minilm · contract | 0.24 | 0.46 | 0.54 | 0.49 | 0.44 | 190 |
| sentence_window · e5-base · minilm · contract | 0.24 | 0.46 | 0.53 | 0.49 | 0.44 | 197 |
| sentence_window · bge-small · minilm · contract | 0.24 | 0.47 | 0.53 | 0.49 | 0.44 | 190 |
| sentence_window · BM25+e5-base (RRF) · minilm · contract | 0.24 | 0.46 | 0.54 | 0.49 | 0.44 | 197 |
| sentence_window · BM25+e5-base (weighted a=0.4) · minilm · contract | 0.24 | 0.46 | 0.54 | 0.49 | 0.44 | 197 |
| fixed · bge-small · no rerank · contract | 0.22 | 0.57 | 0.73 | 0.49 | 0.50 | 14 |
| sentence_window · BM25+bge-base (RRF) · minilm · contract | 0.24 | 0.45 | 0.52 | 0.49 | 0.43 | 196 |
| sentence_window · BM25 · minilm · contract | 0.24 | 0.45 | 0.52 | 0.49 | 0.43 | 176 |
| sentence_window · e5-base · no rerank · contract | 0.24 | 0.46 | 0.51 | 0.48 | 0.42 | 21 |
| sentence_window · bge-base · no rerank · contract | 0.21 | 0.45 | 0.53 | 0.47 | 0.43 | 20 |
| sentence_window · BM25+bge-small (RRF) · no rerank · contract | 0.19 | 0.49 | 0.58 | 0.47 | 0.45 | 15 |
| fixed · e5-base · no rerank · contract | 0.21 | 0.62 | 0.74 | 0.47 | 0.50 | 21 |
| sentence_window · BM25+bge-small (weighted a=0.4) · no rerank · contract | 0.17 | 0.48 | 0.58 | 0.46 | 0.44 | 15 |
| sentence_window · BM25 · no rerank · contract | 0.18 | 0.41 | 0.52 | 0.46 | 0.41 | 0 |
| fixed · e5-base · no rerank · contract · name kept in query | 0.15 | 0.54 | 0.77 | 0.39 | 0.44 | 21 |
| sentence_window · bge-small · no rerank · contract | 0.16 | 0.40 | 0.45 | 0.39 | 0.36 | 14 |
| fixed · BM25+e5-base (RRF) · no rerank · contract · name kept in query | 0.13 | 0.58 | 0.79 | 0.37 | 0.45 | 22 |
| sentence_window · bge-base · bge-reranker-base · contract | 0.17 | 0.37 | 0.41 | 0.37 | 0.34 | 1032 |
| sentence_window · BM25 · bge-reranker-base · contract | 0.17 | 0.38 | 0.41 | 0.37 | 0.34 | 1012 |
| sentence_window · BM25+bge-base (weighted a=0.6) · bge-reranker-base · contract | 0.17 | 0.37 | 0.41 | 0.37 | 0.33 | 1033 |
| sentence_window · e5-base · bge-reranker-base · contract | 0.16 | 0.36 | 0.41 | 0.36 | 0.33 | 1034 |
| sentence_window · BM25+e5-base (weighted a=0.4) · bge-reranker-base · contract | 0.16 | 0.35 | 0.41 | 0.36 | 0.33 | 1034 |
| sentence_window · BM25+e5-base (RRF) · bge-reranker-base · contract | 0.16 | 0.35 | 0.41 | 0.36 | 0.33 | 1034 |
| sentence_window · bge-small · bge-reranker-base · contract | 0.16 | 0.37 | 0.41 | 0.36 | 0.33 | 1027 |
| sentence_window · BM25+bge-small (weighted a=0.4) · bge-reranker-base · contract | 0.16 | 0.37 | 0.41 | 0.35 | 0.33 | 1027 |
| sentence_window · BM25+bge-small (RRF) · bge-reranker-base · contract | 0.16 | 0.37 | 0.41 | 0.35 | 0.33 | 1027 |
| sentence_window · BM25+bge-base (RRF) · bge-reranker-base · contract | 0.15 | 0.36 | 0.39 | 0.35 | 0.32 | 1033 |
| fixed · BM25 · no rerank · contract · name kept in query | 0.11 | 0.48 | 0.69 | 0.34 | 0.40 | 0 |
| section · e5-base · no rerank · contract · name kept in query | 0.06 | 0.31 | 0.56 | 0.23 | 0.30 | 21 |
| section · BM25+e5-base (RRF) · no rerank · contract · name kept in query | 0.05 | 0.33 | 0.55 | 0.22 | 0.29 | 22 |
| section · BM25 · no rerank · contract · name kept in query | 0.05 | 0.29 | 0.44 | 0.19 | 0.24 | 0 |
| sentence_window · e5-base · no rerank · contract · name kept in query | 0.03 | 0.18 | 0.30 | 0.14 | 0.16 | 21 |
| sentence_window · BM25+e5-base (RRF) · no rerank · contract · name kept in query | 0.03 | 0.10 | 0.19 | 0.09 | 0.11 | 22 |
| sentence_window · BM25 · no rerank · contract · name kept in query | 0.02 | 0.06 | 0.18 | 0.06 | 0.09 | 0 |
| fixed · bge-base · bge-reranker-base · corpus | 0.12 | 0.46 | 0.60 | 0.34 | 0.37 | 1032 |
| fixed · BM25+bge-base (weighted a=0.4) · bge-reranker-base · corpus | 0.12 | 0.41 | 0.62 | 0.34 | 0.37 | 1033 |
| fixed · BM25+bge-base (RRF) · bge-reranker-base · corpus | 0.12 | 0.42 | 0.61 | 0.34 | 0.37 | 1033 |
| fixed · e5-base · no rerank · corpus | 0.15 | 0.43 | 0.56 | 0.34 | 0.35 | 21 |
| fixed · BM25+e5-base (RRF) · bge-reranker-base · corpus | 0.12 | 0.41 | 0.61 | 0.34 | 0.37 | 1034 |
| fixed · BM25+e5-base (weighted a=0.4) · bge-reranker-base · corpus | 0.12 | 0.41 | 0.61 | 0.34 | 0.37 | 1034 |
| fixed · BM25+bge-small (RRF) · bge-reranker-base · corpus | 0.12 | 0.43 | 0.61 | 0.33 | 0.37 | 1027 |
| fixed · BM25+bge-small (weighted a=0.4) · bge-reranker-base · corpus | 0.12 | 0.42 | 0.61 | 0.33 | 0.37 | 1027 |
| fixed · BM25+e5-base (weighted a=0.4) · no rerank · corpus | 0.13 | 0.41 | 0.60 | 0.33 | 0.37 | 22 |
| fixed · e5-base · bge-reranker-base · corpus | 0.12 | 0.43 | 0.59 | 0.33 | 0.36 | 1034 |
| fixed · bge-small · bge-reranker-base · corpus | 0.12 | 0.42 | 0.57 | 0.33 | 0.36 | 1027 |
| fixed · BM25 · bge-reranker-base · corpus | 0.12 | 0.40 | 0.57 | 0.33 | 0.36 | 1012 |
| fixed · BM25+e5-base (RRF) · no rerank · corpus | 0.13 | 0.41 | 0.59 | 0.32 | 0.36 | 22 |
| fixed · bge-base · minilm · corpus | 0.13 | 0.44 | 0.62 | 0.32 | 0.37 | 196 |
| fixed · bge-base · no rerank · corpus | 0.14 | 0.36 | 0.55 | 0.32 | 0.34 | 20 |
| fixed · BM25+bge-base (weighted a=0.4) · no rerank · corpus | 0.13 | 0.39 | 0.56 | 0.32 | 0.35 | 20 |
| fixed · e5-base · minilm · corpus | 0.13 | 0.44 | 0.60 | 0.32 | 0.36 | 197 |
| fixed · BM25+bge-base (RRF) · no rerank · corpus | 0.13 | 0.40 | 0.59 | 0.32 | 0.36 | 20 |
| fixed · bge-small · no rerank · corpus | 0.13 | 0.40 | 0.52 | 0.32 | 0.33 | 14 |
| fixed · BM25+bge-base (RRF) · minilm · corpus | 0.13 | 0.44 | 0.58 | 0.32 | 0.36 | 196 |
| fixed · BM25+bge-base (weighted a=0.4) · minilm · corpus | 0.13 | 0.44 | 0.59 | 0.32 | 0.36 | 196 |
| fixed · BM25+e5-base (RRF) · minilm · corpus | 0.13 | 0.44 | 0.59 | 0.32 | 0.36 | 197 |
| fixed · BM25+e5-base (weighted a=0.4) · minilm · corpus | 0.13 | 0.44 | 0.59 | 0.32 | 0.36 | 197 |
| fixed · BM25 · minilm · corpus | 0.13 | 0.44 | 0.59 | 0.32 | 0.36 | 176 |
| fixed · BM25+bge-small (weighted a=0.4) · minilm · corpus | 0.13 | 0.44 | 0.59 | 0.32 | 0.36 | 190 |
| fixed · BM25+bge-small (RRF) · minilm · corpus | 0.13 | 0.44 | 0.59 | 0.31 | 0.36 | 190 |
| fixed · bge-small · minilm · corpus | 0.13 | 0.42 | 0.59 | 0.31 | 0.36 | 190 |
| fixed · BM25+bge-small (weighted a=0.4) · no rerank · corpus | 0.11 | 0.38 | 0.53 | 0.31 | 0.33 | 15 |
| fixed · BM25+bge-small (RRF) · no rerank · corpus | 0.11 | 0.37 | 0.55 | 0.29 | 0.33 | 15 |
| fixed · BM25 · no rerank · corpus | 0.11 | 0.33 | 0.49 | 0.29 | 0.31 | 0 |
| section · bge-base · bge-reranker-base · corpus | 0.08 | 0.28 | 0.40 | 0.22 | 0.25 | 1032 |
| section · BM25+bge-base (RRF) · bge-reranker-base · corpus | 0.08 | 0.29 | 0.42 | 0.21 | 0.25 | 1033 |
| section · BM25+bge-base (weighted a=0.4) · bge-reranker-base · corpus | 0.08 | 0.29 | 0.42 | 0.21 | 0.25 | 1033 |
| section · BM25+bge-base (RRF) · minilm · corpus | 0.08 | 0.32 | 0.48 | 0.21 | 0.27 | 196 |
| section · BM25 · bge-reranker-base · corpus | 0.08 | 0.28 | 0.39 | 0.21 | 0.23 | 1012 |
| section · BM25+bge-base (weighted a=0.4) · minilm · corpus | 0.08 | 0.32 | 0.46 | 0.21 | 0.27 | 196 |
| section · e5-base · bge-reranker-base · corpus | 0.08 | 0.27 | 0.37 | 0.21 | 0.23 | 1034 |
| section · bge-small · bge-reranker-base · corpus | 0.08 | 0.25 | 0.36 | 0.21 | 0.23 | 1027 |
| section · bge-base · minilm · corpus | 0.08 | 0.33 | 0.44 | 0.21 | 0.26 | 196 |
| section · BM25 · minilm · corpus | 0.08 | 0.34 | 0.44 | 0.21 | 0.25 | 176 |
| section · BM25+bge-small (weighted a=0.2) · bge-reranker-base · corpus | 0.08 | 0.27 | 0.38 | 0.21 | 0.23 | 1027 |
| section · BM25+bge-small (RRF) · bge-reranker-base · corpus | 0.08 | 0.27 | 0.37 | 0.21 | 0.23 | 1027 |
| section · BM25+e5-base (RRF) · bge-reranker-base · corpus | 0.08 | 0.27 | 0.38 | 0.21 | 0.24 | 1034 |
| section · BM25+e5-base (weighted a=0.2) · bge-reranker-base · corpus | 0.08 | 0.27 | 0.39 | 0.21 | 0.24 | 1034 |
| section · BM25+e5-base (weighted a=0.2) · minilm · corpus | 0.08 | 0.30 | 0.44 | 0.21 | 0.26 | 197 |
| section · BM25+bge-small (weighted a=0.2) · minilm · corpus | 0.08 | 0.32 | 0.43 | 0.21 | 0.25 | 190 |
| section · BM25+bge-small (RRF) · minilm · corpus | 0.08 | 0.31 | 0.43 | 0.21 | 0.25 | 190 |
| section · BM25+e5-base (RRF) · minilm · corpus | 0.08 | 0.28 | 0.44 | 0.20 | 0.26 | 197 |
| section · bge-small · minilm · corpus | 0.08 | 0.30 | 0.40 | 0.20 | 0.24 | 190 |
| section · e5-base · minilm · corpus | 0.08 | 0.28 | 0.40 | 0.20 | 0.24 | 197 |
| section · BM25+bge-base (RRF) · no rerank · corpus | 0.07 | 0.25 | 0.36 | 0.19 | 0.21 | 20 |
| section · e5-base · no rerank · corpus | 0.06 | 0.20 | 0.32 | 0.19 | 0.20 | 21 |
| section · bge-base · no rerank · corpus | 0.07 | 0.24 | 0.36 | 0.18 | 0.22 | 20 |
| section · bge-small · no rerank · corpus | 0.07 | 0.22 | 0.34 | 0.18 | 0.20 | 14 |
| section · BM25+bge-base (weighted a=0.4) · no rerank · corpus | 0.07 | 0.25 | 0.32 | 0.18 | 0.20 | 20 |
| section · BM25+e5-base (RRF) · no rerank · corpus | 0.05 | 0.25 | 0.35 | 0.16 | 0.20 | 22 |
| section · BM25+bge-small (RRF) · no rerank · corpus | 0.07 | 0.21 | 0.28 | 0.16 | 0.18 | 15 |
| section · BM25+bge-small (weighted a=0.2) · no rerank · corpus | 0.07 | 0.22 | 0.31 | 0.16 | 0.19 | 15 |
| section · BM25+e5-base (weighted a=0.2) · no rerank · corpus | 0.05 | 0.23 | 0.33 | 0.16 | 0.19 | 22 |
| sentence_window · bge-small · bge-reranker-base · corpus | 0.03 | 0.15 | 0.21 | 0.13 | 0.12 | 1027 |
| sentence_window · bge-small · minilm · corpus | 0.04 | 0.15 | 0.20 | 0.13 | 0.13 | 190 |
| sentence_window · bge-base · bge-reranker-base · corpus | 0.03 | 0.16 | 0.21 | 0.13 | 0.12 | 1032 |
| sentence_window · bge-base · minilm · corpus | 0.04 | 0.13 | 0.20 | 0.13 | 0.12 | 196 |
| sentence_window · e5-base · minilm · corpus | 0.04 | 0.11 | 0.18 | 0.13 | 0.12 | 197 |
| section · BM25 · no rerank · corpus | 0.05 | 0.19 | 0.25 | 0.13 | 0.15 | 0 |
| sentence_window · BM25+bge-small (RRF) · bge-reranker-base · corpus | 0.03 | 0.14 | 0.22 | 0.13 | 0.13 | 1027 |
| sentence_window · BM25+e5-base (weighted a=0.4) · minilm · corpus | 0.04 | 0.10 | 0.17 | 0.12 | 0.11 | 197 |
| sentence_window · BM25+bge-small (weighted a=0.4) · bge-reranker-base · corpus | 0.03 | 0.14 | 0.22 | 0.12 | 0.12 | 1027 |
| sentence_window · BM25+e5-base (RRF) · minilm · corpus | 0.04 | 0.11 | 0.17 | 0.12 | 0.11 | 197 |
| sentence_window · BM25+bge-base (RRF) · minilm · corpus | 0.04 | 0.12 | 0.18 | 0.12 | 0.12 | 196 |
| sentence_window · BM25+bge-base (weighted a=0.6) · minilm · corpus | 0.04 | 0.12 | 0.18 | 0.12 | 0.12 | 196 |
| sentence_window · e5-base · bge-reranker-base · corpus | 0.03 | 0.13 | 0.20 | 0.12 | 0.12 | 1034 |
| sentence_window · BM25+bge-small (RRF) · minilm · corpus | 0.04 | 0.12 | 0.16 | 0.12 | 0.11 | 190 |
| sentence_window · BM25+bge-small (weighted a=0.4) · minilm · corpus | 0.04 | 0.11 | 0.16 | 0.12 | 0.10 | 190 |
| sentence_window · BM25+bge-base (weighted a=0.6) · bge-reranker-base · corpus | 0.02 | 0.16 | 0.25 | 0.11 | 0.13 | 1033 |
| sentence_window · BM25+e5-base (RRF) · bge-reranker-base · corpus | 0.02 | 0.12 | 0.21 | 0.11 | 0.11 | 1034 |
| sentence_window · BM25+e5-base (weighted a=0.4) · bge-reranker-base · corpus | 0.02 | 0.13 | 0.21 | 0.11 | 0.11 | 1034 |
| sentence_window · BM25+bge-base (RRF) · bge-reranker-base · corpus | 0.02 | 0.12 | 0.23 | 0.11 | 0.12 | 1033 |
| sentence_window · BM25 · minilm · corpus | 0.04 | 0.10 | 0.14 | 0.10 | 0.09 | 176 |
| sentence_window · e5-base · no rerank · corpus | 0.03 | 0.08 | 0.18 | 0.10 | 0.10 | 21 |
| sentence_window · BM25 · bge-reranker-base · corpus | 0.02 | 0.09 | 0.13 | 0.09 | 0.08 | 1012 |
| sentence_window · BM25+e5-base (RRF) · no rerank · corpus | 0.03 | 0.06 | 0.08 | 0.06 | 0.06 | 22 |
| sentence_window · BM25+e5-base (weighted a=0.4) · no rerank · corpus | 0.02 | 0.06 | 0.07 | 0.05 | 0.05 | 22 |
| sentence_window · BM25+bge-small (RRF) · no rerank · corpus | 0.02 | 0.03 | 0.08 | 0.05 | 0.05 | 15 |
| sentence_window · BM25+bge-base (RRF) · no rerank · corpus | 0.02 | 0.03 | 0.07 | 0.04 | 0.04 | 20 |
| sentence_window · bge-small · no rerank · corpus | 0.02 | 0.05 | 0.07 | 0.04 | 0.04 | 14 |
| sentence_window · bge-base · no rerank · corpus | 0.02 | 0.04 | 0.07 | 0.04 | 0.04 | 20 |
| sentence_window · BM25+bge-small (weighted a=0.4) · no rerank · corpus | 0.02 | 0.05 | 0.08 | 0.04 | 0.05 | 15 |
| sentence_window · BM25 · no rerank · corpus | 0.02 | 0.04 | 0.08 | 0.03 | 0.04 | 0 |
| sentence_window · BM25+bge-base (weighted a=0.6) · no rerank · corpus | 0.02 | 0.03 | 0.05 | 0.03 | 0.03 | 20 |
