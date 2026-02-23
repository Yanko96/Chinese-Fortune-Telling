# Multi-hop RAG Benchmark Report

生成时间：2026-02-22 21:19

数据集：benchmarks/qa_multihop.json

样本数：1

## 汇总对比

| Config | chain_mean | hop_ok≥0.6 | full_ok=1.0 | cross_hit | p50 lat |
|--------|-----------|-----------|------------|----------|---------|
| hyde_rerank_topn7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0s |

## chain_score 按推理步骤数细分

| Config | 3hop |
|--------|--------|
| hyde_rerank_topn7 | 0.0 |

## chain_score 按书籍对细分

| Config | 三命通会×子平真诠 |
|--------|--------|
| hyde_rerank_topn7 | 0.0 |

## 逐题得分（hyde_rerank_topn7）

| # | chain | hops | books | question (truncated) |
|---|-------|------|-------|----------------------|
| 1 | 0.0 | 3 | 三命通会×子平真诠 | 根据《三命通会》中关于正财的论述和《子平真诠》中关于星辰与格局的讨论，如何理解在… |