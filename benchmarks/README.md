# RAG Benchmark

本目录存放测试数据集与评测结果，对应两个脚本：

| 脚本 | 作用 |
|------|------|
| [scripts/generate_qa_dataset.py](../scripts/generate_qa_dataset.py) | 从 Chroma 语料合成带黄金答案的问答集 |
| [scripts/rag_bench.py](../scripts/rag_bench.py) | 批量跑多配置并输出 RAGAS 指标 |

---

## 快速开始

### 1. 安装依赖

```bash
# 在 api/ 虚拟环境里追加安装
pip install -r benchmarks/requirements.txt
```

### 2. 生成问答数据集

```bash
cd e:\repos\Chinese-Fortune-Telling
python scripts/generate_qa_dataset.py \
    --num-samples 50 \
    --output benchmarks/qa_dataset.json \
    --chroma-dir ./chroma_db
```

生成文件格式：
```json
[
  {
    "id": "uuid",
    "question": "八字中的日主是什么意思？",
    "golden_answer": "日主是...",
    "source_chunks": ["..."],
    "metadata": { "source": "san_ming_tong_hui.pdf", "page": 12 }
  }
]
```

### 3. 跑 Baseline

```bash
python scripts/rag_bench.py \
    --config configs/rag/baseline.yaml \
    --dataset benchmarks/qa_dataset.json \
    --output-dir benchmarks/results
```

### 4. 对比所有配置

```bash
python scripts/rag_bench.py \
    --configs configs/rag/baseline.yaml \
              configs/rag/top_k.yaml \
              configs/rag/hybrid.yaml \
              configs/rag/rerank.yaml \
              configs/rag/rag_fusion.yaml \
    --max-samples 20          # 快速验证用
```

---

## 配置说明

| 配置文件 | 检索策略 | 预期优势 |
|---------|---------|---------|
| `baseline.yaml` | 纯向量 k=3 | 当前生产基线 |
| `top_k.yaml` | 纯向量 k=8 | 提升 context recall |
| `hybrid.yaml` | BM25(0.4) + 向量(0.6) | 古典汉语字面匹配 |
| `rerank.yaml` | 向量粗召回 k=15 + 交叉编码器重排 top=3 | 提升 context precision |
| `rag_fusion.yaml` | 多查询变体 + RRF 融合 | 改善模糊查询 |

---

## 评测指标

| 指标 | 含义 | 目标 |
|------|------|------|
| `faithfulness` | 回答与检索到的上下文事实一致性 | ↑ 越高越好 |
| `answer_relevancy` | 回答与问题的相关程度 | ↑ 越高越好 |
| `context_recall` | 检索上下文覆盖黄金答案的程度 | ↑ 越高越好 |
| `context_precision` | 检索上下文中有用内容的比例 | ↑ 越高越好 |
| `latency_p50/p95` | 端到端延迟（秒） | ↓ 越低越好 |

---

## 结果文件

```
benchmarks/
├── qa_dataset.json                          # 问答数据集（由 generate_qa_dataset.py 生成）
├── requirements.txt                         # benchmark 专用依赖
└── results/
    ├── baseline_20260221_143000_detail.json   # 单配置详细结果（每条问题的答案、延迟）
    ├── hybrid_20260221_143500_detail.json
    └── comparison_20260221_144000.json        # 多配置汇总对比表
```

---

## 回归测试

在 CI 中加入如下步骤可防止指标倒退：

```yaml
# .github/workflows/rag-bench.yml (示例)
- name: Run RAG benchmark regression
  run: |
    python scripts/rag_bench.py \
      --config configs/rag/baseline.yaml \
      --max-samples 10 \
      --output-dir benchmarks/results
```
