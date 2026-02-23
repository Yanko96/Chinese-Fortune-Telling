"""
generate_qa_dataset.py
~~~~~~~~~~~~~~~~~~~~~~
从现有 Chroma 向量库中随机采样文档块，使用 Gemini 合成问答对，
生成供 rag_bench.py 使用的黄金测试集。

用法:
    python scripts/generate_qa_dataset.py \
        --num-samples 50 \
        --output benchmarks/qa_dataset.json \
        --chroma-dir ./chroma_db

输出 JSON 格式:
    [
      {
        "id": "uuid",
        "question": "...",
        "golden_answer": "...",
        "source_chunks": ["...", ...],
        "metadata": { "source": "...", "page": 0 }
      },
      ...
    ]
"""

import argparse
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# 把 api/ 加入路径，复用已有的 chroma_utils
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# 生成单个 QA pair
# ---------------------------------------------------------------------------

QA_GENERATION_PROMPT = """\
你是一位中国传统命理学专家。下面是一段来自古典命理文献的文字：

---
{chunk}
---

请根据这段文字生成一个高质量的问答对，要求：
1. 问题（question）必须是一个自然的用户提问，聚焦于这段文字的核心知识点
2. 答案（golden_answer）必须仅基于这段文字，完整且准确，不超过 300 字
3. 问题和答案都使用**中文**

请严格以下面的 JSON 格式输出，不要添加任何额外内容：
{{
  "question": "...",
  "golden_answer": "..."
}}
"""


def generate_qa_pair(chunk_text: str, llm) -> dict[str, str] | None:
    """调用 LLM 生成一个 QA pair，返回 dict 或 None（失败时）。"""
    import re

    prompt = QA_GENERATION_PROMPT.format(chunk=chunk_text[:2000])
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        # 提取 JSON（兼容模型在 JSON 前后添加 markdown 代码块的情况）
        json_match = re.search(r"\{[\s\S]*?\}", content)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"  [WARN] QA generation failed: {e}")
    return None


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate RAG benchmark QA dataset")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of QA pairs to generate")
    parser.add_argument("--output", default="benchmarks/qa_dataset.json", help="Output JSON file path")
    parser.add_argument("--chroma-dir", default="./chroma_db", help="Path to Chroma persist directory")
    parser.add_argument(
        "--provider",
        default="kimi",
        choices=["kimi", "gemini", "groq", "deepseek"],
        help="LLM provider (default: kimi)",
    )
    parser.add_argument(
        "--model",
        default="moonshot-v1-32k",
        help="Model name (default: moonshot-v1-32k for kimi)",
    )
    parser.add_argument("--request-delay", type=float, default=0.5, help="Seconds between API calls")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--min-chunk-len", type=int, default=200, help="Minimum chunk character length to include")
    args = parser.parse_args()

    random.seed(args.seed)

    # --- 加载 Chroma ---
    print(f"Loading Chroma from: {args.chroma_dir}")
    os.environ.setdefault("CHROMA_DIR", args.chroma_dir)
    from chroma_utils import get_vectorstore

    vs = get_vectorstore()
    all_docs_result: dict[str, Any] = vs.get()  # {"ids": [...], "documents": [...], "metadatas": [...]}
    all_ids: list[str] = all_docs_result["ids"]
    all_texts: list[str] = all_docs_result["documents"]
    all_meta: list[dict] = all_docs_result["metadatas"]

    print(f"Total chunks in Chroma: {len(all_ids)}")

    # 过滤太短的块
    candidates = [
        (doc_id, text, meta)
        for doc_id, text, meta in zip(all_ids, all_texts, all_meta)
        if text and len(text) >= args.min_chunk_len
    ]
    print(f"Candidates after min-length filter ({args.min_chunk_len} chars): {len(candidates)}")

    if len(candidates) == 0:
        print("[ERROR] No candidates found. Check --chroma-dir or --min-chunk-len.")
        sys.exit(1)

    # 随机采样（不超过候选数量）
    sample_size = min(args.num_samples, len(candidates))
    sampled = random.sample(candidates, sample_size)
    print(f"Sampled {sample_size} chunks for QA generation")

    # --- 初始化 LLM ---
    from langchain_openai import ChatOpenAI

    print(f"LLM provider: {args.provider}  model: {args.model}")
    if args.provider == "kimi":
        api_key = os.environ.get("KIMI_API_KEY")
        if not api_key:
            print("[ERROR] KIMI_API_KEY not set in .env"); sys.exit(1)
        llm = ChatOpenAI(model=args.model, temperature=0.3,
                         base_url="https://api.moonshot.cn/v1", api_key=api_key)
    elif args.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=args.model, temperature=0.3, max_retries=3)
    elif args.provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("[ERROR] GROQ_API_KEY not set in .env"); sys.exit(1)
        llm = ChatGroq(model=args.model, temperature=0.3, api_key=api_key)
    elif args.provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("[ERROR] DEEPSEEK_API_KEY not set in .env"); sys.exit(1)
        llm = ChatOpenAI(model=args.model, temperature=0.3,
                         base_url="https://api.deepseek.com", api_key=api_key)

    # --- 批量生成 ---
    dataset: list[dict] = []
    failed = 0
    for i, (doc_id, chunk_text, meta) in enumerate(sampled, 1):
        print(f"  [{i}/{sample_size}] Generating QA for chunk {doc_id[:8]}...")
        if i > 1:
            time.sleep(args.request_delay)
        qa = generate_qa_pair(chunk_text, llm)
        if qa:
            dataset.append(
                {
                    "id": str(uuid.uuid4()),
                    "question": qa["question"],
                    "golden_answer": qa["golden_answer"],
                    "source_chunks": [chunk_text],
                    "metadata": meta or {},
                }
            )
        else:
            failed += 1

    print(f"\nGenerated {len(dataset)} QA pairs ({failed} failed)")

    # --- 保存 ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
