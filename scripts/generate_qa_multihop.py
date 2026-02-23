"""
generate_qa_multihop.py
~~~~~~~~~~~~~~~~~~~~~~~
生成多跳推理 QA 数据集，用于评估 Graph RAG vs Flat RAG。

核心理念：
  每道题需要同时引用两个原文片段（通常来自不同书籍）进行推理——
  "片段 A 说明概念 X，片段 B 说明 X 在情境 Y 下的含义，问：情境 Y 下的结论是？"
  单独任何一个片段都不足以完整回答。

输出格式（benchmarks/qa_multihop.json）:
  [{
    "id": "uuid",
    "question": "...",
    "reasoning_chain": ["步骤1: ...", "步骤2: ...", "步骤3: ..."],
    "required_hops": 3,
    "golden_answer": "...",
    "source_chunks": ["片段1文本", "片段2文本"],
    "metadata": {
      "book1": "三命通会",
      "book2": "子平真诠",
      "bridge_term": "官星"
    }
  }, ...]

用法:
    cd E:\\repos\\Chinese-Fortune-Telling
    E:\\Software\\Anaconda3\\envs\\rag\\python.exe scripts/generate_qa_multihop.py

    # 自定义每书对数量
    E:\\Software\\Anaconda3\\envs\\rag\\python.exe scripts/generate_qa_multihop.py \\
        --per-pair 15 --output benchmarks/qa_multihop.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "api"))
from dotenv import load_dotenv

load_dotenv(override=True)

# ── 超参 ──────────────────────────────────────────────────────────────────

BOOKS              = ["三命通会", "滴天髓", "子平真诠"]
DEFAULT_PER_PAIR   = 12        # 每对书籍生成多少道题（3 对 × 12 = 36 题目标）
DEFAULT_CHROMA_DIR = "./chroma_db_bge"
DEFAULT_OUTPUT     = "benchmarks/qa_multihop.json"
CACHE_FILE         = "benchmarks/multihop_cache.jsonl"
MIN_CHUNK_LEN      = 200
MAX_CHUNK_LEN      = 800       # 截断，避免 prompt 过长
MAX_WORKERS        = 3
SEED               = 42
GEN_MODEL          = "moonshot-v1-32k"   # 用更强的模型来生成多跳题

# 桥接词：两个片段至少共享其中一个词，才能形成有意义的多跳关联
BRIDGE_TERMS = [
    "官星", "正官", "偏官", "七杀",
    "财星", "正财", "偏财",
    "印绶", "正印", "偏印", "枭神",
    "伤官", "食神",
    "日主", "日元",
    "大运", "行运", "流年", "岁运",
    "格局", "命格", "用神",
    "五行", "生克", "制化",
    "木", "火", "土", "金", "水",
    "官印", "财官", "食财",
]


# ── 生成 Prompt ───────────────────────────────────────────────────────────

MULTIHOP_PROMPT = """\
你是中国传统命理学专家和命题专家。以下是来自两本古籍的原文片段：

【片段一 · 来自《{book1}》】
{chunk1}

【片段二 · 来自《{book2}》】
{chunk2}

请设计一道**多跳推理问题**，要求：
1. 问题必须同时利用这两个片段的内容才能完整回答（单看任一片段无法作答）
2. 推理链：2-4 个步骤，每步一句话，展示从片段一到片段二的逻辑链条
   例如："步骤1：片段一指出甲木喜水 → 步骤2：片段二说明印绶运水旺 → 步骤3：因此甲木日主走印绶运吉"
3. 答案：整合两片段，200-300 字，逻辑严密
4. 如果两个片段无法形成有意义的多跳推理（主题差异太大），返回 null

只输出以下 JSON，不加其他内容：
{{
  "question": "...",
  "reasoning_chain": ["步骤1: ...", "步骤2: ...", "步骤3: ..."],
  "required_hops": 3,
  "golden_answer": "..."
}}

或者返回：null
"""


# ── 缓存 ─────────────────────────────────────────────────────────────────

def load_cache(path: str) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    p = Path(path)
    if not p.exists():
        return cache
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                cache[obj["pair_id"]] = obj
            except Exception:
                pass
    print(f"  Resume: {len(cache)} pairs loaded from cache")
    return cache


def append_cache(path: str, entry: dict, lock: threading.Lock):
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── 单对生成 ─────────────────────────────────────────────────────────────

def generate_one_pair(
    pair_id: str,
    book1: str, chunk1: str,
    book2: str, chunk2: str,
    bridge_term: str,
    client,
    cache_path: str,
    lock: threading.Lock,
) -> dict | None:
    import re

    prompt = MULTIHOP_PROMPT.format(
        book1=book1,
        chunk1=chunk1[:MAX_CHUNK_LEN],
        book2=book2,
        chunk2=chunk2[:MAX_CHUNK_LEN],
    )
    try:
        resp = client.chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=800,
        )
        content = resp.choices[0].message.content.strip()

        # 明确返回 null → 片段不相关
        if content.lower().strip() in ("null", "null.", "none"):
            return None

        m = re.search(r"\{[\s\S]*?\}", content)
        if not m:
            return None

        qa = json.loads(m.group())
        if not qa.get("question") or not qa.get("golden_answer") or not qa.get("reasoning_chain"):
            return None
        if not isinstance(qa["reasoning_chain"], list) or len(qa["reasoning_chain"]) < 2:
            return None

        entry = {
            "pair_id":       pair_id,
            "id":            str(uuid.uuid4()),
            "question":      qa["question"].strip(),
            "reasoning_chain": [s.strip() for s in qa["reasoning_chain"]],
            "required_hops": int(qa.get("required_hops", len(qa["reasoning_chain"]))),
            "golden_answer": qa["golden_answer"].strip(),
            "source_chunks": [chunk1, chunk2],
            "metadata": {
                "book1": book1,
                "book2": book2,
                "bridge_term": bridge_term,
            },
        }
        append_cache(cache_path, entry, lock)
        return entry
    except Exception as e:
        print(f"\n  [ERR] {pair_id[:8]} ({book1}×{book2}): {e}")
        return None


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-pair",   type=int, default=DEFAULT_PER_PAIR)
    parser.add_argument("--chroma-dir", default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--output",     default=DEFAULT_OUTPUT)
    parser.add_argument("--seed",       type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    root = Path(__file__).parent.parent
    os.chdir(root)

    # ── API 客户端 ───────────────────────────────────────────────────────
    from openai import OpenAI

    api_key = os.environ.get("KIMI_API_KEY")
    if not api_key:
        raise EnvironmentError("KIMI_API_KEY not set in .env")
    client = OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")

    # ── 加载 Chroma ──────────────────────────────────────────────────────
    print(f"Loading chunks from {args.chroma_dir} ...")
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = Chroma(persist_directory=args.chroma_dir, embedding_function=emb)
    result = vs.get()

    # 按书整理候选块（过滤太短的块）
    book_chunks: dict[str, list[tuple[str, str]]] = {b: [] for b in BOOKS}
    for cid, text, meta in zip(result["ids"], result["documents"], result["metadatas"]):
        if not text or len(text) < MIN_CHUNK_LEN:
            continue
        book = (meta or {}).get("book", "")
        if book in book_chunks:
            book_chunks[book].append((cid, text))

    for book, chunks in book_chunks.items():
        print(f"  {book}: {len(chunks)} candidates")

    # ── 断点续传 ─────────────────────────────────────────────────────────
    cache_path = str(root / CACHE_FILE)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path)
    cached_pair_ids = set(cache.keys())

    # ── 构建候选对 ────────────────────────────────────────────────────────
    # 策略：对每对书，找到含有相同桥接词的片段对（关联度更高），
    # 取 per_pair * 3 个候选（大部分 Kimi 可能返回 null），实际成功约 per_pair 个
    book_pairs = list(combinations(BOOKS, 2))
    todo: list[tuple[str, str, str, str, str, str]] = []   # (pair_id, b1, c1, b2, c2, bridge)

    for book1, book2 in book_pairs:
        chunks1 = book_chunks[book1]
        chunks2 = book_chunks[book2]

        # 找共享桥接词的片段对
        bridged_pairs: list[tuple[str, str, str, str, str]] = []  # (c1_id, c1, c2_id, c2, term)
        for term in BRIDGE_TERMS:
            c1_with = [(cid, t) for cid, t in chunks1 if term in t]
            c2_with = [(cid, t) for cid, t in chunks2 if term in t]
            if c1_with and c2_with:
                for c1 in random.sample(c1_with, min(4, len(c1_with))):
                    for c2 in random.sample(c2_with, min(4, len(c2_with))):
                        bridged_pairs.append((c1[0], c1[1], c2[0], c2[1], term))

        # 去重并随机采样
        seen = set()
        unique_pairs = []
        for item in bridged_pairs:
            key = (item[0], item[2])
            if key not in seen:
                seen.add(key)
                unique_pairs.append(item)

        random.shuffle(unique_pairs)
        # 目标：成功生成 per_pair 题；假设约 50% 的候选被 Kimi 接受，所以取 2× 候选数
        candidates = unique_pairs[: args.per_pair * 2]
        print(f"  {book1} × {book2}: {len(unique_pairs)} bridged pairs → {len(candidates)} selected")

        for c1_id, c1_text, c2_id, c2_text, term in candidates:
            pair_id = f"{c1_id[:8]}_{c2_id[:8]}"
            if pair_id not in cached_pair_ids:
                todo.append((pair_id, book1, c1_text, book2, c2_text, term))

    print(f"\nTotal to generate: {len(todo)}  (already cached: {len(cached_pair_ids)})")

    # ── 并发生成 ──────────────────────────────────────────────────────────
    new_entries: list[dict] = []
    if todo:
        lock = threading.Lock()
        done = failed = skipped = 0
        print(f"Generating with {MAX_WORKERS} concurrent workers ...\n")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {
                pool.submit(
                    generate_one_pair,
                    pair_id, book1, c1_text, book2, c2_text, term,
                    client, cache_path, lock
                ): (pair_id, book1, book2)
                for pair_id, book1, c1_text, book2, c2_text, term in todo
            }
            for fut in as_completed(futs):
                pair_id, book1, book2 = futs[fut]
                entry = fut.result()
                if entry:
                    cache[pair_id] = entry
                    new_entries.append(entry)
                    done += 1
                else:
                    # None = Kimi said not enough multi-hop relation
                    skipped += 1
                n = done + failed + skipped
                print(
                    f"  [{n:3d}/{len(todo)}] ✓{done} ✗{failed} ○{skipped}(null)",
                    end="\r", flush=True
                )
        print(f"\n  Done: {done} generated, {skipped} null (unrelated pairs), {failed} errors")
    else:
        print("All pairs already cached.")

    # ── 组装最终数据集 ────────────────────────────────────────────────────
    # 每对书各取 per_pair 条
    final: list[dict] = []
    for book1, book2 in book_pairs:
        pair_entries = [
            v for v in cache.values()
            if v["metadata"]["book1"] == book1 and v["metadata"]["book2"] == book2
        ]
        random.shuffle(pair_entries)
        taken = pair_entries[: args.per_pair]
        final.extend(taken)
        print(f"  {book1} × {book2}: {len(taken)} multi-hop QA pairs")

    random.shuffle(final)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    # 打印样例
    if final:
        print(f"\n── 样例 ──────────────────────────────────────────────────────")
        ex = final[0]
        print(f"  书籍: {ex['metadata']['book1']} × {ex['metadata']['book2']}  (桥接词: {ex['metadata']['bridge_term']})")
        print(f"  问题: {ex['question']}")
        print(f"  推理链:")
        for step in ex["reasoning_chain"]:
            print(f"    • {step}")

    print(f"\n✅  {len(final)} multi-hop QA pairs → {output_path}")
    print(f"\n下一步 — 运行 baseline 评估:")
    print(f"   python scripts/bench_multihop.py \\")
    print(f"       --configs configs/rag/v2/hybrid.yaml configs/rag/v5/hyde_rerank_topn7.yaml \\")
    print(f"       --dataset {output_path}")


if __name__ == "__main__":
    main()
