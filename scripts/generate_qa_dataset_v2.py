"""
generate_qa_dataset_v2.py
~~~~~~~~~~~~~~~~~~~~~~~~~
从 chroma_db_bge（语义块索引）生成均衡 QA 数据集。

改进（vs generate_qa_dataset.py）:
  - 每本书各采样 N 个块（默认 50），确保三书均衡覆盖
  - 并发 API 调用（3 workers），速度约为顺序方式的 3×
  - 断点续传：中途中断后重跑自动跳过已生成的条目
  - 保存 source/book 元数据，便于后续按书评分

用法:
    cd E:\\repos\\Chinese-Fortune-Telling
    E:\\Software\\Anaconda3\\envs\\rag\\python.exe scripts/generate_qa_dataset_v2.py

    # 自定义每书样本数
    E:\\Software\\Anaconda3\\envs\\rag\\python.exe scripts/generate_qa_dataset_v2.py \
        --per-book 30 --output benchmarks/qa_dataset_v2.json

输出: benchmarks/qa_dataset_v2.json
格式:
  [{ "id", "question", "golden_answer", "source_chunks", "metadata": {"book", ...} }, ...]
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "api"))
from dotenv import load_dotenv

load_dotenv(override=True)

# ── 超参 ──────────────────────────────────────────────────────────────────

BOOKS = ["三命通会", "滴天髓", "子平真诠"]
DEFAULT_PER_BOOK   = 50
DEFAULT_CHROMA_DIR = "./chroma_db_bge"
DEFAULT_OUTPUT     = "benchmarks/qa_dataset_v2.json"
CACHE_FILE         = "benchmarks/qa_gen_cache_v2.jsonl"   # 断点续传缓存
MIN_CHUNK_LEN      = 150      # 太短的块跳过
MAX_CHUNK_LEN      = 1200     # 太长的块截断后生成
MAX_WORKERS        = 3        # 并发 API 调用数
SEED               = 42

QA_PROMPT = """\
你是中国传统命理学专家。以下是一段来自古籍《{book}》的原文片段：

---
{chunk}
---

请根据这段原文生成一个**高质量问答对**，要求：
1. **问题**：自然的中文提问，聚焦该片段的核心知识点，不能在原文中找到完整答案的字面句子
2. **答案**：仅依据原文作答，准确完整，150-250 字，保留必要的术语
3. 问答均用中文

严格以下 JSON 格式输出，不加任何多余内容：
{{
  "question": "...",
  "golden_answer": "..."
}}
"""


# ── 缓存 ──────────────────────────────────────────────────────────────────

def load_cache(path: str) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    p = Path(path)
    if not p.exists():
        return cache
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                cache[obj["chunk_id"]] = obj
            except Exception:
                pass
    print(f"  Resume: {len(cache)} entries loaded from cache")
    return cache


def append_cache(path: str, entry: dict, lock: threading.Lock):
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── 单条生成 ─────────────────────────────────────────────────────────────

def generate_one(
    chunk_id: str,
    chunk_text: str,
    book: str,
    client,
    cache_path: str,
    lock: threading.Lock,
) -> dict | None:
    import re

    prompt = QA_PROMPT.format(book=book, chunk=chunk_text[:MAX_CHUNK_LEN])
    try:
        resp = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=600,
        )
        content = resp.choices[0].message.content.strip()
        m = re.search(r"\{[\s\S]*?\}", content)
        if not m:
            return None
        qa = json.loads(m.group())
        if not qa.get("question") or not qa.get("golden_answer"):
            return None
        entry = {
            "chunk_id": chunk_id,
            "id": str(uuid.uuid4()),
            "question": qa["question"].strip(),
            "golden_answer": qa["golden_answer"].strip(),
            "source_chunks": [chunk_text],
            "metadata": {"book": book},
        }
        append_cache(cache_path, entry, lock)
        return entry
    except Exception as e:
        print(f"\n  [ERR] {chunk_id[:8]} ({book}): {e}")
        return None


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-book", type=int, default=DEFAULT_PER_BOOK)
    parser.add_argument("--chroma-dir", default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
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

    # 按书整理候选块
    book_candidates: dict[str, list[tuple]] = {b: [] for b in BOOKS}
    for cid, text, meta in zip(result["ids"], result["documents"], result["metadatas"]):
        if not text or len(text) < MIN_CHUNK_LEN:
            continue
        book = (meta or {}).get("book", "")
        if book in book_candidates:
            book_candidates[book].append((cid, text))

    for book, cands in book_candidates.items():
        print(f"  {book}: {len(cands)} candidates (want {args.per_book})")

    # ── 断点续传 ─────────────────────────────────────────────────────────
    cache_path = str(root / CACHE_FILE)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path)
    cached_ids = set(cache.keys())

    # ── 采样 ─────────────────────────────────────────────────────────────
    todo: list[tuple[str, str, str]] = []   # (chunk_id, text, book)
    for book, cands in book_candidates.items():
        # 优先选择缓存中没有的块，不足时允许复用
        fresh = [(cid, t) for cid, t in cands if cid not in cached_ids]
        n_fresh = min(args.per_book, len(fresh))
        sampled = random.sample(fresh, n_fresh)

        # 如果 fresh 不够，从剩余候选补齐（缓存里已有的直接用）
        needed = args.per_book - n_fresh
        if needed > 0:
            rest = [(cid, t) for cid, t in cands if cid in cached_ids]
            sampled += random.sample(rest, min(needed, len(rest)))

        for cid, text in sampled:
            if cid not in cached_ids:
                todo.append((cid, text, book))

    print(f"\nTotal to generate: {len(todo)}  (already cached: {sum(1 for b in BOOKS for cid,_ in book_candidates[b] if cid in cached_ids)})")

    # ── 并发生成 ──────────────────────────────────────────────────────────
    if todo:
        lock = threading.Lock()
        done = failed = 0
        print(f"Generating with {MAX_WORKERS} concurrent workers ...\n")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {
                pool.submit(generate_one, cid, text, book, client, cache_path, lock): (cid, book)
                for cid, text, book in todo
            }
            for fut in as_completed(futs):
                cid, book = futs[fut]
                result_entry = fut.result()
                if result_entry:
                    cache[cid] = result_entry
                    done += 1
                else:
                    failed += 1
                n = done + failed
                print(f"  [{n:3d}/{len(todo)}] ✓{done} ✗{failed}  ({book})", end="\r", flush=True)
        print(f"\n  Done: {done} generated, {failed} failed")
    else:
        print("All samples already cached.")

    # ── 组装最终数据集 ─────────────────────────────────────────────────────
    # 每书各取 per_book 条（优先使用新生成的，不足时补已缓存的）
    final: list[dict] = []
    for book, cands in book_candidates.items():
        book_entries = [
            cache[cid] for cid, _ in cands
            if cid in cache and cache[cid].get("question")
        ]
        random.shuffle(book_entries)
        taken = book_entries[: args.per_book]
        final.extend(taken)
        print(f"  {book}: {len(taken)} QA pairs")

    random.shuffle(final)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"\n✅  {len(final)} QA pairs → {output_path}")
    print("   Run benchmark with:")
    print(f"   python scripts/rag_bench.py --configs configs/rag/v2/hybrid.yaml "
          f"--dataset {output_path} --max-samples 150 --output-dir benchmarks/results/v2_150")


if __name__ == "__main__":
    main()
