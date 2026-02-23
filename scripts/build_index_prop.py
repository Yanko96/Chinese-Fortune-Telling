"""
build_index_prop.py
~~~~~~~~~~~~~~~~~~~
命题索引（Proposition Indexing）构建器

核心思路：
  检索时用"命题"（粒度细 → 精准匹配问题关键词）
  生成时返回命题对应的"父块"原文（粒度粗 → 保留完整上下文）

流程：
  1. 从 chroma_db_bge/ 读取全部语义块（671 块）
  2. 用 Kimi moonshot-v1-8k 并发提取每块的原子命题（3-8 条/块）
  3. 命题写入 chroma_db_prop/（bge-small-zh-v1.5 嵌入）
     每条命题 metadata["parent_chunk"] 保存父块原文

断点续传：提取结果缓存到 propositions_cache.jsonl，
          重跑时自动跳过已处理的块。

用时估计：671 块 × ~2s（3 并发） ≈ 7 分钟

用法：
    cd E:\\repos\\Chinese-Fortune-Telling
    E:\\Software\\Anaconda3\\envs\\rag\\python.exe scripts/build_index_prop.py
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# ── Hyper-parameters ─────────────────────────────────────────────────────────

CHROMA_BGE      = "./chroma_db_bge"
CHROMA_PROP     = "./chroma_db_prop"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
CACHE_FILE      = "propositions_cache.jsonl"
KIMI_MODEL      = "moonshot-v1-8k"   # 短输出、速度快，适合批量提取
MAX_WORKERS     = 3                   # 并发 API 调用数（太多会触发限速）
MAX_PROPS       = 12                  # 每块命题上限

EXTRACT_PROMPT = """\
你是中国古典命理学专家。请从以下命理古籍片段中，提取所有独立的事实性命题。

要求：
1. 每条命题是完整的中文陈述句，自包含（读者无需查看原文即可理解含义）
2. 每条命题不超过 120 字
3. 仅提取原文中明确表达的内容，不推断，不引申
4. 专有术语（五行、十神、格局名等）保持原文表达
5. 每行一条命题，不加序号或任何前缀，不输出空行

原文片段：
{text}

命题（每行一条，不加序号）："""


# ── Step 1: 从 chroma_db_bge 读取语义块 ──────────────────────────────────────

def load_chunks(chroma_dir: str, emb_model: str) -> list[dict]:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    print(f"Loading chunks from {chroma_dir} ...")
    emb = HuggingFaceEmbeddings(
        model_name=emb_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = Chroma(persist_directory=chroma_dir, embedding_function=emb)
    result = vs.get()
    chunks = [
        {"id": cid, "text": text.strip(), "meta": meta or {}}
        for cid, text, meta in zip(
            result["ids"], result["documents"], result["metadatas"]
        )
        if text and text.strip()
    ]
    print(f"  {len(chunks)} non-empty chunks")

    # 分书统计
    from collections import Counter
    book_cnt = Counter(c["meta"].get("book", "unknown") for c in chunks)
    for book, cnt in sorted(book_cnt.items()):
        print(f"    {book}: {cnt}")
    return chunks


# ── Step 2: 缓存读写 ──────────────────────────────────────────────────────────

def load_cache(path: str) -> dict[str, list[str]]:
    cache: dict[str, list[str]] = {}
    p = Path(path)
    if not p.exists():
        return cache
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                cache[obj["chunk_id"]] = obj["propositions"]
            except Exception:
                pass
    print(f"  Loaded {len(cache)} cached entries from {p.name}")
    return cache


def append_cache(path: str, chunk_id: str, props: list[str], lock: threading.Lock):
    entry = json.dumps(
        {"chunk_id": chunk_id, "propositions": props}, ensure_ascii=False
    )
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")


# ── Step 3: Kimi 命题提取（单块）────────────────────────────────────────────

def call_kimi(text: str, client) -> list[str]:
    resp = client.chat.completions.create(
        model=KIMI_MODEL,
        messages=[{"role": "user", "content": EXTRACT_PROMPT.format(text=text)}],
        temperature=0,
        max_tokens=1500,
    )
    raw = resp.choices[0].message.content.strip()
    props = [p.strip() for p in raw.split("\n") if p.strip() and len(p.strip()) > 8]
    return props[:MAX_PROPS]


def worker(
    chunk: dict, client, cache_path: str, lock: threading.Lock
) -> tuple[str, list[str]]:
    cid = chunk["id"]
    try:
        props = call_kimi(chunk["text"], client)
        append_cache(cache_path, cid, props, lock)
        return cid, props
    except Exception as e:
        print(f"\n  [ERR] {cid[:8]}: {e}")
        return cid, []


# ── Step 4: 构建命题向量库 ─────────────────────────────────────────────────────

def build_prop_vectorstore(
    chunks: list[dict],
    all_props: dict[str, list[str]],
    emb_model: str,
    chroma_dir: str,
):
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings

    print(f"\nBuilding proposition vectorstore → {chroma_dir}")
    emb = HuggingFaceEmbeddings(
        model_name=emb_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    prop_docs: list[Document] = []
    missing = 0
    for c in chunks:
        props = all_props.get(c["id"])
        if not props:
            # 兜底：把块前 120 字作为单条命题
            props = [c["text"][:120]]
            missing += 1
        for p in props:
            prop_docs.append(
                Document(
                    page_content=p,
                    metadata={
                        **c["meta"],
                        "parent_chunk": c["text"],
                        "parent_chunk_id": c["id"],
                    },
                )
            )

    total   = len(prop_docs)
    avg_per = total / max(len(chunks), 1)
    print(f"  Total propositions to index  : {total}")
    print(f"  Average per chunk            : {avg_per:.1f}")
    if missing:
        print(f"  [WARN] {missing} chunks had no extracted props → fallback used")

    if Path(chroma_dir).exists():
        print(f"  Removing existing index at {chroma_dir} ...")
        shutil.rmtree(chroma_dir)

    print("  Encoding + persisting (may take ~1 min) ...")
    vs = Chroma.from_documents(
        documents=prop_docs,
        embedding=emb,
        persist_directory=chroma_dir,
        collection_name="langchain",
    )
    count = vs._collection.count()
    print(f"  ✓ Indexed {count} propositions  →  {chroma_dir}")

    # 分书统计
    from collections import Counter
    book_cnt = Counter(
        doc.metadata.get("book", "unknown") for doc in prop_docs
    )
    print("\n  Per-book proposition counts:")
    for book, cnt in sorted(book_cnt.items()):
        print(f"    {book}: {cnt}")

    return vs


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    from openai import OpenAI

    root = Path(__file__).parent.parent
    os.chdir(root)
    cache_path = str(root / CACHE_FILE)

    api_key = os.environ.get("KIMI_API_KEY")
    if not api_key:
        raise EnvironmentError("KIMI_API_KEY not set in .env")
    client = OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")

    # ── 1. 读取语义块 ────────────────────────────────────────────────────────
    chunks = load_chunks(CHROMA_BGE, EMBEDDING_MODEL)

    # ── 2. 加载断点缓存 ──────────────────────────────────────────────────────
    cache = load_cache(cache_path)
    todo  = [c for c in chunks if c["id"] not in cache]
    print(f"  To extract: {len(todo)} / {len(chunks)}  (cached: {len(cache)})\n")

    # ── 3. 并发提取命题 ──────────────────────────────────────────────────────
    if todo:
        print(f"Extracting propositions  ({MAX_WORKERS} concurrent workers) ...")
        t0 = time.time()
        lock = threading.Lock()
        done_ok = done_err = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {pool.submit(worker, c, client, cache_path, lock): c for c in todo}
            for fut in as_completed(futs):
                cid, props = fut.result()
                if props:
                    cache[cid] = props
                    done_ok += 1
                else:
                    done_err += 1
                n       = done_ok + done_err
                elapsed = time.time() - t0
                eta     = elapsed / n * (len(todo) - n) if n else 0
                print(
                    f"  [{n:3d}/{len(todo)}]  ✓{done_ok}  ✗{done_err}"
                    f"  {elapsed:.0f}s elapsed  ETA {eta:.0f}s   ",
                    end="\r",
                    flush=True,
                )

        print(f"\n  Extraction done: {done_ok} ok, {done_err} failed")
    else:
        print("All chunks already in cache — skipping extraction.")

    # ── 4. 构建命题向量库 ────────────────────────────────────────────────────
    build_prop_vectorstore(chunks, cache, EMBEDDING_MODEL, CHROMA_PROP)
    print("\n✅  Done!  Use configs/rag/v3/*.yaml to benchmark proposition retrieval.")


if __name__ == "__main__":
    main()
