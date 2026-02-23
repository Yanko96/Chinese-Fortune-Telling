"""
build_index_bge.py
~~~~~~~~~~~~~~~~~~
用语义边界分块 + 中文 BGE 嵌入，重新构建向量索引。

分块策略（按书目）:
  三命通会 : 按 ○ 标记切分（298 处），每节为一完整语义单元
  子平真诠 : 按汉字章节序号（十八、…）切分（94 章）
  滴天髓   : 按空行（\\n\\n）切分，将四字诀 + 原注 + 任氏曰 合并为一个语义块

二次分割：超过 MAX_CHUNK_CHARS 的块用 RecursiveCharacterTextSplitter 再切；
合并小块：短于 MIN_CHUNK_CHARS 的块合并到下一块。

输出: chroma_db_bge/  (collection: "langchain")
嵌入: BAAI/bge-small-zh-v1.5  (384-dim，中文优化)

用法:
    cd E:\\repos\\Chinese-Fortune-Telling
    E:\\Software\\Anaconda3\\envs\\rag\\python.exe scripts/build_index_bge.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# ── 超参 ──────────────────────────────────────────────────────────────────

BOOKS = [
    {
        "path": "fortune_books/san_ming_tong_hui.pdf",
        "label": "三命通会",
        "splitter": "circle_marker",   # 按 ○ 切分
    },
    {
        "path": "fortune_books/di_tian_sui.pdf",
        "label": "滴天髓",
        "splitter": "blank_line",      # 按空行 + 合并四字诀段落
    },
    {
        "path": "fortune_books/zi_ping_zhen_quan.pdf",
        "label": "子平真诠",
        "splitter": "chapter_number",  # 按汉字章节号切分
    },
]

CHROMA_DIR     = "./chroma_db_bge"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
MAX_CHUNK_CHARS = 1000   # 超出则二次切分
MIN_CHUNK_CHARS = 50     # 低于此则合并到下一块
COLLECTION_NAME = "langchain"

# ── 页眉清理规则 ──────────────────────────────────────────────────────────

HEADER_PATTERNS = [
    r"三命通会\s*·\s*\d+\s*·",          # 三命通会 ·9·
    r"滴天髓\s*·\s*\d+\s*·",            # 滴天髓 ·3·
    r"-\s*\d+\s*/\s*\d+\s*-",           # -9/153-  (子平真诠)
    r"子平真诠[^\n]{0,20}·\s*\d+\s*·",  # 子平真诠-沈孝瞻 ·5·
]


def clean_text(raw: str) -> str:
    """去掉页眉、规范化连续空行（三行以上压缩为两行）。"""
    for pat in HEADER_PATTERNS:
        raw = re.sub(pat, "", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def extract_full_text(pdf_path: str) -> str:
    """拼接 PDF 所有页的文本。"""
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        t = page.extract_text() or ""
        pages.append(t)
    return clean_text("\n\n".join(pages))


# ── 各书专属分块函数 ──────────────────────────────────────────────────────

def split_by_circle_marker(text: str) -> list[str]:
    """
    按 ○ 切分（三命通会）。
    ○ 标记始终出现在小节首行，如「○论五行生成」，
    用前瞻断言保留 ○ 在每个块的开头。
    """
    parts = re.split(r"(?=○)", text)
    return [p.strip() for p in parts if p.strip()]


def split_by_chapter_number(text: str) -> list[str]:
    """
    按汉字章节号切分（子平真诠）。
    Pattern: 行首 + 汉字数字序列 + 顿号/句号，例如「十八、论四吉神能破格」。
    用前瞻断言保留序号在每个块开头。
    """
    parts = re.split(r"(?=\n[一二三四五六七八九十百千]+[、．。])", text)
    return [p.strip() for p in parts if p.strip()]


def split_by_blank_line(text: str) -> list[str]:
    """
    按空行切分，并将「四字诀 + 原注 + 任氏曰」合并为一个语义块（滴天髓）。

    滴天髓结构：
        支神只以冲为重，刑与穿兮动不动。   ← 四字诀（短，<100字）
        [空行]
        原注：冲者必是相克...              ← 原注段落
        [空行]
        任氏曰：...                        ← 任注段落
        [空行]
        下一个四字诀...                    ← 新语义块起点

    合并规则：
        - 遇到短段落（≤100字，不以"原注"/"任氏曰"开头）→ 视为四字诀，
          与其后的段落合并，直到下一个四字诀出现或合并体超过 MAX_CHUNK_CHARS。
    """
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    merged: list[str] = []
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        # 判断是否为「四字诀」：短段、不以原注/任氏曰开头
        is_verse = (
            len(para) <= 100
            and not para.startswith("原注")
            and not para.startswith("任氏曰")
            and not para.startswith("●")
            and not para.startswith("【")
        )
        if is_verse and i + 1 < len(paragraphs):
            # 合并后续段落直到遇到下一个四字诀或超出大小限制
            combined = para
            j = i + 1
            while j < len(paragraphs):
                next_para = paragraphs[j]
                next_is_verse = (
                    len(next_para) <= 100
                    and not next_para.startswith("原注")
                    and not next_para.startswith("任氏曰")
                    and not next_para.startswith("●")
                    and not next_para.startswith("【")
                )
                # 已有实质内容（>100字）时遇到新四字诀 → 停止合并
                if next_is_verse and len(combined) > 100:
                    break
                combined = combined + "\n\n" + next_para
                j += 1
                if len(combined) >= MAX_CHUNK_CHARS:
                    break
            merged.append(combined)
            i = j
        else:
            merged.append(para)
            i += 1

    return merged


# ── 二次分割 + 合并极小块 ─────────────────────────────────────────────────

def secondary_split(chunks: list[str]) -> list[str]:
    """
    过大块（>1.5×MAX_CHUNK_CHARS）→ RecursiveCharacterTextSplitter 二次切。
    极小块（<MIN_CHUNK_CHARS）→ 合并到下一块，避免孤立碎片。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    secondary = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_CHARS,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", "。", "，", "、", " ", ""],
    )

    expanded: list[str] = []
    for chunk in chunks:
        if len(chunk) > MAX_CHUNK_CHARS * 1.5:
            sub_docs = secondary.split_documents([Document(page_content=chunk)])
            expanded.extend(s.page_content for s in sub_docs)
        else:
            expanded.append(chunk)

    # 合并极小块
    final: list[str] = []
    buffer = ""
    for chunk in expanded:
        if len(chunk) < MIN_CHUNK_CHARS:
            buffer += (" " if buffer else "") + chunk
        else:
            if buffer:
                final.append(buffer.strip())
                buffer = ""
            final.append(chunk)
    if buffer:
        if final:
            final[-1] = final[-1] + " " + buffer.strip()
        else:
            final.append(buffer.strip())

    return final


# ── 统计辅助 ─────────────────────────────────────────────────────────────

def chunk_stats(chunks: list[str]) -> dict:
    sizes = sorted(len(c) for c in chunks)
    n = len(sizes)
    return {
        "count": n,
        "min": sizes[0],
        "p25": sizes[n // 4],
        "median": sizes[n // 2],
        "p75": sizes[3 * n // 4],
        "max": sizes[-1],
        "mean": int(sum(sizes) / n),
    }


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    import os
    import shutil
    from langchain_core.documents import Document
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # ── 加载嵌入模型 ──────────────────────────────────────────────────────
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # BGE 推荐归一化
    )
    print("  ✓ Embedding model loaded\n")

    all_docs: list[Document] = []

    # ── 各书处理 ──────────────────────────────────────────────────────────
    for book_cfg in BOOKS:
        pdf_path  = book_cfg["path"]
        label     = book_cfg["label"]
        stype     = book_cfg["splitter"]

        print(f"{'─'*55}")
        print(f"📖  {label}  ({pdf_path})")

        full_text = extract_full_text(pdf_path)
        print(f"    Total chars : {len(full_text):,}")

        if stype == "circle_marker":
            raw_chunks = split_by_circle_marker(full_text)
        elif stype == "blank_line":
            raw_chunks = split_by_blank_line(full_text)
        elif stype == "chapter_number":
            raw_chunks = split_by_chapter_number(full_text)
        else:
            raise ValueError(f"Unknown splitter type: {stype}")

        print(f"    After semantic split : {len(raw_chunks)} chunks")

        final_chunks = secondary_split(raw_chunks)
        st = chunk_stats(final_chunks)
        print(f"    After secondary split: {st['count']} chunks")
        print(f"    Size stats  : min={st['min']}  p25={st['p25']}  "
              f"median={st['median']}  p75={st['p75']}  max={st['max']}  mean={st['mean']}")

        for chunk_text in final_chunks:
            all_docs.append(Document(
                page_content=chunk_text,
                metadata={"source": label, "book": label},
            ))

    print(f"\n{'='*55}")
    print(f"Total documents to index: {len(all_docs)}")

    # ── 清空并重建 ChromaDB ───────────────────────────────────────────────
    chroma_path = Path(CHROMA_DIR)
    if chroma_path.exists():
        print(f"Removing existing index at {CHROMA_DIR} ...")
        shutil.rmtree(chroma_path)

    print(f"Building Chroma index at: {CHROMA_DIR}")
    print("(This may take a minute while encoding all documents...)")

    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=emb,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    count = vectorstore._collection.count()
    print(f"\n✓ Indexed {count} chunks  →  {CHROMA_DIR}")

    # ── 分书统计 ──────────────────────────────────────────────────────────
    print("\nPer-book chunk counts:")
    for book_cfg in BOOKS:
        label = book_cfg["label"]
        res = vectorstore.get(where={"book": label})
        print(f"  {label}: {len(res['ids'])} chunks")

    print("\n✅  Done! Use configs/rag/v2/*.yaml to benchmark against this index.")


if __name__ == "__main__":
    main()
