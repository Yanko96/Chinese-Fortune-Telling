"""Production Shadow Eval

Runs the production HyDE+Rerank pipeline (k=8, top_n=5) on paired queries:
  - control: the original Chinese question
  - treatment: the production-shaped variant ("BaZi analysis for someone born
    on {date}, gender: {sex}. {question}")

Both queries hit the same retriever; GPT-4o then scores each retrieval's
relevance to the *underlying Chinese intent* on a 1-5 scale. Reports the mean
delta — quantifying how much the English wrapper degrades retrieval quality.

Usage:
    python scripts/shadow_eval.py \
        --chroma-dir ./chroma_db_bge \
        --dataset benchmarks/qa_production_shadow.json \
        --output benchmarks/results/shadow_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make api/ importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

from dotenv import load_dotenv  # type: ignore

load_dotenv(ROOT / ".env")

# Accept either MOONSHOT_API_KEY (prod env var name) or KIMI_API_KEY (.env alias)
if "MOONSHOT_API_KEY" not in os.environ and "KIMI_API_KEY" in os.environ:
    os.environ["MOONSHOT_API_KEY"] = os.environ["KIMI_API_KEY"]


def build_retriever(chroma_dir: str, hyde_k: int = 8, top_n: int = 5):
    """Construct the same HyDE+Rerank retriever the production API uses."""
    from langchain_openai import ChatOpenAI
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    from langchain_core.documents import Document

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

    llm = ChatOpenAI(
        model="moonshot-v1-8k",
        openai_api_key=os.environ["MOONSHOT_API_KEY"],
        openai_api_base="https://api.moonshot.cn/v1",
        temperature=0.7,
    )

    HYDE_PROMPT = (
        "你是中国传统命理学专家。请根据以下问题，"
        "仿照古典命理文献（文言文/半文言文）的风格，"
        "写一段80-150字的原文片段，直接包含问题答案所涉及的术语和论述。"
        "只输出片段本身，不要标题、序号或解释。\n\n"
        "问题：{question}"
    )

    def retrieve(query: str) -> list[Document]:
        # Step 1: HyDE
        try:
            hyp = llm.invoke(HYDE_PROMPT.format(question=query)).content.strip()
        except Exception:
            hyp = query
        # Step 2: Wide recall on hypothesis
        candidates = vectorstore.similarity_search(hyp, k=hyde_k)
        if not candidates:
            return []
        # Step 3: BGE rerank against *original* query
        pairs = [[query, d.page_content] for d in candidates]
        scores = encoder.score(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked[:top_n]]

    return retrieve


JUDGE_PROMPT = """你是中国传统命理学评估专家。请评估以下检索结果对**用户原始问题**的相关性。

用户原始问题（中文）：{question}

检索到的古籍片段（共{n}段）：
{chunks}

评分标准（1-5）：
- 5：所有片段都高度相关，直接回答问题
- 4：大多数片段相关，少量边缘内容
- 3：约一半片段相关
- 2：少量片段相关，多数偏离主题
- 1：几乎所有片段与问题无关

只输出一个 1 到 5 之间的整数，不要解释。
"""


def score_retrieval(judge_llm, question: str, docs) -> int:
    chunks = "\n---\n".join(
        f"[{i+1}] {d.page_content[:400]}" for i, d in enumerate(docs)
    )
    prompt = JUDGE_PROMPT.format(question=question, n=len(docs), chunks=chunks)
    resp = judge_llm.invoke(prompt).content.strip()
    # Extract first integer 1-5 from response
    for ch in resp:
        if ch.isdigit():
            val = int(ch)
            if 1 <= val <= 5:
                return val
    return 0  # parsing failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroma-dir", default="./chroma_db_bge")
    ap.add_argument("--dataset", default="benchmarks/qa_production_shadow.json")
    ap.add_argument("--output", default="benchmarks/results/shadow_eval.json")
    ap.add_argument("--hyde-k", type=int, default=8)
    ap.add_argument("--top-n", type=int, default=5)
    args = ap.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)
    pairs = dataset["pairs"]
    print(f"Running shadow eval on {len(pairs)} paired queries")

    retrieve = build_retriever(args.chroma_dir, args.hyde_k, args.top_n)

    from langchain_openai import ChatOpenAI
    judge = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.0,
    )

    results = []
    for pair in pairs:
        pid = pair["id"]
        control_q = pair["original"]
        treatment_q = (
            f"BaZi analysis for someone born on {pair['birth_date']}, "
            f"gender: {pair['gender']}. {pair['original']}"
        )

        print(f"\n[{pid}] retrieving (control)...", flush=True)
        try:
            control_docs = retrieve(control_q)
        except Exception as e:
            print(f"  control retrieval failed: {e}")
            control_docs = []

        print(f"[{pid}] retrieving (treatment)...", flush=True)
        try:
            treatment_docs = retrieve(treatment_q)
        except Exception as e:
            print(f"  treatment retrieval failed: {e}")
            treatment_docs = []

        print(f"[{pid}] judging (gpt-4o)...", flush=True)
        # IMPORTANT: judge against the ORIGINAL Chinese question for both
        # — we're measuring whether the prefix degrades retrieval quality
        # with respect to the underlying user intent, not the surface query.
        control_score = score_retrieval(judge, control_q, control_docs) if control_docs else 0
        treatment_score = score_retrieval(judge, control_q, treatment_docs) if treatment_docs else 0

        delta = treatment_score - control_score
        print(f"[{pid}] control={control_score}, treatment={treatment_score}, delta={delta:+d}")

        results.append({
            "id": pid,
            "original_question": control_q,
            "production_query": treatment_q,
            "control_score": control_score,
            "treatment_score": treatment_score,
            "delta": delta,
            "control_chunks": [d.page_content[:200] for d in control_docs],
            "treatment_chunks": [d.page_content[:200] for d in treatment_docs],
        })

        # Be polite to API rate limits
        time.sleep(1)

    # Aggregate
    n = len(results)
    valid = [r for r in results if r["control_score"] > 0 and r["treatment_score"] > 0]
    control_mean = sum(r["control_score"] for r in valid) / max(len(valid), 1)
    treatment_mean = sum(r["treatment_score"] for r in valid) / max(len(valid), 1)
    delta_mean = treatment_mean - control_mean
    win = sum(1 for r in valid if r["delta"] > 0)
    tie = sum(1 for r in valid if r["delta"] == 0)
    loss = sum(1 for r in valid if r["delta"] < 0)

    summary = {
        "n_pairs": n,
        "n_valid": len(valid),
        "control_mean_score": round(control_mean, 3),
        "treatment_mean_score": round(treatment_mean, 3),
        "delta_mean": round(delta_mean, 3),
        "treatment_wins": win,
        "ties": tie,
        "control_wins": loss,
        "hyde_k": args.hyde_k,
        "top_n": args.top_n,
        "config": "production HyDE+Rerank (k=8, top_n=5), bge-reranker-base",
        "judge": "gpt-4o, temperature=0",
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_pair": results}, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("SHADOW EVAL SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
