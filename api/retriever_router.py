"""api/retriever_router.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-query retrieval-strategy routing.

Production runs HyDE + BGE Rerank as the default (highest faithfulness on
single-hop, AVG=0.812 on 22Q normal benchmark). For queries that are
*clearly* cross-book or multihop in nature, we route to Graph RAG v7
(`vector_filter_k=50`, multihop chain_score=0.729 vs HyDE's 0.593).

Routing policy is intentionally conservative: HyDE is the high-faithfulness
default winner, so we only divert traffic when we have HIGH confidence the
query benefits from graph topology. False positives are more costly than
false negatives because:
  - Routing to Graph adds ~10–20 s latency vs HyDE
  - HyDE is a strict superset of "just do dense + rerank" — never wrong
    by construction, only sometimes suboptimal

See README §1c. Routing for the rationale and benchmark backing.

Detectors are pure functions; they're independently unit-tested in
tests/test_retriever_router.py. The retriever factory is wired into
get_fortune_chain() — see api/fortune_langchain_utils.py.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
from typing import Any

from langchain_core.runnables import RunnableLambda


# ── Routing detectors ─────────────────────────────────────────────────────

# Classical-text book quote marks 《...》 — unambiguous in this domain
BOOK_PATTERN = re.compile(r"《([^》]+)》")

# Comparison / contrast / multihop keywords.
# Carefully scoped to phrases that strongly imply a cross-reference query.
# "比较好" / "比较一下" alone are intentionally NOT here — too noisy.
COMPARE_PATTERN = re.compile(
    r"对比|"                              # 对比 X 和 Y
    r"区别|异同|不同点|差异|"              # X 和 Y 的区别
    r"\bvs\b|\bVS\b|"                    # X vs Y
    r"以及.{1,15}[与和]|"                # X 以及 Y 与 Z 的关系
    r"何者|孰[更优强]|"                  # 何者更优 / 孰更
    r"相较|相比之下|"                    # 相较 X，Y...
    r"哪个更|哪一个更"                    # 哪个更 X
)

# Bridge-term vocabulary loaded lazily from data/chunk_index.json — the same
# 41 curated cross-book terms the knowledge graph was built on. Reusing
# this vocab means our entity detector has high precision for free.
_bridge_terms_cache: set[str] | None = None


def _get_bridge_terms() -> set[str]:
    """Load bridge-term vocabulary from data/chunk_index.json (lazy, cached).

    Returns an empty set if the chunk index isn't available — detector
    degrades gracefully to "no entity signal" rather than crashing.
    """
    global _bridge_terms_cache
    if _bridge_terms_cache is not None:
        return _bridge_terms_cache

    path = os.getenv("CHUNK_INDEX_PATH", "data/chunk_index.json")
    try:
        with open(path, encoding="utf-8") as f:
            chunk_index = json.load(f)
    except FileNotFoundError:
        logging.info(f"[router] chunk_index.json not found at {path}; "
                     "entity detection disabled")
        _bridge_terms_cache = set()
        return _bridge_terms_cache

    terms: set[str] = set()
    for info in chunk_index.values():
        terms.update(info.get("bridge_terms", []))
    _bridge_terms_cache = terms
    logging.info(f"[router] loaded {len(terms)} bridge terms for entity detection")
    return terms


def detect_cross_book(query: str) -> int:
    """Count *distinct* classical-book mentions (《X》patterns)."""
    return len(set(BOOK_PATTERN.findall(query)))


def has_compare_signal(query: str) -> bool:
    """Detect comparison / contrast / multihop intent via keyword patterns."""
    return bool(COMPARE_PATTERN.search(query))


def detect_multi_entity(query: str) -> int:
    """Count distinct bridge-term (命理 concept) mentions in the query."""
    terms = _get_bridge_terms()
    if not terms:
        return 0
    return sum(1 for t in terms if t in query)


# ── Routing decision ──────────────────────────────────────────────────────

def should_route_to_graph(query: str) -> tuple[bool, str]:
    """Decide whether this query benefits from Graph RAG over HyDE+Rerank.

    Returns (should_route_to_graph, reason_string).
    The reason string is logged for observability — lets us audit routing
    decisions in CloudWatch without re-running the heuristic.

    Conservative policy (default to HyDE, the production stable winner):

      Trigger graph path if EITHER:
        - 2+ distinct classical books mentioned (《X》《Y》) — clearly cross-book
        - 2+命理 bridge terms AND a compare keyword — clearly relational query

      Otherwise: HyDE+Rerank.

    Why conservative: HyDE wins single-hop (AVG=0.812 vs Graph v8's 0.804)
    AND is faster on a 0.5 vCPU task. Graph wins only on multihop chain
    reasoning (0.729 vs HyDE 0.593), which is a minority of production
    traffic. False-positive routing (sending single-hop to Graph) costs
    latency + a small quality loss; false-negative routing (keeping a
    multihop on HyDE) costs reasoning chain completeness.
    """
    cross_book = detect_cross_book(query)
    multi_entity = detect_multi_entity(query)
    compare = has_compare_signal(query)

    if cross_book >= 2:
        return True, f"cross_book_mentions={cross_book}"
    if multi_entity >= 2 and compare:
        return True, f"multi_entity={multi_entity}+compare_kw"

    return False, (
        f"default_hyde (cross_book={cross_book}, "
        f"multi_entity={multi_entity}, compare={compare})"
    )


# ── Retriever factory (lazy-load Graph state) ─────────────────────────────

_graph_state_cache: Any = None  # tuple[graph, chunk_index] or "missing"


def _get_graph_state():
    """Lazy-load knowledge_graph.pkl + chunk_index.json. Returns
    (graph, chunk_index) or None if artifacts aren't present (graceful
    degradation — router will fall through to HyDE for every query)."""
    global _graph_state_cache
    if _graph_state_cache == "missing":
        return None
    if _graph_state_cache is not None:
        return _graph_state_cache

    graph_path = os.getenv("KG_GRAPH_PATH", "data/knowledge_graph.pkl")
    index_path = os.getenv("KG_INDEX_PATH", "data/chunk_index.json")

    try:
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        with open(index_path, encoding="utf-8") as f:
            chunk_index = json.load(f)
    except FileNotFoundError as e:
        logging.warning(f"[router] graph artifacts missing ({e}); "
                        "graph route disabled, all queries go to HyDE")
        _graph_state_cache = "missing"
        return None

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    logging.info(f"[router] loaded knowledge graph: {n_nodes} nodes, "
                 f"{n_edges} edges, {len(chunk_index)} chunks")
    _graph_state_cache = (graph, chunk_index)
    return _graph_state_cache


def build_routed_retriever(llm, hyde_retriever, vectorstore):
    """Build a router that picks between HyDE+Rerank and Graph RAG per query.

    Parameters
    ----------
    llm : ChatOpenAI
        Kimi LLM (unused by Graph v7 directly, but kept for future routing
        strategies that might want to call an LLM for query classification)
    hyde_retriever : Runnable
        The RunnableLambda from _build_hyde_rerank_retriever — used as the
        default and as fallback when graph state is unavailable.
    vectorstore : Chroma
        Same vector store used by both paths.

    Returns a RunnableLambda that:
      1. Inspects the query (after stripping production English prefix)
      2. Calls should_route_to_graph(...)
      3. Routes to GraphRetriever(v7 config) or hyde_retriever
      4. Falls back to hyde_retriever if the graph path errors or state
         is unavailable.
    """
    # Reuse the prefix-strip function so router decisions are based on the
    # user's actual Chinese intent, not the English BaZi/Forecast wrapper.
    from fortune_langchain_utils import _strip_query_prefix  # type: ignore

    graph_retriever: Any = None  # lazy-built on first graph-routed query

    def _ensure_graph_retriever():
        nonlocal graph_retriever
        if graph_retriever is not None:
            return graph_retriever
        state = _get_graph_state()
        if state is None:
            return None
        from graph_retriever import GraphRetriever  # type: ignore
        graph, chunk_index = state
        graph_retriever = GraphRetriever(
            vectorstore=vectorstore,
            graph=graph,
            chunk_index=chunk_index,
            k=8,
            hop=1,
            top_n=5,
            reranker_model="BAAI/bge-reranker-base",
            max_neighbors=10,
            vector_filter_k=50,  # ★ v7 winning config — semantic gate on BFS neighbors
        )
        return graph_retriever

    def _route(inp):
        # Accept both dict (from history-aware retriever) and raw string
        q_full = inp.get("input", "") if isinstance(inp, dict) else str(inp)
        q_clean = _strip_query_prefix(q_full)

        route_to_graph, reason = should_route_to_graph(q_clean)
        logging.info(
            f"[router] decision={'graph' if route_to_graph else 'hyde'}: {reason}"
        )

        if route_to_graph:
            gr = _ensure_graph_retriever()
            if gr is not None:
                try:
                    # GraphRetriever's BaseRetriever interface takes a string
                    return gr.invoke(q_clean)
                except Exception as e:
                    logging.warning(
                        f"[router] graph path failed ({type(e).__name__}: {e}); "
                        "falling back to HyDE"
                    )
            # If we get here: graph state missing OR graph call errored.
            # Fall through to HyDE.

        # Default / fallback path. HyDE retriever does its own prefix strip
        # internally — pass the original dict so chat_history is preserved.
        return hyde_retriever.invoke(inp)

    return RunnableLambda(_route)
