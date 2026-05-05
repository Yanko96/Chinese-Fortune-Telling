"""Unit tests for api/graph_retriever.py — the BFS expansion + vector_filter_k
gate + book-diversity logic that powers Graph RAG.

These tests deliberately avoid the BGE cross-encoder (`reranker_model=""`)
so they run in <1 s without loading any models.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from graph_retriever import (  # noqa: E402  (sys.path patched by conftest)
    GraphRetriever,
    _diverse_top_n,
    _interleave_by_book,
    _text_to_chunk_id,
)


# ── _bfs_neighbors ───────────────────────────────────────────────────────────

def test_bfs_one_hop_returns_only_cross_book_neighbors(tiny_graph, tiny_chunk_index, fake_vectorstore):
    """BFS from smth_1 with hop=1 should reach dts_1 (cross-book) but
    NOT smth_2 (same book, even though it's a direct neighbor)."""
    r = GraphRetriever(
        vectorstore=fake_vectorstore,
        graph=tiny_graph,
        chunk_index=tiny_chunk_index,
        k=1, hop=1, top_n=5,
        reranker_model="",
        max_neighbors=10,
    )
    neighbors = r._bfs_neighbors("smth_1")
    assert "dts_1" in neighbors
    assert "smth_2" not in neighbors, "same-book neighbor must not be returned"


def test_bfs_two_hop_reaches_further_books(tiny_graph, tiny_chunk_index, fake_vectorstore):
    """BFS from smth_1 with hop=2 should reach zpzq_1 (smth_1 → dts_1 → zpzq_1)."""
    r = GraphRetriever(
        vectorstore=fake_vectorstore,
        graph=tiny_graph,
        chunk_index=tiny_chunk_index,
        k=1, hop=2, top_n=5,
        reranker_model="",
        max_neighbors=10,
    )
    neighbors = r._bfs_neighbors("smth_1")
    assert "dts_1" in neighbors
    assert "zpzq_1" in neighbors, "2-hop BFS must reach the indirect cross-book neighbor"


def test_bfs_respects_max_neighbors_cap(tiny_graph, tiny_chunk_index, fake_vectorstore):
    """max_neighbors=1 should hard-cap the returned list."""
    r = GraphRetriever(
        vectorstore=fake_vectorstore,
        graph=tiny_graph,
        chunk_index=tiny_chunk_index,
        k=1, hop=2, top_n=5,
        reranker_model="",
        max_neighbors=1,
    )
    neighbors = r._bfs_neighbors("smth_1")
    assert len(neighbors) == 1


def test_bfs_explores_high_weight_edges_first(tiny_graph, tiny_chunk_index, fake_vectorstore):
    """When max_neighbors limits expansion, the higher-weight edge wins.
    smth_1 → dts_1 (w=10) outranks smth_1 → smth_2 (w=4, but same-book filtered)."""
    r = GraphRetriever(
        vectorstore=fake_vectorstore,
        graph=tiny_graph,
        chunk_index=tiny_chunk_index,
        k=1, hop=1, top_n=5,
        reranker_model="",
        max_neighbors=1,
    )
    neighbors = r._bfs_neighbors("smth_1")
    assert neighbors == ["dts_1"], "high-weight cross-book edge must be picked first"


def test_bfs_returns_empty_for_unknown_node(tiny_graph, tiny_chunk_index, fake_vectorstore):
    r = GraphRetriever(
        vectorstore=fake_vectorstore,
        graph=tiny_graph,
        chunk_index=tiny_chunk_index,
        k=1, hop=1, top_n=5,
        reranker_model="",
    )
    assert r._bfs_neighbors("does_not_exist") == []


# ── End-to-end retrieve (no reranker) ────────────────────────────────────────

def test_retrieve_combines_seeds_with_graph_neighbors(tiny_graph, tiny_chunk_index, fake_vectorstore):
    """Seeds come from vectorstore; cross-book neighbors come from the graph.
    Final docs should contain both."""
    r = GraphRetriever(
        vectorstore=fake_vectorstore,
        graph=tiny_graph,
        chunk_index=tiny_chunk_index,
        k=2, hop=1, top_n=10,
        reranker_model="",
        max_neighbors=5,
    )
    docs = r._get_relevant_documents("正财格如何判断")
    books = {(d.metadata or {}).get("book") for d in docs}
    assert "三命通会" in books, "seed book must appear"
    assert "滴天髓" in books, "neighbor book must appear via graph expansion"


def test_vector_filter_k_drops_neighbors_not_in_whitelist(tiny_chunk_index, tiny_graph):
    """When vector_filter_k>0 and the wide-recall whitelist excludes dts_1,
    the BFS-discovered dts_1 must be dropped from the final docs."""
    from langchain_core.documents import Document

    class WhitelistVectorstore:
        def __init__(self):
            self.calls = 0

        def similarity_search(self, query: str, k: int = 10):
            self.calls += 1
            # First call (k=1 = seed): return smth_1
            # Second call (k=50 = wide whitelist): deliberately exclude dts_1
            if self.calls == 1:
                return [Document(page_content=tiny_chunk_index["smth_1"]["text"],
                                 metadata={"id": "smth_1"})]
            # Whitelist contains only smth_1 and zpzq_1 (NOT dts_1)
            return [
                Document(page_content=tiny_chunk_index["smth_1"]["text"], metadata={"id": "smth_1"}),
                Document(page_content=tiny_chunk_index["zpzq_1"]["text"], metadata={"id": "zpzq_1"}),
            ]

    vs = WhitelistVectorstore()
    r = GraphRetriever(
        vectorstore=vs, graph=tiny_graph, chunk_index=tiny_chunk_index,
        k=1, hop=1, top_n=10, reranker_model="",
        vector_filter_k=50, max_neighbors=10,
    )
    docs = r._get_relevant_documents("正财格")
    contents = [d.page_content for d in docs]
    assert tiny_chunk_index["dts_1"]["text"] not in contents, \
        "dts_1 must be filtered out because it's not in the vector whitelist"


# ── _diverse_top_n ────────────────────────────────────────────────────────────

def test_diverse_top_n_swaps_when_single_book():
    """If the top-K from rerank score is all one book, swap the lowest-scored
    in-bucket doc for the highest-scored cross-book doc."""
    A1 = Document(page_content="a1", metadata={"book": "A"})
    A2 = Document(page_content="a2", metadata={"book": "A"})
    A3 = Document(page_content="a3", metadata={"book": "A"})
    B1 = Document(page_content="b1", metadata={"book": "B"})
    ranked = [(0.9, A1), (0.8, A2), (0.7, A3), (0.5, B1)]
    selected = _diverse_top_n(ranked, top_n=3)
    books = {(d.metadata or {}).get("book") for d in selected}
    assert "A" in books and "B" in books, \
        "must contain at least 2 books even when scores favor one"


def test_diverse_top_n_keeps_pure_ranking_when_diverse():
    """If the top-K already spans ≥2 books, no swap should happen."""
    A1 = Document(page_content="a1", metadata={"book": "A"})
    B1 = Document(page_content="b1", metadata={"book": "B"})
    A2 = Document(page_content="a2", metadata={"book": "A"})
    ranked = [(0.9, A1), (0.8, B1), (0.7, A2)]
    selected = _diverse_top_n(ranked, top_n=2)
    assert selected[0].page_content == "a1"
    assert selected[1].page_content == "b1"


# ── _interleave_by_book ──────────────────────────────────────────────────────

def test_interleave_by_book_round_robins():
    docs = [
        Document(page_content="a1", metadata={"book": "A"}),
        Document(page_content="a2", metadata={"book": "A"}),
        Document(page_content="b1", metadata={"book": "B"}),
    ]
    result = _interleave_by_book(docs)
    # First two slots should be the first doc from each book
    books_seen_first_two = {result[0].metadata["book"], result[1].metadata["book"]}
    assert books_seen_first_two == {"A", "B"}
    assert result[2].metadata["book"] == "A"  # A's second doc fills the last slot


# ── _text_to_chunk_id ────────────────────────────────────────────────────────

def test_text_to_chunk_id_matches_by_prefix(tiny_chunk_index):
    text = tiny_chunk_index["smth_1"]["text"]
    assert _text_to_chunk_id(text, tiny_chunk_index) == "smth_1"


def test_text_to_chunk_id_returns_none_when_no_match(tiny_chunk_index):
    assert _text_to_chunk_id("totally unrelated content", tiny_chunk_index) is None
