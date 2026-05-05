"""Shared fixtures for the test suite.

The tests deliberately avoid touching the real ChromaDB, BGE reranker,
or any external LLM. Fixtures here provide lightweight stand-ins so the
suite runs offline in under a few seconds.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

# Make api/ and scripts/ importable as flat modules (matches production layout
# where api/fortune_main.py imports `from chroma_utils import ...`).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT))

# Tests never need a real API key — set placeholders so module-level reads
# of os.environ don't fail.
os.environ.setdefault("MOONSHOT_API_KEY", "test-not-a-real-key")
os.environ.setdefault("OPENAI_API_KEY", "test-not-a-real-key")


# ── Fake knowledge graph fixtures ────────────────────────────────────────────

@pytest.fixture
def tiny_graph():
    """A 6-node graph spanning 3 books, with edge weights.

    Layout:
        SMTH_1 ── (w=10) ── DTS_1 ── (w=8) ── ZPZQ_1
          │                                      │
        (w=4)                                  (w=6)
          │                                      │
        SMTH_2                                 DTS_2  (cross-book to SMTH_2 at w=3)

    Useful for exercising BFS neighbor expansion across books.
    """
    import networkx as nx

    g = nx.Graph()
    g.add_edge("smth_1", "dts_1", weight=10.0)
    g.add_edge("dts_1", "zpzq_1", weight=8.0)
    g.add_edge("smth_1", "smth_2", weight=4.0)
    g.add_edge("zpzq_1", "dts_2", weight=6.0)
    g.add_edge("smth_2", "dts_2", weight=3.0)
    return g


@pytest.fixture
def tiny_chunk_index():
    """Companion chunk_index for tiny_graph. Maps node id → {book, text, bridge_terms}."""
    return {
        "smth_1": {"book": "三命通会", "text": "三命通会论正财格" * 10, "bridge_terms": ["正财", "格局"]},
        "smth_2": {"book": "三命通会", "text": "三命通会论七杀" * 10, "bridge_terms": ["七杀"]},
        "dts_1":  {"book": "滴天髓",   "text": "滴天髓论用神" * 10,   "bridge_terms": ["用神"]},
        "dts_2":  {"book": "滴天髓",   "text": "滴天髓论格局" * 10,   "bridge_terms": ["格局"]},
        "zpzq_1": {"book": "子平真诠", "text": "子平真诠论正官" * 10, "bridge_terms": ["正官"]},
    }


# ── Fake vectorstore (mimics chromadb's similarity_search signature) ─────────

class FakeVectorstore:
    """Returns a fixed sequence of LangChain `Document` objects.

    Tests can configure `.docs` to control what similarity_search returns.
    The k argument is honored (we truncate `.docs[:k]`) so behavior matches
    a real Chroma instance closely enough for retriever unit tests.
    """

    def __init__(self, docs: list[Any]):
        self.docs = docs

    def similarity_search(self, query: str, k: int = 10):
        return self.docs[:k]


@pytest.fixture
def fake_vectorstore(tiny_chunk_index):
    """A FakeVectorstore prepopulated with two SMTH seeds (id metadata set)."""
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content=tiny_chunk_index["smth_1"]["text"],
            metadata={"id": "smth_1", "book": "三命通会"},
        ),
        Document(
            page_content=tiny_chunk_index["smth_2"]["text"],
            metadata={"id": "smth_2", "book": "三命通会"},
        ),
    ]
    return FakeVectorstore(docs)


# ── Stub fortune chain (for API endpoint tests) ──────────────────────────────

class _StubChain:
    """Mimics the LangChain RAG chain's `.invoke()` contract.

    Returns a deterministic dict with the keys the FastAPI handler reads.
    """

    def invoke(self, inputs):
        return {
            "answer": "stub answer — testing only",
            "input": inputs.get("input", ""),
        }


@pytest.fixture
def stub_fortune_chain():
    return _StubChain()
