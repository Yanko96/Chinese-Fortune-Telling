"""Tests for api/retriever_router.py.

Covers the three detectors (cross-book / multi-entity / compare-keyword)
and the should_route_to_graph policy. The router's full integration
(actually invoking Graph RAG vs HyDE) is exercised in test_api_endpoints
via the FastAPI smoke tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retriever_router import (  # noqa: E402  (sys.path patched by conftest)
    detect_cross_book,
    detect_multi_entity,
    has_compare_signal,
    should_route_to_graph,
    should_skip_rag,
    _get_bridge_terms,
)


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _ensure_bridge_terms(monkeypatch):
    """Point the bridge-term loader at the real data/chunk_index.json so the
    multi-entity detector tests run against the actual 41-term vocabulary.
    """
    chunk_index_path = ROOT / "data" / "chunk_index.json"
    if chunk_index_path.exists():
        monkeypatch.setenv("CHUNK_INDEX_PATH", str(chunk_index_path))
    # Clear the module-level cache so each test reloads fresh
    import retriever_router
    retriever_router._bridge_terms_cache = None


# ── detect_cross_book ────────────────────────────────────────────────────

def test_cross_book_two_distinct_books():
    q = "《滴天髓》和《子平真诠》对正官的论述有何不同？"
    assert detect_cross_book(q) == 2


def test_cross_book_three_books():
    q = "《三命通会》《滴天髓》《子平真诠》对七杀的看法各有差异。"
    assert detect_cross_book(q) == 3


def test_cross_book_same_book_quoted_twice_counts_once():
    q = "《滴天髓》中说... 同样在《滴天髓》也提到..."
    assert detect_cross_book(q) == 1


def test_cross_book_no_books():
    assert detect_cross_book("什么是正财格？") == 0


# ── has_compare_signal ───────────────────────────────────────────────────

@pytest.mark.parametrize(
    "q",
    [
        "正官与七杀对比有何不同？",
        "两者的区别是什么？",
        "正财格和偏财格的差异如何？",
        "正官 vs 七杀，哪个更主贵？",
        "比较起来，哪个更适合？",  # 哪个更
        "相较于正官，七杀的特点是？",  # 相较
    ],
)
def test_compare_positive(q):
    assert has_compare_signal(q), f"should detect compare in: {q}"


@pytest.mark.parametrize(
    "q",
    [
        "什么是正财格？",
        "如何判断一个八字是否构成正财格？",
        "请解释一下用神的概念。",
        "我今年的运势如何？",
    ],
)
def test_compare_negative(q):
    assert not has_compare_signal(q), f"should NOT detect compare in: {q}"


# ── detect_multi_entity (uses real bridge_terms) ─────────────────────────

@pytest.mark.skipif(
    not (ROOT / "data" / "chunk_index.json").exists(),
    reason="data/chunk_index.json not present — entity detection skipped",
)
def test_multi_entity_two_terms():
    # Both 正官 and 偏官 (七杀 in formal classical terminology) are in the
    # bridge vocabulary — corpus uses 偏官/七煞 rather than 七杀.
    q = "正官和偏官的格局如何区分？"
    n = detect_multi_entity(q)
    assert n >= 2, f"expected ≥2 entities (正官, 偏官), got {n}"


@pytest.mark.skipif(
    not (ROOT / "data" / "chunk_index.json").exists(),
    reason="data/chunk_index.json not present — entity detection skipped",
)
def test_multi_entity_none():
    assert detect_multi_entity("今天天气怎么样") == 0


def test_multi_entity_degrades_gracefully_when_index_missing(monkeypatch, tmp_path):
    """If chunk_index.json is missing, detector should return 0 rather than crash."""
    monkeypatch.setenv("CHUNK_INDEX_PATH", str(tmp_path / "nonexistent.json"))
    import retriever_router
    retriever_router._bridge_terms_cache = None
    assert detect_multi_entity("正官七杀") == 0


# ── should_route_to_graph (routing policy) ───────────────────────────────

def test_route_to_graph_on_two_books():
    """Most direct multihop signal: two distinct《》mentions."""
    q = "《滴天髓》和《子平真诠》对正官的论述有何不同？"
    route, reason = should_route_to_graph(q)
    assert route is True
    assert "cross_book" in reason, f"reason should mention cross_book, got: {reason}"


def test_route_to_graph_on_three_books():
    q = "《三命通会》《滴天髓》《子平真诠》如何论述用神？"
    route, _ = should_route_to_graph(q)
    assert route is True


@pytest.mark.skipif(
    not (ROOT / "data" / "chunk_index.json").exists(),
    reason="data/chunk_index.json not present",
)
def test_route_to_graph_on_multi_entity_with_compare():
    """Two 十神 entities + a compare keyword = clearly cross-reference.
    Using 正官/偏官 here because the corpus's bridge-term vocab uses
    偏官 (formal classical name) rather than 七杀 (modern usage)."""
    q = "正官和偏官的区别是什么？"
    route, reason = should_route_to_graph(q)
    assert route is True
    assert "multi_entity" in reason


def test_route_to_hyde_default_single_hop():
    """Plain single-hop question goes to HyDE."""
    q = "什么是正财格？"
    route, _ = should_route_to_graph(q)
    assert route is False


def test_route_to_hyde_single_entity_with_compare():
    """One entity + compare keyword — not strong enough signal. Conservative
    default to HyDE."""
    q = "正官如何区分喜忌？"  # has "区分" but only one entity
    route, _ = should_route_to_graph(q)
    assert route is False


def test_route_to_hyde_with_only_compare_no_entities():
    """A compare keyword without classical concepts shouldn't trigger Graph."""
    q = "两种方法相比哪个好？"
    route, _ = should_route_to_graph(q)
    assert route is False


def test_route_to_hyde_for_production_bazi_path():
    """Production BaZi queries (with English wrapper) are typical single-hop —
    must not accidentally route to Graph due to the English prefix tokens."""
    q = ("BaZi analysis for someone born on 1990-07-22 14:15, gender: female. "
         "什么是正财格？")
    route, _ = should_route_to_graph(q)
    assert route is False, "BaZi production query must default to HyDE"


def test_route_to_graph_for_production_multihop_bazi():
    """A multihop query that *also* has the English BaZi wrapper should still
    route to Graph based on the embedded Chinese intent."""
    q = ("BaZi analysis for someone born on 1990-07-22 14:15, gender: female. "
         "《滴天髓》和《子平真诠》对正官的论述有何不同？")
    route, reason = should_route_to_graph(q)
    assert route is True
    assert "cross_book" in reason


# ── should_skip_rag (no-RAG fast path) ────────────────────────────────────

@pytest.mark.parametrize(
    "q,expected_reason",
    [
        ("你好", "greeting"),
        ("您好", "greeting"),
        ("你好！", "greeting"),
        ("你好。", "greeting"),
        ("早上好", "greeting"),
        ("晚安", "greeting"),
        ("hi", "greeting"),
        ("Hello!", "greeting"),
        ("hey", "greeting"),
        ("Good morning", "greeting"),
        ("你是谁？", "meta_question"),
        ("你能做什么？", "meta_question"),
        ("介绍一下你自己", "meta_question"),
        ("自我介绍一下", "meta_question"),
        ("Who are you?", "meta_question"),
        ("What can you do?", "meta_question"),
        ("谢谢", "thanks_or_farewell"),
        ("谢谢！", "thanks_or_farewell"),
        ("感谢", "thanks_or_farewell"),
        ("再见", "thanks_or_farewell"),
        ("bye", "thanks_or_farewell"),
        ("thanks", "thanks_or_farewell"),
    ],
)
def test_skip_rag_positive(q, expected_reason):
    skip, reason = should_skip_rag(q)
    assert skip is True, f"should skip RAG for {q!r}"
    assert reason == expected_reason


@pytest.mark.parametrize(
    "q",
    [
        "什么是正财格？",                          # real question
        "你好，请问什么是正财格？",                # greeting + real question
        "我命中带什么？",                          # real divination ask
        "正官和偏官的区别？",                      # multihop question (should go to graph, not skip)
        "今年运势如何",                            # real forecast ask
        "Hi, what is BaZi?",                       # greeting + real question
        "你能算一下我的命吗？",                    # contains 你能 but full sentence — real ask
        "你介绍一下正财格",                        # contains 介绍 but for a topic, not self-intro
    ],
)
def test_skip_rag_negative(q):
    skip, _ = should_skip_rag(q)
    assert skip is False, f"should NOT skip RAG for {q!r}"


def test_skip_rag_does_not_misfire_on_real_question_after_greeting():
    """A query like '你好，请问什么是正财格？' must still go through retrieval.
    Anchored ^...$ patterns ensure greeting tokens at the start of a longer
    sentence don't trigger skip-RAG."""
    skip, _ = should_skip_rag("你好，请问什么是正财格？")
    assert skip is False


# ── Reason strings are useful for observability ──────────────────────────

def test_reason_includes_decision_features():
    """The reason string is logged for routing observability — it must
    include the underlying feature counts."""
    _, reason = should_route_to_graph("什么是正财格？")
    assert "cross_book=" in reason
    assert "multi_entity=" in reason
    assert "compare=" in reason
