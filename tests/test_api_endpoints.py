"""FastAPI endpoint smoke tests.

We use FastAPI's TestClient to hit the routes directly. The retrieval
chain is replaced with a stub so the tests run offline — they verify
the HTTP contract (status codes, response shape, validation) without
loading ChromaDB or making any LLM calls.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch, stub_fortune_chain):
    """A TestClient against the live FastAPI app, with:
      - rag_app.db redirected into a temp directory (no test pollution)
      - get_fortune_chain mocked to return a deterministic stub
    """
    # Run the test in a temp cwd so the sqlite file lands there
    monkeypatch.chdir(tmp_path)

    # Force re-import so db_utils re-initializes tables under the new cwd
    for mod in ("fortune_main", "fortune_langchain_utils", "db_utils", "chroma_utils"):
        sys.modules.pop(mod, None)

    import fortune_main

    monkeypatch.setattr(fortune_main, "get_fortune_chain", lambda **kw: stub_fortune_chain)
    return TestClient(fortune_main.app)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_healthz_api_prefix_alias(client):
    """The same handler should also be reachable under /api/ (ALB path routing)."""
    r = client.get("/api/healthz")
    assert r.status_code == 200


def test_zodiac_signs_returns_twelve(client):
    r = client.get("/zodiac-signs")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 12
    # spot-check that the Chinese characters survive JSON encoding
    assert any("龙" in s for s in data)


def test_fortune_methods_has_required_query_types(client):
    r = client.get("/fortune-methods")
    assert r.status_code == 200
    methods = r.json()
    qts = {m["query_type"] for m in methods}
    assert qts.issuperset({"bazi", "general", "forecast"})


def test_general_fortune_returns_stub_answer(client):
    r = client.post("/fortune", json={
        "question": "Tell me about my year.",
        "query_type": "general",
        "model": "moonshot-v1-8k",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "stub answer — testing only"
    assert body["query_type"] == "general"
    assert body["session_id"]  # generated


def test_bazi_without_birth_date_returns_400(client):
    """BaZi requests must include a valid birth_date — server validates this
    before invoking the chain."""
    r = client.post("/fortune", json={
        "question": "BaZi please",
        "query_type": "bazi",
    })
    assert r.status_code == 400
    assert "birth date" in r.json()["detail"].lower()


def test_bazi_with_invalid_birth_date_format(client):
    r = client.post("/fortune", json={
        "question": "BaZi please",
        "query_type": "bazi",
        "birth_date": "not-a-date",
    })
    assert r.status_code == 400


def test_bazi_with_valid_birth_date_passes(client):
    r = client.post("/fortune", json={
        "question": "BaZi please",
        "query_type": "bazi",
        "birth_date": "1990-01-15 14:30",
        "birth_gender": "male",
    })
    assert r.status_code == 200, r.text


def test_session_id_threaded_through(client):
    """Server should echo back any session_id the client supplies (so the
    Streamlit frontend can keep the same conversation)."""
    sid = "test-session-1234"
    r = client.post("/fortune", json={
        "question": "hello",
        "query_type": "general",
        "session_id": sid,
    })
    assert r.status_code == 200
    assert r.json()["session_id"] == sid


def test_arbitrary_model_string_is_accepted(client):
    """After the rolling-deploy enum incident, the `model` field is a plain
    str — any value should validate (even legacy ones like 'gemini-2.5-flash').
    See docs/DEPLOYMENT_NOTES.md §2 for the incident this guards against."""
    r = client.post("/fortune", json={
        "question": "hi",
        "query_type": "general",
        "model": "gemini-2.5-flash",  # legacy value from a stale client
    })
    assert r.status_code == 200, "permissive str validation must accept legacy values"


# ── Upstream-error mapping (Kimi 429 / overloaded / timeout) ──────────────

def _chain_that_raises(exc):
    """Helper: fortune chain that raises exc on .invoke()."""
    class _Boom:
        def invoke(self, _inputs):
            raise exc
    return _Boom()


def test_kimi_429_maps_to_503(tmp_path, monkeypatch):
    """Kimi 429 / 'engine_overloaded_error' must return HTTP 503 (not 500),
    so callers can distinguish upstream busy from server failure and retry
    with backoff. Regression for the prod incident on 2026-05-10."""
    monkeypatch.chdir(tmp_path)
    import sys
    for mod in ("fortune_main", "fortune_langchain_utils", "db_utils", "chroma_utils"):
        sys.modules.pop(mod, None)
    import fortune_main
    from fastapi.testclient import TestClient

    # Mimic the actual error string Kimi returns when overloaded
    boom = _chain_that_raises(
        Exception("Error code: 429 - {'error': {'message': "
                  "'The engine is currently overloaded, please try again later', "
                  "'type': 'engine_overloaded_error'}}")
    )
    monkeypatch.setattr(fortune_main, "get_fortune_chain", lambda **kw: boom)
    client = TestClient(fortune_main.app)

    r = client.post("/fortune", json={
        "question": "test", "query_type": "general", "model": "moonshot-v1-8k",
    })
    assert r.status_code == 503, f"expected 503, got {r.status_code}: {r.text}"
    body = r.json()
    assert "overloaded" in body["detail"].lower() or "拥堵" in body["detail"]


def test_kimi_timeout_maps_to_504(tmp_path, monkeypatch):
    """Upstream connection/timeout errors map to HTTP 504 Gateway Timeout."""
    monkeypatch.chdir(tmp_path)
    import sys
    for mod in ("fortune_main", "fortune_langchain_utils", "db_utils", "chroma_utils"):
        sys.modules.pop(mod, None)
    import fortune_main
    from fastapi.testclient import TestClient

    boom = _chain_that_raises(Exception("Connection error: read timed out after 90s"))
    monkeypatch.setattr(fortune_main, "get_fortune_chain", lambda **kw: boom)
    client = TestClient(fortune_main.app)

    r = client.post("/fortune", json={
        "question": "test", "query_type": "general", "model": "moonshot-v1-8k",
    })
    assert r.status_code == 504


# ── skip_rag overrides query_type (UX guardrail) ─────────────────────────

def test_greeting_with_bazi_query_type_downgrades_to_general(client):
    """Frontend sets query_type=bazi whenever the user has a birthday in the
    sidebar — even for chitchat. A '你好' should downgrade to GENERAL so the
    generic persona prompt answers, not the BaZi-specialized one that would
    force a four-pillar calculation. UX regression from prod self-test."""
    r = client.post("/fortune", json={
        "question": "你好",
        "query_type": "bazi",
        "birth_date": "1990-01-01 12:00",
        "birth_gender": "female",
        "model": "moonshot-v1-8k",
    })
    assert r.status_code == 200, r.text
    assert r.json()["query_type"] == "general", (
        "skip_rag must downgrade query_type so the client knows it wasn't "
        "actually a BaZi reading"
    )


def test_greeting_with_bazi_query_type_no_birthday_does_not_400(client):
    """When skip_rag fires, the BaZi birth_date validation must be bypassed
    — the user typing '你好' without a birthday shouldn't get a validation
    error just because the frontend defaulted to bazi mode."""
    r = client.post("/fortune", json={
        "question": "你好",
        "query_type": "bazi",
        # no birth_date intentionally
        "model": "moonshot-v1-8k",
    })
    assert r.status_code == 200, r.text


def test_real_bazi_question_still_validates_birthday(client):
    """Regression: real BaZi questions without birth_date must still 400.
    The skip_rag bypass shouldn't accidentally weaken the validation for
    actual BaZi requests."""
    r = client.post("/fortune", json={
        "question": "什么是正财格？",  # real divination question, not chitchat
        "query_type": "bazi",
        # no birth_date — should fail because skip_rag won't fire
        "model": "moonshot-v1-8k",
    })
    assert r.status_code == 400


def test_real_bazi_question_with_birthday_keeps_bazi_type(client):
    """Sanity: a substantive question with bazi+birthday should NOT downgrade
    to general — that path is reserved for chitchat only."""
    r = client.post("/fortune", json={
        "question": "请帮我分析正财格在我命中的体现",
        "query_type": "bazi",
        "birth_date": "1990-01-01 12:00",
        "birth_gender": "female",
        "model": "moonshot-v1-8k",
    })
    assert r.status_code == 200
    assert r.json()["query_type"] == "bazi"


def test_meta_question_with_forecast_query_type_downgrades(client):
    """Same downgrade also fires for forecast path — '你能做什么' with a
    zodiac sign selected should not produce a yearly forecast."""
    r = client.post("/fortune", json={
        "question": "你能做什么？",
        "query_type": "forecast",
        "zodiac_sign": "Dragon (龙)",
        "model": "moonshot-v1-8k",
    })
    assert r.status_code == 200
    assert r.json()["query_type"] == "general"


def test_unexpected_error_still_maps_to_500(tmp_path, monkeypatch):
    """Errors that aren't recognized 429/timeout patterns still get the
    generic 500 — we don't want to hide bugs behind a misleading 503."""
    monkeypatch.chdir(tmp_path)
    import sys
    for mod in ("fortune_main", "fortune_langchain_utils", "db_utils", "chroma_utils"):
        sys.modules.pop(mod, None)
    import fortune_main
    from fastapi.testclient import TestClient

    boom = _chain_that_raises(ValueError("totally unrelated app bug"))
    monkeypatch.setattr(fortune_main, "get_fortune_chain", lambda **kw: boom)
    client = TestClient(fortune_main.app)

    r = client.post("/fortune", json={
        "question": "test", "query_type": "general", "model": "moonshot-v1-8k",
    })
    assert r.status_code == 500
