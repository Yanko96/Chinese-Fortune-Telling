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
