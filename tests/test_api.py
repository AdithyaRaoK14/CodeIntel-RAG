"""
tests/test_api.py
──────────────────
FastAPI endpoint tests using TestClient.

Run:  pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from api import app, _DEFAULT_STATE, AppState


@pytest.fixture()
def client():
    """Fresh TestClient for each test."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def loaded_client(tmp_path):
    """Client with a minimal in-memory repo already loaded into state."""
    _DEFAULT_STATE.chunks = [
        {"name": "login",    "file": "auth.py",  "calls": [],      "code": "def login(): pass",    "line": 1},
        {"name": "register", "file": "auth.py",  "calls": ["login"],"code": "def register(): pass", "line": 5},
    ]
    _DEFAULT_STATE.docs   = [{"file_name": "auth.py", "file_path": str(tmp_path / "auth.py"), "content": ""}]
    _DEFAULT_STATE.repo   = str(tmp_path)
    yield TestClient(app, raise_server_exceptions=False)
    # cleanup
    _DEFAULT_STATE.chunks = []
    _DEFAULT_STATE.docs   = []
    _DEFAULT_STATE.repo   = ""


# ── 1. Health ─────────────────────────────────────────────────────────────────

def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── 2. Chat without repo → 400 ────────────────────────────────────────────────

def test_chat_without_repo_returns_400(client):
    _DEFAULT_STATE.chunks = []
    resp = client.post("/chat", json={"query": "where is login?"})
    assert resp.status_code == 400


# ── 3. /load — nonexistent path → 400 ────────────────────────────────────────

def test_load_invalid_path_returns_400(client):
    resp = client.post("/load", json={"repo_path": "/nonexistent/path/xyz"})
    assert resp.status_code == 400


# ── 4. /load — path traversal blocked ────────────────────────────────────────

def test_load_path_traversal_blocked(client):
    resp = client.post("/load", json={"repo_path": "../etc/passwd"})
    assert resp.status_code == 400


# ── 5. /clone — non-GitHub URL blocked ───────────────────────────────────────

def test_clone_invalid_url_blocked(client):
    resp = client.post("/clone", json={"github_url": "https://evil.com/repo"})
    assert resp.status_code == 400


# ── 6. /clone — URL with @ sign blocked ──────────────────────────────────────

def test_clone_url_with_at_sign_blocked(client):
    resp = client.post("/clone",
                       json={"github_url": "https://github.com/x@evil.com/y"})
    assert resp.status_code == 400


# ── 7. /metrics endpoint ──────────────────────────────────────────────────────

def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_queries" in body
    assert "avg_latency_ms" in body


# ── 8. /impact — no repo → 400 ───────────────────────────────────────────────

def test_impact_no_repo_returns_400(client):
    _DEFAULT_STATE.chunks = []
    resp = client.post("/impact", json={"function_name": "login"})
    assert resp.status_code == 400
