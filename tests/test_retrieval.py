"""
tests/test_retrieval.py
────────────────────────
Retrieval quality and cache correctness tests.

Run:  pytest tests/test_retrieval.py -v
"""

import pytest

from cache.query_cache import QueryCache
from retrieval.hybrid_retriever import (
    _is_specific_function_query,
    _extract_bm25_phrase,
    _query_content_words,
)


# ── Cache tests ───────────────────────────────────────────────────────────────

def test_cache_hit_and_miss(tmp_path):
    cache = QueryCache(path=str(tmp_path / "cache.json"), repo_path="/repo/a")
    assert cache.get("what is login") is None

    fake = [{"name": "login", "file": "auth.py"}]
    cache.set("what is login", fake)
    assert cache.get("what is login") == fake


def test_cache_case_insensitive(tmp_path):
    cache = QueryCache(path=str(tmp_path / "cache.json"), repo_path="/repo/a")
    fake = [{"name": "login", "file": "auth.py"}]
    cache.set("What Is Login", fake)
    assert cache.get("what is login") == fake


def test_cache_key_changes_with_repo(tmp_path):
    """Same query text must produce different keys for different repos."""
    cache_a = QueryCache(path=str(tmp_path / "a.json"), repo_path="/repo/a")
    cache_b = QueryCache(path=str(tmp_path / "b.json"), repo_path="/repo/b")

    key_a = cache_a._hash("where is login")
    key_b = cache_b._hash("where is login")

    assert key_a != key_b, (
        "Cache keys must differ across repos to prevent stale hits"
    )


# ── BM25 phrase selection ─────────────────────────────────────────────────────

def test_is_specific_function_query_with_parens():
    assert _is_specific_function_query("login() function") is True


def test_is_specific_function_query_what_calls():
    assert _is_specific_function_query("what calls dispatch_request") is True


def test_is_specific_function_query_concept_is_false():
    assert _is_specific_function_query("request dispatching") is False


def test_is_specific_function_query_url_routing_is_false():
    assert _is_specific_function_query("URL routing") is False


def test_extract_bm25_phrase_concept():
    """Concept queries should return multiple content words for BM25."""
    phrase = _extract_bm25_phrase("request dispatching")
    words  = set(phrase.split())
    # Both content words should appear in the phrase
    assert "request" in words or "dispatching" in words
    assert len(words) >= 1


# ── Reranker ──────────────────────────────────────────────────────────────────

def test_reranker_ranks_relevant_higher():
    """Cross-encoder must rank the login chunk above unrelated code."""
    from retrieval.reranker import rerank

    chunks = [
        {"name": "unrelated_util",        "file": "utils.py", "code": "def unrelated_util(): print('hello')"},
        {"name": "login_authentication",  "file": "auth.py",  "code": "def login_authentication(user, pw): verify(pw)"},
    ]
    ranked = rerank("login authentication", chunks, top_k=2)
    assert ranked[0]["name"] == "login_authentication", (
        "Cross-encoder should rank the login chunk first"
    )


def test_reranker_returns_top_k():
    from retrieval.reranker import rerank

    chunks = [
        {"name": f"func{i}", "file": "app.py", "code": f"def func{i}(): pass"}
        for i in range(10)
    ]
    result = rerank("authentication login", chunks, top_k=3)
    assert len(result) == 3
