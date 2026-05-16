"""
cache/query_cache.py
─────────────────────
Disk-backed query result cache.

Why this matters:
  Embedding + BM25 + reranking costs ~200-400 ms per query.
  Repeated queries (very common in demos) return instantly from cache.
  TTL defaults to 1 hour — re-embeds only after repo changes.
"""

import hashlib
import json
import os
import time


class QueryCache:
    def __init__(self, path: str = ".query_cache.json", ttl: int = 3600,
                 repo_path: str = ""):
        self._path = path
        self._ttl = ttl
        self._repo = repo_path          # scopes keys so the same query text
        # returns different results per repo
        self._cache = self._load()

    def get(self, query: str) -> list[dict] | None:
        """Return cached results for query, or None if not found / expired."""
        key = self._hash(query)
        entry = self._cache.get(key)
        if entry and time.time() - entry["ts"] < self._ttl:
            return entry["results"]
        return None

    def set(self, query: str, results: list[dict]) -> None:
        """Store results for query."""
        key = self._hash(query)
        self._cache[key] = {"results": results, "ts": time.time()}
        self._save()

    def clear(self) -> None:
        """Wipe all cached results.  Call after loading a new repo."""
        self._cache = {}
        self._save()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _hash(self, query: str) -> str:
        """Case-insensitive MD5 key, scoped to the current repo.

        Prevents stale hits when a different repo is loaded without restarting
        the process.  The repo path is included in the hash input so the same
        query string produces a different key for each repo.
        """
        key_input = f"{self._repo}::{query.lower().strip()}"
        return hashlib.md5(key_input.encode()).hexdigest()

    def _load(self) -> dict:
        try:
            with open(self._path) as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self) -> None:
        try:
            with open(self._path, "w") as f:
                json.dump(self._cache, f)
        except Exception:
            pass


# ── Module-level singleton ────────────────────────────────────────────────────
_CACHE = QueryCache()


def get(query: str) -> list[dict] | None:
    return _CACHE.get(query)


def set(query: str, results: list[dict]) -> None:
    _CACHE.set(query, results)


def clear() -> None:
    _CACHE.clear()
