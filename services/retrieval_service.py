"""
services/retrieval_service.py
──────────────────────────────
Two-stage retrieval logic extracted from api.py.

hybrid_search → rerank → cache read/write

CHANGES
───────
• No code change required here — rerank_candidates is read from settings, so
  bumping it to 20 in config.py is sufficient. File included for completeness.
"""

from __future__ import annotations

import json
import logging
import time

import cache.query_cache as query_cache
from config import settings
from retrieval.hybrid_retriever import hybrid_search
from retrieval.reranker import rerank

log = logging.getLogger(__name__)


def retrieve(query: str, state, metrics: dict | None = None) -> list[dict]:
    """Return reranked results for *query* against *state*.

    Parameters
    ----------
    query   : raw user query string
    state   : AppState (chunks + bm25_index)
    metrics : optional dict to accumulate telemetry counters (mutated in-place)
    """
    t0 = time.monotonic()
    if metrics is not None:
        metrics["total_queries"] = metrics.get("total_queries", 0) + 1

    cached = query_cache.get(query)
    if cached:
        log.info("Cache hit: %s", query[:50])
        if metrics is not None:
            metrics["cache_hits"] = metrics.get("cache_hits", 0) + 1
            metrics["total_latency_ms"] = (
                metrics.get("total_latency_ms", 0.0)
                + (time.monotonic() - t0) * 1000
            )
        return cached

    candidates = hybrid_search(query, state.chunks, state.bm25_index)
    # settings.rerank_candidates is now 20 (was 15) — see config.py
    results = rerank(query, candidates[:settings.rerank_candidates])
    query_cache.set(query, results)

    if metrics is not None:
        metrics["rerank_calls"] = metrics.get("rerank_calls", 0) + 1
        metrics["total_latency_ms"] = (
            metrics.get("total_latency_ms", 0.0)
            + (time.monotonic() - t0) * 1000
        )

    return results


def format_results(results: list[dict], top_k: int | None = None) -> list[dict]:
    """Normalise raw result dicts for JSON serialisation."""
    if top_k is None:
        top_k = settings.top_k
    out = []
    for r in results[:top_k]:
        calls = r.get("calls", [])
        if isinstance(calls, str):
            try:
                calls = json.loads(calls)
            except Exception:
                calls = [c.strip() for c in calls.split(",") if c.strip()]
        out.append({
            "name":     r.get("name", ""),
            "file":     r.get("file", ""),
            "line":     r.get("line", 0),
            "calls":    calls,
            "code":     r.get("code", ""),
            "language": r.get("language", ""),
        })
    return out
