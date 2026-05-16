"""
services/indexing_service.py
─────────────────────────────
Repository loading + indexing extracted from api.py.

Accepting `state` as an explicit parameter (no global access) makes this
logic unit-testable and reusable outside the HTTP layer.
"""

from __future__ import annotations

import logging

import cache.query_cache as query_cache
from embeddings.vector_store import clear_collection, store_chunks
from parser.code_chunker import chunk_generic_code, chunk_python_code
from parser.repo_loader import load_repository
from retrieval.hybrid_retriever import build_bm25_index

log = logging.getLogger(__name__)


def load_repo(repo_path: str, state) -> None:
    """Parse, embed, and index a repository into *state*.

    Parameters
    ----------
    repo_path : absolute (already validated) path to the repository root
    state     : AppState instance to populate (passed explicitly, not global)
    """
    docs = load_repository(repo_path)
    chunks: list[dict] = []

    for doc in docs:
        if doc["file_name"].endswith(".py"):
            chunks.extend(chunk_python_code(doc["content"], doc["file_name"]))
        else:
            chunks.extend(chunk_generic_code(doc["content"], doc["file_name"]))

    clear_collection()
    store_chunks(chunks)
    query_cache.clear()
    query_cache._CACHE._repo = repo_path   # scope cache keys to this repo

    bm25 = build_bm25_index(chunks)

    with state._lock:
        state.docs       = docs
        state.chunks     = chunks
        state.repo       = repo_path
        state.bm25_index = bm25

    log.info("Indexed %s  (%d functions across %d files)",
             repo_path, len(chunks), len(docs))
