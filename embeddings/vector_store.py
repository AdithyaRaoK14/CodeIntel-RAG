"""
embeddings/vector_store.py
──────────────────────────
FAISS vector store — now using BAAI/bge-base-en-v1.5.

Why the model was changed
─────────────────────────
The original microsoft/codebert-base is a masked language model fine-tuned
on code tasks (clone detection, code completion). When adapted by
SentenceTransformers it produces reasonable code-to-code similarity but has
near-zero ability to bridge natural-language queries to function names/code.
This is why the pure semantic baseline scored 0.0 on all 25 benchmark queries.

BAAI/bge-base-en-v1.5 is a retrieval-trained bi-encoder that tops the MTEB
leaderboard. It handles NL→Code retrieval correctly because it was trained on
diverse retrieval pairs including code-related QA. It is the same dimension
(768) so no FAISS index schema changes are needed.

If you want to stay code-specific, the next-best drop-in is:
  "flax-sentence-embeddings/st-codesearch-distilroberta-base"
which was trained explicitly on NL→Code pairs from CodeSearchNet.

Other changes (unchanged from original)
────────────────────────────────────────
• Vector store    : FAISS IndexFlatIP (exact cosine on normalised vectors)
• MAX_TOKENS      : 512
• Sliding window  : long functions averaged across windows, then re-normalised
• Persistence     : index + metadata saved to disk, reloaded on restart
• Logging         : replaced print() with proper logging
"""

import json
import logging
import os
import pickle
import threading

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

# ── Embedding model ───────────────────────────────────────────────────────────
# Changed from microsoft/codebert-base to BAAI/bge-base-en-v1.5.
# First run will download ~438 MB; subsequent runs use the HuggingFace cache.
#
# Alternative if you need a smaller model (~90 MB):
#   "BAAI/bge-small-en-v1.5"   — 384-dim, change _DIMENSION to 384
#
# Alternative if you want pure NL→Code specialisation:
#   "flax-sentence-embeddings/st-codesearch-distilroberta-base"  — 768-dim
_MODEL = SentenceTransformer("BAAI/bge-base-en-v1.5")
_DIMENSION = 768
_MAX_TOKENS = 512

# BGE models expect a query prefix for retrieval tasks
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── Disk paths ────────────────────────────────────────────────────────────────
_BASE = os.path.join(os.path.dirname(__file__), "..")
_IDX_PATH = os.path.join(_BASE, ".faiss_index.bin")
_META_PATH = os.path.join(_BASE, ".faiss_meta.pkl")


def _load_or_create_index() -> faiss.IndexFlatIP:
    if os.path.exists(_IDX_PATH):
        try:
            idx = faiss.read_index(_IDX_PATH)
            log.info("Loaded FAISS index (%d vectors)", idx.ntotal)
            return idx
        except Exception as exc:
            log.warning("Could not load FAISS index: %s — creating fresh", exc)
    return faiss.IndexFlatIP(_DIMENSION)


def _load_metadata() -> list[dict]:
    if os.path.exists(_META_PATH):
        try:
            with open(_META_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return []


_index = _load_or_create_index()
_metadata = _load_metadata()
_index_lock = threading.Lock()  # guards all reads and writes to _index / _metadata


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _embed_passage(text: str) -> np.ndarray:
    """Embed a code passage (document side — no prefix needed)."""
    tokenizer = _MODEL.tokenizer
    token_ids = tokenizer.encode(text)

    if len(token_ids) <= _MAX_TOKENS:
        vec = _MODEL.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)

    stride = _MAX_TOKENS // 2
    windows = []
    for start in range(0, len(token_ids), stride):
        chunk_ids = token_ids[start: start + _MAX_TOKENS]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        windows.append(_MODEL.encode(chunk_text, normalize_embeddings=True))
        if start + _MAX_TOKENS >= len(token_ids):
            break

    averaged = np.mean(windows, axis=0)
    return _normalize(averaged).astype(np.float32)


def _embed_query(query: str) -> np.ndarray:
    """
    Embed a natural-language query.

    BGE models are trained with an instruction prefix on the query side.
    Passage side should NOT use the prefix (already handled in _embed_passage).
    """
    prefixed = _QUERY_PREFIX + query
    vec = _MODEL.encode(prefixed, normalize_embeddings=True)
    return vec.astype(np.float32)


def _chunk_to_text(chunk: dict) -> str:
    calls_str = ", ".join(chunk.get("calls", []))
    return (
        f"File: {chunk['file']}\n"
        f"Language: {_detect_language(chunk['file'])}\n"
        f"Function: {chunk['name']}\n"
        f"Calls: {calls_str or 'none'}\n\n"
        f"{chunk.get('code', '')}"
    )


def _detect_language(file_name: str) -> str:
    ext_map = {
        ".py":   "Python",
        ".java": "Java",
        ".js":   "JavaScript",
        ".cpp":  "C++",
        ".ts":   "TypeScript",
    }
    for ext, lang in ext_map.items():
        if file_name.endswith(ext):
            return lang
    return "Unknown"


def _save_to_disk() -> None:
    try:
        faiss.write_index(_index, _IDX_PATH)
        with open(_META_PATH, "wb") as f:
            pickle.dump(_metadata, f)
    except Exception as exc:
        log.error("Could not save FAISS index to disk: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def store_chunks(chunks: list[dict]) -> None:
    if not chunks:
        return

    vectors = []
    meta = []

    for chunk in chunks:
        text = _chunk_to_text(chunk)
        vec = _embed_passage(text)       # passage side — no prefix
        vectors.append(vec)
        meta.append({
            "name":  chunk["name"],
            "file":  chunk["file"],
            "type":  chunk.get("type", "function"),
            "line":  chunk.get("line", 0),
            "calls": json.dumps(chunk.get("calls", [])),
            "code":  chunk.get("code", "")[:2000],
        })

    matrix = np.stack(vectors)

    with _index_lock:
        _index.add(matrix)
        _metadata.extend(meta)
        _save_to_disk()

    log.info("Stored %d chunks (%d total in index)",
             len(chunks), _index.ntotal)


def search_chunks(query: str, n: int = 5) -> list[dict]:
    with _index_lock:
        if _index.ntotal == 0:
            return []

        n_safe = min(n, _index.ntotal)
        query_vec = _embed_query(query).reshape(
            1, -1)   # query side — with prefix

        distances, indices = _index.search(query_vec, n_safe)
        snapshot = list(_metadata)   # safe copy while holding lock

    results = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(snapshot):
            continue
        meta = snapshot[idx]
        results.append({
            "name":     meta["name"],
            "file":     meta["file"],
            "type":     meta.get("type", "function"),
            "line":     meta.get("line", 0),
            "language": _detect_language(meta["file"]),
            "calls":    json.loads(meta.get("calls", "[]")),
            "code":     meta.get("code", ""),
        })

    return results


def keyword_search(chunks: list[dict], query: str) -> list[dict]:
    q = query.lower()
    return [
        c for c in chunks
        if q in c["name"].lower() or q in c["file"].lower()
    ]


def clear_collection() -> None:
    global _index, _metadata
    with _index_lock:
        _index = faiss.IndexFlatIP(_DIMENSION)
        _metadata = []

        for path in [_IDX_PATH, _META_PATH]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    log.info("FAISS index cleared")
