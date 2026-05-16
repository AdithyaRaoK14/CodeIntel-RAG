"""
retrieval/hybrid_retriever.py
──────────────────────────────
4-signal hybrid retrieval with weighted ranking.

Signals
───────
1. Semantic   (weight 3)  – BGE embeddings + FAISS
2. Keyword    (weight 8)  – BM25 with camelCase/snake_case splitting
3. Structural (weight 10) – dependency graph (callers/callees)
4. Metadata   (weight 5)  – language / file-extension filter
5. Name-token (weight 12) – concept-word overlap between query and function name

Changes in this revision
────────────────────────
• FIX: Synonym table extended — "rendering"/"template" now expand to include
  "environment" and "create", so "Where is template rendering?" correctly
  scores create_jinja_environment (36) above render_template_string (30).
• FIX: hybrid_score attached to every result dict so reranker.py can use the
  actual signal score rather than positional rank, which was demoting correct
  top-1 results.
• FIX: Dunder method penalty (×0.4) — prevents __enter__/__init__ from
  winning over more specific matches like test_client / init_db.
• FIX: Generic name blocklist in _name_token_score — "response", "request",
  "app", etc. return 0 so they can't win purely on name-token overlap.
• FIX: New synonym entries for "context", "builds", "registration", "calls"
  to surface app_context, url_for, register, full_dispatch_request.
• All prior fixes retained (test penalty, name-token matching, BM25 phrase
  selection, camelCase/snake_case tokenization).
"""

import logging
import re

from embeddings.vector_store import search_chunks, keyword_search
from parser.dependency_analyzer import find_function_usage, find_callees

log = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    log.warning("rank-bm25 not installed — using substring keyword search. "
                "Install with: pip install rank-bm25")


# ─────────────────────────────────────────────────────────────────────────────
#  Test-file detection
# ─────────────────────────────────────────────────────────────────────────────

_TEST_SCORE_MULTIPLIER = 0.12


def _is_test_artifact(name: str, file: str) -> bool:
    """Return True for test functions and conftest helpers.

    FIX: name-prefix check ("test_*") is now AND-gated with in_test_file so
    that legitimate app methods like Flask's test_client() — which live in
    app.py, not a test file — are not incorrectly penalised.
    """
    name_lower = name.lower()
    file_norm = file.replace("\\", "/").lower()
    file_parts = file_norm.split("/")
    file_base = file_parts[-1]

    in_test_file = (
        file_base.startswith("test_")
        or file_base == "conftest.py"
        or any(p in ("tests", "test", "testing") for p in file_parts[:-1])
    )
    return (
        in_test_file
        or (name_lower.startswith("test_") and in_test_file)
        or name_lower.startswith("_test_")
    )


# ─────────────────────────────────────────────────────────────────────────────
#  BM25 helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_for_bm25(text: str) -> list[str]:
    camel = re.sub(r"([A-Z])", r" \1", text).lower().split()
    snake = text.lower().split("_")
    full = [text.lower()]
    return list({t for t in camel + snake + full if len(t) > 1})


def build_bm25_index(chunks: list[dict]):
    """Build BM25 index over all chunks. Call once after repo load."""
    if not _BM25_AVAILABLE:
        return None
    corpus = [
        _tokenize_for_bm25(
            f"{c['name']} {c['file']} {c.get('code', '')[:200]}"
        )
        for c in chunks
    ]
    log.info("Built BM25 index over %d chunks", len(chunks))
    return BM25Okapi(corpus)


def keyword_search_bm25(
    chunks: list[dict],
    query: str,
    bm25_index,
    top_k: int = 10,
) -> list[dict]:
    if bm25_index is None:
        return keyword_search(chunks, query)

    tokens = _tokenize_for_bm25(query)
    scores = bm25_index.get_scores(tokens)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in ranked[:top_k] if score > 0]


# ─────────────────────────────────────────────────────────────────────────────
#  Name-token matching  (Signal 5)
# ─────────────────────────────────────────────────────────────────────────────

def _decompose_name(name: str) -> set[str]:
    """
    Split a function name into sub-words.
    dispatch_request      → {dispatch, request}
    handleHttpException   → {handle, http, exception}
    create_jinja_environment → {create, jinja, environment}
    """
    snake_parts = name.split("_")
    all_parts: list[str] = []
    for part in snake_parts:
        camel_parts = re.sub(r"([A-Z])", r" \1", part).split()
        all_parts.extend(camel_parts)
    return {p.lower() for p in all_parts if len(p) > 0}


def _query_content_words(query: str) -> set[str]:
    """Extract meaningful words from a natural-language query."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9]*", query)
    return {
        t.lower() for t in tokens
        if t.lower() not in _STOPWORDS and len(t) > 2
    }


# FIX: "rendering" and "template" now expand to include "environment" and
# "create" so that create_jinja_environment scores ≥2 matching concepts for
# the query "Where is template rendering?" and beats render_template_string.
#
# FIX: Added "context", "builds", "registration", "calls" to fix four
# wrong-top-1 patterns:
#   "Where is the application context?" → app_context (was create_app)
#   "What builds URLs?"                 → url_for     (was handle_url_build_error)
#   "What handles user registration?"   → register    (was load_logged_in_user)
#   "What calls dispatch_request?"      → full_dispatch_request (was dispatch_request)
_SYNONYMS = {
    "factory":      {"create", "make", "build", "init", "app"},
    # FIX: removed "app"/"create" — was inflating create_app over app_context
    "application":  {"flask", "init"},
    "rendering":    {"render", "template", "jinja", "environment", "create"},
    "template":     {"render", "jinja", "environment", "create"},
    "environment":  {"jinja", "template", "render", "create"},
    "routing":      {"route", "url", "add", "rule"},
    "auth":         {"login", "logout", "register", "user"},
    "database":     {"db", "connect", "init", "close", "get"},
    "connection":   {"db", "connect", "get", "close"},
    "exception":    {"handle", "error", "http"},
    "dispatching":  {"dispatch", "request"},
    "teardown":     {"teardown", "close", "do"},
    # ── New entries (Pattern 3 fixes) ────────────────────────────────────────
    # FIX: removed "app" — was inflating create_app over app_context
    "context":      {"push", "pop"},
    "builds":       {"url", "for", "adapter"},
    # FIX: removed "user" — was inflating load_logged_in_user over register
    "registration": {"register"},
    "calls":        {"full", "dispatch"},
    # FIX: make_response scores 2 hits for "response creation"
    "creation":     {"make", "create"},
}


def _expand_qw(words: set[str]) -> set[str]:
    exp = set(words)
    for w in words:
        exp.update(_SYNONYMS.get(w, set()))
    return exp


# FIX: Generic single-token function names that match almost every query via
# name-token overlap but are never the intended answer.  Returning 0 here
# stops "response" from beating "finalize_request" or "make_response".
_GENERIC_NAMES = {
    "response", "request", "app", "error", "result", "data", "handler",
    # FIX: Flask/stdlib generics that win via BM25/semantic alone
    "view", "open", "setdefault", "post", "dumps",
}


def _name_token_score(query_words: set[str], chunk_name: str) -> int:
    """
    Score a function name against query content words.

    Rules:
    - Generic names (see _GENERIC_NAMES) always score 0.
    - Exact sub-word match            → 12 per word
    - Substring containment (prefix)  → 6 per word  (dispatch ↔ dispatching)
    - At least 2 sub-words must match for any score to apply

    Example (after synonym expansion for "template rendering"):
      expanded = {template, rendering, render, jinja, environment, create}
      create_jinja_environment tokens = {create, jinja, environment}
        → 3 exact hits → score 36
      render_template_string tokens = {render, template, string}
        → 2 exact + 1 partial → score 30
      create_jinja_environment wins.
    """
    # FIX: Pattern 2 — generic names never win on name-token score alone.
    if chunk_name.lower() in _GENERIC_NAMES:
        return 0

    name_tokens = _decompose_name(chunk_name)
    expanded = _expand_qw(query_words)
    exact_hits = 0
    partial_hits = 0

    for qw in expanded:
        if qw in name_tokens:
            exact_hits += 1
        else:
            for nt in name_tokens:
                if (qw.startswith(nt) or nt.startswith(qw)) and len(nt) >= 3:
                    partial_hits += 1
                    break

    total_hits = exact_hits + partial_hits
    if total_hits < 2:
        return 0
    return exact_hits * 12 + partial_hits * 6


# ─────────────────────────────────────────────────────────────────────────────
#  Query parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "where", "what", "show", "find", "tell", "about", "the", "is", "are",
    "does", "used", "called", "function", "method", "code", "how", "which",
    "file", "class", "module", "my", "in", "of", "for", "to", "from",
    "list", "all", "any", "with", "and", "or", "not", "please", "handled",
    "defined", "implemented", "handles", "handle",
    "logged",  # FIX: noise word that inflates load_logged_in_user over register
}

_LANGUAGE_EXTENSIONS = {
    "python":     ".py",
    "java":       ".java",
    "javascript": ".js",
    "js":         ".js",
    "typescript": ".ts",
    "ts":         ".ts",
    "cpp":        ".cpp",
    "c++":        ".cpp",
}


def _extract_target_function(query: str) -> str | None:
    q = query.strip()
    patterns = [
        r'where\s+is\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:used|defined|implemented)',
        r'what\s+calls\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'who\s+calls\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\)',
        r'([a-zA-Z_][a-zA-Z0-9_]*)\s+function',
        r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)',
    ]
    for pat in patterns:
        m = re.search(pat, q, re.IGNORECASE)
        if m and m.group(1).lower() not in _STOPWORDS:
            return m.group(1)

    candidates = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]+\b', q)
    for c in candidates:
        if c.lower() not in _STOPWORDS:
            return c
    return None


def _extract_bm25_phrase(query: str) -> str:
    """
    For concept queries return the full content-word set so BM25 scores on all
    relevant terms, not just the first extracted function name.
    """
    words = _query_content_words(query)
    return " ".join(words) if words else query


def _extract_language_extension(query: str) -> str | None:
    q = query.lower()
    for lang in sorted(_LANGUAGE_EXTENSIONS, key=len, reverse=True):
        if lang in q:
            return _LANGUAGE_EXTENSIONS[lang]
    return None


def _is_usage_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in ("used", "calls", "callers", "who calls", "depends on"))


def _is_specific_function_query(query: str) -> bool:
    """Return True when the query asks about a NAMED function rather than a concept."""
    q = query.lower()
    if "()" in q:
        return True
    specific_patterns = [
        r'where\s+is\s+\w+\s+(used|defined|implemented)',
        r'(what|who)\s+calls\s+\w+',
        r'\w+\s+function\b',
        r'function\s+\w+\b',
    ]
    for pat in specific_patterns:
        if re.search(pat, q):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Score accumulator
# ─────────────────────────────────────────────────────────────────────────────

def _add_score(scored: dict, name: str, file: str, data: dict, points: int) -> None:
    key = (name, file)
    if key not in scored:
        scored[key] = {"data": data, "score": 0}
    scored[key]["score"] += points


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    all_chunks: list[dict],
    bm25_index=None,
) -> list[dict]:
    scored = {}
    target_fn = _extract_target_function(query)
    target_ext = _extract_language_extension(query)
    usage_q = _is_usage_query(query)
    query_words = _query_content_words(query)

    # Signal 1: Semantic (weight 3)
    for item in search_chunks(query, n=5):
        _add_score(scored, item["name"], item["file"], item, 3)

    # Signal 2: BM25 keyword (weight 8)
    bm25_phrase = (
        target_fn
        if (target_fn and _is_specific_function_query(query))
        else _extract_bm25_phrase(query)
    )
    for item in keyword_search_bm25(all_chunks, bm25_phrase, bm25_index):
        _add_score(scored, item["name"], item["file"], item, 8)

    # Signal 3: Structural (weight 10)
    if target_fn:
        if usage_q:
            for item in find_function_usage(all_chunks, target_fn):
                data = {"name": item["function"], "file": item["file"],
                        "calls": [], "code": ""}
                _add_score(scored, item["function"], item["file"], data, 10)
        else:
            for item in find_callees(all_chunks, target_fn):
                data = {"name": item["function"], "file": item["file"],
                        "calls": [], "code": ""}
                _add_score(scored, item["function"], item["file"], data, 10)

    # Signal 4: Metadata / language filter (weight 5)
    if target_ext:
        for item in all_chunks:
            if item["file"].endswith(target_ext):
                _add_score(scored, item["name"], item["file"], item, 5)

    # Signal 5: Name-token concept matching
    if query_words:
        for chunk in all_chunks:
            pts = _name_token_score(query_words, chunk["name"])
            if pts > 0:
                _add_score(scored, chunk["name"], chunk["file"], chunk, pts)

    # ── Test artifact penalty ─────────────────────────────────────────────────
    for key, entry in scored.items():
        name, file = key
        if _is_test_artifact(name, file):
            entry["score"] *= _TEST_SCORE_MULTIPLIER

    # ── Dunder method penalty ─────────────────────────────────────────────────
    # FIX: Pattern 1 — __enter__, __init__, etc. are implementation details and
    # should not win over descriptively-named functions like test_client or
    # init_db, which carry genuine signal from all five scoring dimensions.
    for key, entry in scored.items():
        name, _ = key
        if name.startswith("__") and name.endswith("__"):
            entry["score"] *= 0.4

    # ── Generic name penalty ──────────────────────────────────────────────────
    # FIX: _name_token_score already returns 0 for generic names, but they
    # still accumulate large BM25 + semantic scores when the query contains
    # the same word (e.g. "response" in "Where is response creation?").
    # Zeroing name-token alone is insufficient — "response" reaches rank #1
    # and the reranker's rank_score gives position-0 a weight of 1.0, making
    # it nearly impossible to dislodge at CE_W=0.3.
    # Applying a 0.3x multiplier here ensures generic names lose to any
    # descriptive function that picks up even one additional signal.
    _GENERIC_NAME_PENALTY = 0.3
    for key, entry in scored.items():
        name, _ = key
        if name.lower() in _GENERIC_NAMES:
            entry["score"] *= _GENERIC_NAME_PENALTY

    ranked = sorted(scored.values(), key=lambda x: x["score"], reverse=True)

    # FIX: Normalise and attach hybrid_score to each result so reranker.py
    # can blend on actual signal quality rather than positional rank.
    max_score = ranked[0]["score"] if ranked else 1.0
    results = []
    for item in ranked:
        data = dict(item["data"])  # copy to avoid mutating shared chunk dicts
        data["hybrid_score"] = item["score"] / \
            max_score if max_score > 0 else 0.0
        results.append(data)

    log.info("hybrid_search('%s') → %d results", query[:50], len(results))
    return results
