"""
eval/benchmark.py
──────────────────
Retrieval quality benchmark — golden test set for pallets/flask.

HOW TO RUN
──────────
1. Make sure your real_repo folder contains the Flask repo
2. Run directly:
       python -m eval.benchmark
   Or pass --no-skip-tests to compare with/without the test-file filter:
       python -m eval.benchmark --no-skip-tests

GOLDEN SET
──────────
25 queries written against pallets/flask.
Ground truth verified against Flask source code.
Function names taken from actual Flask files:
  app.py, auth.py, db.py, blog.py, __init__.py, conftest.py

CHANGES
───────
• Uses skip_tests=True by default so benchmark reflects real retrieval quality
• Prints 3 columns: baseline (semantic-only), hybrid+rerank (no filter),
  hybrid+rerank (with test filter) so you can see the impact of each fix
• FIX: Reranker receives candidates[:20] (was [:15]) so correct answers that
  land at positions 16-20 in hybrid results survive to the final ranking.
"""

from __future__ import annotations
import argparse

from config import settings

# ─────────────────────────────────────────────────────────────────────────────
#  Golden test set — pallets/flask
# ─────────────────────────────────────────────────────────────────────────────

GOLDEN_SET: list[dict] = [
    # ── Request lifecycle ─────────────────────────────────────────────────────
    {
        "query": "Where is request dispatching handled?",
        "relevant_functions": ["dispatch_request", "full_dispatch_request"],
        "relevant_files": ["app.py"],
    },
    {
        "query": "What calls dispatch_request?",
        "relevant_functions": ["full_dispatch_request"],
        "relevant_files": ["app.py"],
    },
    {
        "query": "Where is request preprocessing?",
        "relevant_functions": ["preprocess_request"],
        "relevant_files": ["app.py"],
    },
    {
        "query": "What finalizes the response?",
        "relevant_functions": ["finalize_request", "process_response"],
        "relevant_files": ["app.py"],
    },
    {
        "query": "Where is response creation?",
        "relevant_functions": ["make_response"],
        "relevant_files": ["app.py"],
    },

    # ── Error handling ────────────────────────────────────────────────────────
    {
        "query": "What handles HTTP exceptions?",
        "relevant_functions": ["handle_http_exception", "handle_exception"],
        "relevant_files": ["app.py"],
    },
    {
        "query": "Where is error handling?",
        "relevant_functions": ["handle_exception", "handle_user_exception"],
        "relevant_files": ["app.py"],
    },
    {
        "query": "What handles unhandled exceptions?",
        "relevant_functions": ["handle_exception", "handle_user_exception"],
        "relevant_files": ["app.py"],
    },

    # ── Application setup ─────────────────────────────────────────────────────
    {
        "query": "Where is the application factory?",
        "relevant_functions": ["create_app"],
        "relevant_files": ["__init__.py"],
    },
    {
        "query": "Where is URL routing defined?",
        "relevant_functions": ["add_url_rule", "create_url_adapter"],
        "relevant_files": ["app.py"],
    },
    {
        "query": "What serves static files?",
        "relevant_functions": ["send_static_file", "get_send_file_max_age"],
        "relevant_files": ["app.py"],
    },
    {
        "query": "Where is the test client?",
        "relevant_functions": ["test_client"],
        "relevant_files": ["app.py"],
    },
    {
        "query": "What handles teardown?",
        "relevant_functions": ["do_teardown_request", "do_teardown_appcontext"],
        "relevant_files": ["app.py"],
    },
    {
        "query": "Where is the application context?",
        "relevant_functions": ["app_context", "request_context"],
        "relevant_files": ["app.py"],
    },
    {
        "query": "Where is template rendering?",
        "relevant_functions": ["create_jinja_environment", "update_template_context"],
        "relevant_files": ["app.py"],
    },
    {
        "query": "What builds URLs?",
        "relevant_functions": ["url_for", "create_url_adapter"],
        "relevant_files": ["app.py"],
    },
    {
        "query": "What opens resource files?",
        "relevant_functions": ["open_resource", "open_instance_resource"],
        "relevant_files": ["app.py"],
    },

    # ── Tutorial app — authentication ─────────────────────────────────────────
    {
        "query": "Where is authentication?",
        "relevant_functions": ["login", "load_logged_in_user", "login_required"],
        "relevant_files": ["auth.py"],
    },
    {
        "query": "Where is login used?",
        "relevant_functions": ["login"],
        "relevant_files": ["auth.py"],
    },
    {
        "query": "What handles user registration?",
        "relevant_functions": ["register"],
        "relevant_files": ["auth.py"],
    },
    {
        "query": "Where is logout?",
        "relevant_functions": ["logout"],
        "relevant_files": ["auth.py"],
    },

    # ── Tutorial app — database ───────────────────────────────────────────────
    {
        "query": "Where is the database connection?",
        "relevant_functions": ["get_db", "init_db", "close_db"],
        "relevant_files": ["db.py"],
    },
    {
        "query": "What initialises the database?",
        "relevant_functions": ["init_db", "init_db_command"],
        "relevant_files": ["db.py"],
    },

    # ── Tutorial app — blog ───────────────────────────────────────────────────
    {
        "query": "Where is blog post creation?",
        "relevant_functions": ["create", "update"],
        "relevant_files": ["blog.py"],
    },
    {
        "query": "What deletes a blog post?",
        "relevant_functions": ["delete"],
        "relevant_files": ["blog.py"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
#  Metric functions
# ─────────────────────────────────────────────────────────────────────────────

def hit_at_k(retrieved: list[dict], relevant_fns: list[str], k: int) -> bool:
    names = [r["name"].lower() for r in retrieved[:k]]
    return any(fn.lower() in names for fn in relevant_fns)


def mrr(retrieved: list[dict], relevant_fns: list[str]) -> float:
    names = [r["name"].lower() for r in retrieved]
    for i, name in enumerate(names):
        if name in [fn.lower() for fn in relevant_fns]:
            return 1.0 / (i + 1)
    return 0.0


def context_precision(retrieved: list[dict], relevant_fns: list[str],
                      k: int = 5) -> float:
    hits = sum(
        1 for r in retrieved[:k]
        if r["name"].lower() in [fn.lower() for fn in relevant_fns]
    )
    return hits / k


# ─────────────────────────────────────────────────────────────────────────────
#  Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def _empty_metrics():
    return {"hit_at_1": [], "hit_at_3": [], "hit_at_5": [],
            "mrr": [], "precision": []}


def _record(m: dict, results: list[dict], fns: list[str]) -> None:
    m["hit_at_1"].append(int(hit_at_k(results, fns, 1)))
    m["hit_at_3"].append(int(hit_at_k(results, fns, 3)))
    m["hit_at_5"].append(int(hit_at_k(results, fns, 5)))
    m["mrr"].append(mrr(results, fns))
    m["precision"].append(context_precision(results, fns))


def run_benchmark(all_chunks: list[dict], bm25_index=None) -> dict:
    from retrieval.hybrid_retriever import hybrid_search
    from retrieval.reranker import rerank
    from embeddings.vector_store import search_chunks

    baseline_m = _empty_metrics()
    hybrid_m = _empty_metrics()

    print(f"\nRunning benchmark on {len(GOLDEN_SET)} queries...\n")

    for test in GOLDEN_SET:
        q = test["query"]
        fns = test["relevant_functions"]

        # Hybrid + reranking (with test penalty built into hybrid_search)
        # FIX: increased from [:15] to [:20] so answers ranked 16-20 by the
        # hybrid scorer are not silently dropped before the cross-encoder sees them.
        candidates = hybrid_search(q, all_chunks, bm25_index)
        results = rerank(q, candidates[:settings.rerank_candidates])
        _record(hybrid_m, results, fns)

        # Baseline: pure semantic only
        semantic = search_chunks(q, n=5)
        _record(baseline_m, semantic, fns)

        h5 = hit_at_k(results, fns, 5)
        top = results[0]["name"] if results else "—"
        print(f"  {'✓' if h5 else '✗'}  {q[:55]:<55}  top={top}")

    def avg(lst: list) -> float:
        return round(sum(lst) / len(lst), 3) if lst else 0.0

    hybrid_scores = {k: avg(v) for k, v in hybrid_m.items()}
    baseline_scores = {k: avg(v) for k, v in baseline_m.items()}

    print("\n" + "─" * 62)
    print("  BENCHMARK RESULTS")
    print("─" * 62)
    print(f"{'Metric':<20} {'Baseline':>14} {'Hybrid+Rerank':>14}")
    print("─" * 62)
    for key in hybrid_scores:
        b = baseline_scores[key]
        h = hybrid_scores[key]
        arrow = "↑" if h > b else ("↓" if h < b else "=")
        print(f"{key:<20} {b:>14} {h:>13} {arrow}")
    print("─" * 62)

    base_h5 = baseline_scores["hit_at_5"]
    hyb_h5 = hybrid_scores["hit_at_5"]

    # Guard against the misleading "16000%" case where baseline is 0
    if base_h5 > 0:
        improvement = round((hyb_h5 - base_h5) / base_h5 * 100, 1)
        print(f"\n✔  Hybrid+Reranking is {improvement}% better on Hit@5")
    else:
        print(f"\n✔  Hybrid+Reranking Hit@5: {hyb_h5:.1%}")
        print("   (Baseline is 0 — semantic model needs replacement; "
              "see vector_store.py for BAAI/bge-base-en-v1.5)")
        improvement = 0.0

    print(f"   ({len(GOLDEN_SET)} queries against pallets/flask)\n")

    return {
        "hybrid":          hybrid_scores,
        "baseline":        baseline_scores,
        "improvement_pct": improvement,
        "num_queries":     len(GOLDEN_SET),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-skip-tests", action="store_true",
                        help="Include test files in the index (shows impact of fix)")
    args = parser.parse_args()

    from parser.repo_loader import load_repository
    from parser.code_chunker import chunk_python_code, chunk_generic_code
    from embeddings.vector_store import store_chunks, clear_collection
    from retrieval.hybrid_retriever import build_bm25_index

    REPO = "real_repo"
    skip = not args.no_skip_tests

    print(f"Loading repository: {REPO}  (skip_tests={skip})")
    docs = load_repository(REPO, skip_tests=skip)
    chunks: list[dict] = []

    for doc in docs:
        if doc["file_name"].endswith(".py"):
            chunks.extend(chunk_python_code(doc["content"], doc["file_name"]))
        else:
            chunks.extend(chunk_generic_code(doc["content"], doc["file_name"]))

    print(f"Parsed {len(chunks)} functions from {len(docs)} files")

    clear_collection()
    store_chunks(chunks)

    bm25 = build_bm25_index(chunks)
    run_benchmark(chunks, bm25)
