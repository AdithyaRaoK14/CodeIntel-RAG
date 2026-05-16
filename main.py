"""
main.py  –  CLI entry point for Codebase Intelligence Platform
──────────────────────────────────────────────────────────────
Usage:
    python main.py
    python main.py --repo /path/to/your/repo
    python main.py --repo /path/to/your/repo --query "Where is login used?"
"""

import argparse
import sys

from parser.repo_loader import load_repository
from parser.code_chunker import chunk_python_code, chunk_generic_code
from parser.dependency_analyzer import find_function_usage, get_all_dependencies
from embeddings.vector_store import store_chunks, clear_collection
from retrieval.hybrid_retriever import hybrid_search
from generator.response_generator import generate_answer


# ─────────────────────────────────────────────────────────────────────────────
#  CLI args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Codebase Intelligence Platform – CLI"
    )
    parser.add_argument(
        "--repo",
        default="sample_repo",
        help="Path to the repository to analyse (default: sample_repo)",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Question to ask about the codebase (optional; prompts if omitted)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  CODEBASE INTELLIGENCE PLATFORM")
    print(f"{'═'*60}")

    try:
        docs = load_repository(args.repo)
    except ValueError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    if not docs:
        print("\n[ERROR] No supported source files found in repository.")
        sys.exit(1)

    # ── 2. Parse ──────────────────────────────────────────────────────────────
    all_chunks: list[dict] = []

    for doc in docs:
        if doc["file_name"].endswith(".py"):
            chunks = chunk_python_code(doc["content"], doc["file_name"])
        else:
            chunks = chunk_generic_code(doc["content"], doc["file_name"])
        all_chunks.extend(chunks)

    print(f"\n✔ Total functions parsed: {len(all_chunks)}")

    # ── 3. Function summary ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("PARSED FUNCTIONS")
    print(f"{'─'*60}")
    for chunk in all_chunks:
        calls_str = ", ".join(chunk.get("calls", [])) or "—"
        print(
            f"  {chunk['name']:<25}  {chunk['file']:<30}  "
            f"line {chunk.get('line', '?'):<5}  calls: {calls_str}"
        )

    # ── 4. Dependency analysis (example: all functions) ───────────────────────
    print(f"\n{'─'*60}")
    print("DEPENDENCY MAP")
    print(f"{'─'*60}")
    dep_map = get_all_dependencies(all_chunks)
    for fn, calls in dep_map.items():
        if calls:
            print(f"  {fn} → {', '.join(calls)}")

    # ── 5. Embed ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Embedding chunks into vector store…")
    clear_collection()
    store_chunks(all_chunks)
    print("✔ Embedding complete.")

    # ── 6. Query loop ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("QUERY ENGINE  (type 'exit' to quit)")
    print(f"{'─'*60}")

    while True:
        if args.query:
            query = args.query
            args.query = None   # only auto-run once; then enter loop
        else:
            query = input("\n❓ Enter query: ").strip()

        if query.lower() in ("exit", "quit", "q", ""):
            print("Bye.")
            break

        results = hybrid_search(query, all_chunks)
        answer = generate_answer(query, results)

        print(f"\n📌 ANSWER:\n  {answer}")

        if results:
            print(f"\n📋 TOP RESULTS ({len(results)} found):")
            for i, r in enumerate(results[:5], 1):
                calls = r.get("calls", [])
                print(
                    f"  {i}. {r.get('name', '?')}() "
                    f"in {r.get('file', '?')}  "
                    f"calls: {', '.join(calls) or '—'}"
                )

        # Show callers for top result
        if results:
            top_fn = results[0].get("name", "")
            callers = find_function_usage(all_chunks, top_fn)
            if callers:
                print(f"\n🔗 CALLERS of {top_fn}():")
                for c in callers:
                    print(f"  {c['function']}() in {c['file']}")


if __name__ == "__main__":
    main()
