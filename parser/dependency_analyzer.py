"""
parser/dependency_analyzer.py
──────────────────────────────
Dependency / impact analysis on the in-memory chunk list.

Fixes applied
─────────────
• Handles calls as either a list OR a JSON-encoded string
  (vector_store stores them as JSON; in-memory chunks keep them as list)
• Added get_callees() – what does this function call?
• Added get_full_dependency_tree() – multi-hop impact analysis
"""

import json


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_calls(chunk: dict) -> list[str]:
    """Always return calls as a plain list, regardless of storage format."""
    calls = chunk.get("calls", [])
    if isinstance(calls, str):
        try:
            calls = json.loads(calls)
        except (json.JSONDecodeError, ValueError):
            calls = [c.strip() for c in calls.split(",") if c.strip()]
    return calls


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def find_function_usage(chunks: list[dict], target: str) -> list[dict]:
    """
    Return every function that CALLS `target`.
    (Callers / reverse dependency / impact analysis)
    """
    target_lower = target.lower()
    return [
        {"function": c["name"], "file": c["file"]}
        for c in chunks
        if any(call.lower() == target_lower for call in _get_calls(c))
    ]


def find_callees(chunks: list[dict], source: str) -> list[dict]:
    """
    Return every function that `source` CALLS.
    (Callees / forward dependency)
    """
    source_lower = source.lower()
    callee_names = set()

    for chunk in chunks:
        if chunk["name"].lower() == source_lower:
            callee_names.update(_get_calls(chunk))

    results = []
    for chunk in chunks:
        if chunk["name"] in callee_names:
            results.append({"function": chunk["name"], "file": chunk["file"]})
            callee_names.discard(chunk["name"])

    # also include names we found in calls but aren't in the repo
    for name in callee_names:
        results.append({"function": name, "file": "(external)"})

    return results


def get_full_dependency_tree(
    chunks: list[dict],
    target: str,
    max_depth: int = 3,
) -> dict:
    """
    Build a multi-hop impact tree.

    Returns a nested dict:
      {
        "function": "login",
        "callers": [
          {"function": "start", "file": "app.py", "callers": [...]}
        ]
      }
    """
    visited = set()

    def _build(name: str, depth: int) -> dict:
        node = {"function": name, "callers": []}
        if depth == 0 or name in visited:
            return node
        visited.add(name)
        for caller in find_function_usage(chunks, name):
            child = _build(caller["function"], depth - 1)
            child["file"] = caller["file"]
            node["callers"].append(child)
        return node

    return _build(target, max_depth)


def get_all_dependencies(chunks: list[dict]) -> dict[str, list[str]]:
    """
    Return a full adjacency map:  function_name → [functions it calls]
    Useful for building graphs.
    """
    return {c["name"]: _get_calls(c) for c in chunks}
