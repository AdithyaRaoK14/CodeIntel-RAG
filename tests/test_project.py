"""
test_project.py
───────────────
Unit tests for the Codebase Intelligence Platform.

Run:  pytest test_project.py -v
"""

import pytest

from parser.code_chunker import chunk_python_code, chunk_generic_code
from parser.dependency_analyzer import find_function_usage, find_callees
from retrieval.hybrid_retriever import (
    _extract_target_function,
    _extract_language_extension,
    _is_usage_query,
)


# ─────────────────────────────────────────────────────────────────────────────
#  1. Python chunker — extracts function name and file
# ─────────────────────────────────────────────────────────────────────────────

def test_python_chunker_extracts_function():
    code = "def login(username, password):\n    return True"
    chunks = chunk_python_code(code, "auth.py")
    assert len(chunks) == 1
    assert chunks[0]["name"] == "login"
    assert chunks[0]["file"] == "auth.py"


# ─────────────────────────────────────────────────────────────────────────────
#  2. Python chunker — extracts called functions
# ─────────────────────────────────────────────────────────────────────────────

def test_python_chunker_extracts_calls():
    code = "def start():\n    login()\n    connect_db()"
    chunks = chunk_python_code(code, "app.py")
    assert "login" in chunks[0]["calls"]
    assert "connect_db" in chunks[0]["calls"]


# ─────────────────────────────────────────────────────────────────────────────
#  3. Python chunker — multiple functions in one file
# ─────────────────────────────────────────────────────────────────────────────

def test_python_chunker_multiple_functions():
    code = (
        "def login():\n    pass\n\n"
        "def logout():\n    pass\n\n"
        "def register():\n    login()\n"
    )
    chunks = chunk_python_code(code, "auth.py")
    names = [c["name"] for c in chunks]
    assert "login" in names
    assert "logout" in names
    assert "register" in names


# ─────────────────────────────────────────────────────────────────────────────
#  4. Dependency analyzer — finds correct callers
# ─────────────────────────────────────────────────────────────────────────────

def test_find_function_usage_returns_callers():
    chunks = [
        {"name": "start",  "file": "app.py",  "calls": ["login"]},
        {"name": "init",   "file": "main.py", "calls": ["login"]},
        {"name": "login",  "file": "auth.py", "calls": []},
    ]
    callers = find_function_usage(chunks, "login")
    names = [c["function"] for c in callers]
    assert "start" in names
    assert "init" in names
    assert "login" not in names


# ─────────────────────────────────────────────────────────────────────────────
#  5. Dependency analyzer — no false positives
# ─────────────────────────────────────────────────────────────────────────────

def test_find_function_usage_no_false_positives():
    chunks = [
        {"name": "connect_db", "file": "db.py", "calls": []},
        {"name": "close_db",   "file": "db.py", "calls": []},
    ]
    callers = find_function_usage(chunks, "login")
    assert callers == []


# ─────────────────────────────────────────────────────────────────────────────
#  6. Query parser — extracts target function name
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("query, expected", [
    ("Where is login used?",              "login"),
    ("What calls dispatch_request?",      "dispatch_request"),
    ("Where is connect_db defined?",      "connect_db"),
    ("login() function",                  "login"),
    ("Show me the register function",     "register"),
])
def test_extract_target_function(query, expected):
    result = _extract_target_function(query)
    assert result is not None
    assert result.lower() == expected.lower()


# ─────────────────────────────────────────────────────────────────────────────
#  7. Query parser — language extension extraction
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("query, expected_ext", [
    ("Show me Python functions",         ".py"),
    ("Find Java authentication",         ".java"),
    ("Where is JavaScript routing",      ".js"),
    ("Show TypeScript interfaces",       ".ts"),
    ("Where is C++ memory management",   ".cpp"),
])
def test_extract_language_extension(query, expected_ext):
    result = _extract_language_extension(query)
    assert result == expected_ext, (
        f"Query '{query}': expected '{expected_ext}', got '{result}'"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  8. Query classifier — usage vs definition queries
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("query, expected", [
    ("Where is login used?",         True),
    ("What calls dispatch_request?", True),
    ("Who calls connect_db?",        True),
    ("Where is authentication?",     False),
    ("Show me login function",       False),
])
def test_is_usage_query(query, expected):
    assert _is_usage_query(query) == expected


# ─────────────────────────────────────────────────────────────────────────────
#  9. Generic chunker — Java function extraction
# ─────────────────────────────────────────────────────────────────────────────

def test_java_chunker_extracts_function():
    code = (
        "public class AuthService {\n"
        "    public boolean loginUser(String username) {\n"
        "        return true;\n"
        "    }\n"
        "}\n"
    )
    chunks = chunk_generic_code(code, "AuthService.java")
    names = [c["name"] for c in chunks]
    assert "loginUser" in names


# ─────────────────────────────────────────────────────────────────────────────
#  10. Generic chunker — JavaScript function extraction
# ─────────────────────────────────────────────────────────────────────────────

def test_javascript_chunker_extracts_function():
    code = (
        "function loginUser(username, password) {\n"
        "    return verifyPassword(password);\n"
        "}\n"
    )
    chunks = chunk_generic_code(code, "auth.js")
    names = [c["name"] for c in chunks]
    assert "loginUser" in names


# ─────────────────────────────────────────────────────────────────────────────
#  11. repo_loader — skips non-code files
# ─────────────────────────────────────────────────────────────────────────────

def test_loader_skips_non_code_files(tmp_path):
    from parser.repo_loader import load_repository

    (tmp_path / "main.py").write_text("def hello(): pass")
    (tmp_path / "readme.md").write_text("# README")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    docs = load_repository(str(tmp_path))
    assert len(docs) == 1
    assert docs[0]["file_name"] == "main.py"


# ─────────────────────────────────────────────────────────────────────────────
#  12. repo_loader — skips node_modules
# ─────────────────────────────────────────────────────────────────────────────

def test_loader_skips_node_modules(tmp_path):
    from parser.repo_loader import load_repository

    (tmp_path / "main.py").write_text("def hello(): pass")
    nm = tmp_path / "node_modules" / "lib"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("function x(){}")

    docs = load_repository(str(tmp_path))
    assert len(docs) == 1


# ─────────────────────────────────────────────────────────────────────────────
#  13. TypeScript chunker
# ─────────────────────────────────────────────────────────────────────────────

def test_typescript_chunker_extracts_function():
    code = "function loginUser(username: string): boolean { return true; }"
    chunks = chunk_generic_code(code, "auth.ts")
    names = [c["name"] for c in chunks]
    assert "loginUser" in names


# ─────────────────────────────────────────────────────────────────────────────
#  14. Reranker — returns correct count
# ─────────────────────────────────────────────────────────────────────────────

def test_reranker_returns_top_k():
    from retrieval.reranker import rerank

    chunks = [
        {"name": f"func{i}", "file": "app.py", "code": f"def func{i}(): pass"}
        for i in range(10)
    ]
    result = rerank("authentication login", chunks, top_k=3)
    assert len(result) == 3


# ─────────────────────────────────────────────────────────────────────────────
#  15. Query cache — hit and miss
# ─────────────────────────────────────────────────────────────────────────────

def test_query_cache_hit_and_miss(tmp_path):
    from cache.query_cache import QueryCache

    cache = QueryCache(path=str(tmp_path / "cache.json"))

    # Miss
    assert cache.get("what is login") is None

    # Store and hit
    fake = [{"name": "login", "file": "auth.py"}]
    cache.set("what is login", fake)
    assert cache.get("what is login") == fake

    # Case-insensitive hit
    assert cache.get("WHAT IS LOGIN") == fake
