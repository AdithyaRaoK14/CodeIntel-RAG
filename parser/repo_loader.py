"""
parser/repo_loader.py
──────────────────────
Loads source files from a repository directory.

Changes
───────
• Skips .git, __pycache__, node_modules, venv, build folders
• NEW: skip_tests parameter — when True (default), skips test directories
  entirely. This is important for retrieval quality: test files contain
  function names that mirror source functions (test_dispatch_request,
  test_handle_exception) and inflate BM25 scores for source-level queries.
  Set skip_tests=False when you specifically want to search test code.
• Handles non-UTF-8 files without crashing (errors="ignore")
• Skips files larger than 1 MB
• Skips empty files
• Handles OSError / PermissionError gracefully
• Added .ts (TypeScript) support
• Replaced print() with proper logging
"""

import logging
import os

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [".py", ".java", ".js", ".cpp", ".ts"]

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "env", "dist", "build", ".idea", ".vscode", "target",
    "bin", "obj", ".gradle", ".mypy_cache", ".pytest_cache",
    "htmlcov", ".tox",
}

# Directories that contain test code. Skipped by default since test function
# names pollute retrieval results (test_dispatch_request beats dispatch_request).
TEST_DIRS = {"tests", "test", "testing", "spec", "__tests__"}

# Test file name prefixes/suffixes — individual files skipped even outside TEST_DIRS
_TEST_FILE_PREFIXES = ("test_", "tests_")
_TEST_FILE_NAMES = {"conftest.py"}

MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB


def _is_test_file(file_name: str) -> bool:
    lower = file_name.lower()
    return (
        any(lower.startswith(p) for p in _TEST_FILE_PREFIXES)
        or lower in _TEST_FILE_NAMES
        or lower.endswith("_test.py")
        or lower.endswith("_test.js")
        or lower.endswith("_test.ts")
        or lower.endswith(".test.js")
        or lower.endswith(".test.ts")
        or lower.endswith(".spec.js")
        or lower.endswith(".spec.ts")
    )


def load_repository(repo_path: str, skip_tests: bool = True) -> list[dict]:
    """
    Walk repo_path and return a list of file dicts.
    Each dict has: file_name, file_path, content.

    Parameters
    ----------
    repo_path  : root directory of the repository
    skip_tests : if True (default), skip test directories and test files.
                 Set False to include test code in the index.
    """
    if not os.path.isdir(repo_path):
        raise ValueError(f"Repository path not found: {repo_path}")

    skip_dir_set = SKIP_DIRS | (TEST_DIRS if skip_tests else set())
    documents = []
    skipped_test = 0

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip_dir_set]

        for file in files:
            if not any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                continue

            # Skip individual test files even if not in a test directory
            if skip_tests and _is_test_file(file):
                skipped_test += 1
                continue

            file_path = os.path.join(root, file)

            try:
                if os.path.getsize(file_path) > MAX_FILE_SIZE_BYTES:
                    log.warning("Skipping large file: %s", file_path)
                    continue
            except OSError:
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except (OSError, PermissionError) as exc:
                log.error("Cannot read %s: %s", file_path, exc)
                continue

            if not content.strip():
                continue

            documents.append({
                "file_name": file,
                "file_path": file_path,
                "content":   content,
            })

    if skipped_test:
        log.info("Skipped %d test files (pass skip_tests=False to include them)",
                 skipped_test)
    log.info("Loaded %d files from %s", len(documents), repo_path)
    return documents
