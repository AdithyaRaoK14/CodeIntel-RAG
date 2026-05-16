"""
parser/code_chunker.py
───────────────────────
Parse source files into function-level chunks.

Changes
───────
• tree-sitter used for Java, C++, JavaScript, TypeScript
  (more accurate than regex — handles templates, lambdas, nested classes)
• Regex parsers kept as fallback if tree-sitter not installed
• Python still uses ast module (most accurate for Python)
• Replaced print() with proper logging
"""

import ast
import logging
import re

log = logging.getLogger(__name__)

# ── tree-sitter optional import ───────────────────────────────────────────────
try:
    from tree_sitter_languages import get_language, get_parser as _ts_get_parser
    _TS_AVAILABLE = True
    log.info("tree-sitter available — using AST-based parsing for Java/C++/JS/TS")
except ImportError:
    _TS_AVAILABLE = False
    log.warning(
        "tree-sitter-languages not installed — falling back to regex parsing. "
        "Install with: pip install tree-sitter tree-sitter-languages"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

CALL_KEYWORDS = {
    "if", "for", "while", "switch", "catch",
    "function", "return", "new", "typeof",
    "instanceof", "class", "import", "print",
    "len", "range", "input", "super", "this",
    "int", "float", "str", "list", "dict",
    "set", "bool", "void", "null", "true",
    "false", "try", "throw", "finally",
}


def _extract_calls_from_body(body: str) -> list[str]:
    pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    matches = re.findall(pattern, body)
    return [
        m for m in dict.fromkeys(matches)
        if m not in CALL_KEYWORDS
    ]


def _extract_brace_body(code: str, start_pos: int) -> str:
    brace_count = 0
    body_start = None
    i = start_pos

    while i < len(code):
        ch = code[i]
        if ch == "{":
            if body_start is None:
                body_start = i
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0 and body_start is not None:
                return code[body_start: i + 1]
        i += 1
    return ""


def _real_line(code: str, match_start: int) -> int:
    return code[:match_start].count("\n") + 1


# ─────────────────────────────────────────────────────────────────────────────
#  tree-sitter helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts_node_text(node, code_bytes: bytes) -> str:
    return code_bytes[node.start_byte: node.end_byte].decode("utf-8", errors="ignore")


def _ts_line(node, code_bytes: bytes) -> int:
    return code_bytes[: node.start_byte].count(b"\n") + 1


def _ts_extract_calls(node, code_bytes: bytes) -> list[str]:
    """Recursively collect function call names from a tree-sitter node."""
    calls = []
    if node.type == "call_expression":
        fn_node = node.child_by_field_name("function")
        if fn_node:
            if fn_node.type == "identifier":
                calls.append(_ts_node_text(fn_node, code_bytes))
            elif fn_node.type in ("member_expression", "field_expression"):
                prop = fn_node.child_by_field_name("property")
                if prop:
                    calls.append(_ts_node_text(prop, code_bytes))
    for child in node.children:
        calls.extend(_ts_extract_calls(child, code_bytes))
    return calls


def _ts_dedup_calls(calls: list[str], func_name: str) -> list[str]:
    return [
        c for c in dict.fromkeys(calls)
        if c not in CALL_KEYWORDS and c != func_name
    ]


def _ts_walk(node, *type_names):
    """Yield all descendant nodes whose type is in type_names."""
    if node.type in type_names:
        yield node
    for child in node.children:
        yield from _ts_walk(child, *type_names)


# ─────────────────────────────────────────────────────────────────────────────
#  Python Parser  (ast — unchanged, most accurate)
# ─────────────────────────────────────────────────────────────────────────────

def chunk_python_code(code: str, file_name: str) -> list[dict]:
    chunks = []
    try:
        tree = ast.parse(code)
        lines = code.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue

            start = node.lineno - 1
            end = node.end_lineno
            func_code = "\n".join(lines[start:end])

            calls = []
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        if child.func.id not in CALL_KEYWORDS:
                            calls.append(child.func.id)
                    elif isinstance(child.func, ast.Attribute):
                        calls.append(child.func.attr)

            chunks.append({
                "type":  "function",
                "name":  node.name,
                "file":  file_name,
                "line":  node.lineno,
                "code":  func_code,
                "calls": list(dict.fromkeys(calls)),
            })

    except SyntaxError as exc:
        log.error("Python parse error in %s: %s", file_name, exc)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  Java — tree-sitter
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_java_ts(code: str, file_name: str) -> list[dict]:
    parser = _ts_get_parser("java")
    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    chunks = []
    seen = set()  # now stores (class_name, func_name) tuples

    for node in _ts_walk(tree.root_node,
                         "method_declaration",
                         "constructor_declaration"):
        name_node = node.child_by_field_name("name")
        if name_node is None:
            continue

        func_name = _ts_node_text(name_node, code_bytes)

        # Walk up to find the enclosing class/interface name
        class_name = ""
        parent = node.parent
        while parent is not None:
            if parent.type in ("class_declaration", "interface_declaration"):
                cn = parent.child_by_field_name("name")
                if cn:
                    class_name = _ts_node_text(cn, code_bytes)
                break
            parent = parent.parent

        if (class_name, func_name) in seen:
            continue
        seen.add((class_name, func_name))

        func_code = _ts_node_text(node, code_bytes)
        func_line = _ts_line(node, code_bytes)
        calls = _ts_dedup_calls(_ts_extract_calls(node, code_bytes), func_name)

        chunks.append({
            "type":  "function",
            "name":  func_name,
            "file":  file_name,
            "line":  func_line,
            "code":  func_code,
            "calls": calls,
        })

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  C++ — tree-sitter
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_cpp_ts(code: str, file_name: str) -> list[dict]:
    parser = _ts_get_parser("cpp")
    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    chunks = []
    seen = set()

    for node in _ts_walk(tree.root_node, "function_definition"):
        # Navigate: function_definition → declarator → function_declarator → declarator
        decl = node.child_by_field_name("declarator")
        if decl is None:
            continue

        # Unwrap pointer/reference declarators
        while decl and decl.type in ("pointer_declarator", "reference_declarator"):
            decl = decl.child_by_field_name("declarator")

        if decl is None or decl.type != "function_declarator":
            continue

        name_node = decl.child_by_field_name("declarator")
        if name_node is None:
            continue

        # Handle qualified names like MyClass::myMethod
        if name_node.type == "qualified_identifier":
            name_node = name_node.child_by_field_name("name") or name_node

        func_name = _ts_node_text(name_node, code_bytes).split("::")[-1]

        # Derive qualified class from a qualified_identifier ancestor
        qualified_class = ""
        parent = node.parent
        while parent is not None:
            if parent.type in ("class_specifier",):
                cn = parent.child_by_field_name("name")
                if cn:
                    qualified_class = _ts_node_text(cn, code_bytes)
                break
            # Also capture Foo:: prefix from the declarator text
            raw_name = _ts_node_text(name_node, code_bytes)
            if "::" in raw_name:
                qualified_class = "::".join(raw_name.split("::")[:-1])
            parent = parent.parent

        if not func_name or (qualified_class, func_name) in seen:
            continue
        seen.add((qualified_class, func_name))

        func_code = _ts_node_text(node, code_bytes)
        func_line = _ts_line(node, code_bytes)
        calls = _ts_dedup_calls(_ts_extract_calls(node, code_bytes), func_name)

        chunks.append({
            "type":  "function",
            "name":  func_name,
            "file":  file_name,
            "line":  func_line,
            "code":  func_code,
            "calls": calls,
        })

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  JavaScript — tree-sitter
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_js_ts(code: str, file_name: str, lang: str = "javascript") -> list[dict]:
    parser = _ts_get_parser(lang)
    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    chunks = []
    seen = set()

    # Named function declarations:  function foo() {}
    for node in _ts_walk(tree.root_node, "function_declaration"):
        name_node = node.child_by_field_name("name")
        if name_node is None:
            continue
        func_name = _ts_node_text(name_node, code_bytes)
        if func_name in seen or func_name in CALL_KEYWORDS:
            continue
        seen.add(func_name)

        func_code = _ts_node_text(node, code_bytes)
        func_line = _ts_line(node, code_bytes)
        calls = _ts_dedup_calls(_ts_extract_calls(node, code_bytes), func_name)
        chunks.append({"type": "function", "name": func_name, "file": file_name,
                       "line": func_line, "code": func_code, "calls": calls})

    # Arrow / function expressions:  const foo = () => {}
    for node in _ts_walk(tree.root_node, "variable_declarator"):
        name_node = node.child_by_field_name("name")
        value_node = node.child_by_field_name("value")
        if name_node is None or value_node is None:
            continue
        if value_node.type not in ("arrow_function", "function"):
            continue

        func_name = _ts_node_text(name_node, code_bytes)
        if func_name in seen or func_name in CALL_KEYWORDS:
            continue
        seen.add(func_name)

        func_code = _ts_node_text(node.parent or node, code_bytes)
        func_line = _ts_line(node, code_bytes)
        calls = _ts_dedup_calls(_ts_extract_calls(
            value_node, code_bytes), func_name)
        chunks.append({"type": "function", "name": func_name, "file": file_name,
                       "line": func_line, "code": func_code, "calls": calls})

    # Class methods:  methodName() {}
    for node in _ts_walk(tree.root_node, "method_definition"):
        name_node = node.child_by_field_name("name")
        if name_node is None:
            continue
        func_name = _ts_node_text(name_node, code_bytes)
        if func_name in seen or func_name in CALL_KEYWORDS:
            continue
        seen.add(func_name)

        func_code = _ts_node_text(node, code_bytes)
        func_line = _ts_line(node, code_bytes)
        calls = _ts_dedup_calls(_ts_extract_calls(node, code_bytes), func_name)
        chunks.append({"type": "function", "name": func_name, "file": file_name,
                       "line": func_line, "code": func_code, "calls": calls})

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  Regex fallbacks (used when tree-sitter not installed)
# ─────────────────────────────────────────────────────────────────────────────

_JAVA_CPP_PATTERN = re.compile(
    r'(?:(?:public|private|protected|static|final|override|virtual|inline)\s+)*'
    r'(?:void|int|long|float|double|bool|boolean|string|String|auto|[A-Z][a-zA-Z0-9_<>]*)\s+'
    r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
    re.MULTILINE,
)

_JS_PATTERNS = [
    re.compile(r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', re.MULTILINE),
    re.compile(
        r'(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*'
        r'(?:async\s+)?(?:function\s*\(|\([^)]*\)\s*=>)',
        re.MULTILINE,
    ),
    re.compile(r'^\s*(?:async\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{',
               re.MULTILINE),
]

_TS_PATTERNS = _JS_PATTERNS + [
    re.compile(
        r'(?:(?:public|private|protected|static|async|override)\s+)*'
        r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(?::\s*[\w<>\[\]|,\s]+\s*)?\{',
        re.MULTILINE,
    ),
]


def _regex_chunk(code: str, file_name: str, patterns: list) -> list[dict]:
    seen = set()
    chunks = []
    for pattern in patterns:
        for match in pattern.finditer(code):
            func_name = match.group(1)
            if func_name in seen or func_name in CALL_KEYWORDS:
                continue
            seen.add(func_name)
            line_no = _real_line(code, match.start())
            func_body = _extract_brace_body(code, match.end())
            calls = _extract_calls_from_body(func_body) if func_body else []
            calls = [c for c in calls if c != func_name]
            full_code = code[match.start(): match.start() +
                             len(match.group(0))] + func_body
            chunks.append({"type": "function", "name": func_name, "file": file_name,
                           "line": line_no, "code": full_code, "calls": calls})
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  Router
# ─────────────────────────────────────────────────────────────────────────────

def chunk_generic_code(code: str, file_name: str) -> list[dict]:
    """Route to the right parser based on file extension."""

    if file_name.endswith(".java"):
        if _TS_AVAILABLE:
            try:
                return _chunk_java_ts(code, file_name)
            except Exception as exc:
                log.warning(
                    "tree-sitter Java failed for %s: %s — using regex", file_name, exc)
        return _regex_chunk(code, file_name, [_JAVA_CPP_PATTERN])

    if file_name.endswith(".cpp"):
        if _TS_AVAILABLE:
            try:
                return _chunk_cpp_ts(code, file_name)
            except Exception as exc:
                log.warning(
                    "tree-sitter C++ failed for %s: %s — using regex", file_name, exc)
        return _regex_chunk(code, file_name, [_JAVA_CPP_PATTERN])

    if file_name.endswith(".js"):
        if _TS_AVAILABLE:
            try:
                return _chunk_js_ts(code, file_name, lang="javascript")
            except Exception as exc:
                log.warning(
                    "tree-sitter JS failed for %s: %s — using regex", file_name, exc)
        return _regex_chunk(code, file_name, _JS_PATTERNS)

    if file_name.endswith(".ts"):
        if _TS_AVAILABLE:
            try:
                return _chunk_js_ts(code, file_name, lang="typescript")
            except Exception as exc:
                log.warning(
                    "tree-sitter TS failed for %s: %s — using regex", file_name, exc)
        return _regex_chunk(code, file_name, _TS_PATTERNS)

    # Generic fallback
    return _regex_chunk(code, file_name, [_JAVA_CPP_PATTERN])
