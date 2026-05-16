"""
app.py  –  Codebase Intelligence Platform
──────────────────────────────────────────
Run:  streamlit run app.py
"""

import os

import streamlit as st

from parser.repo_loader import load_repository
from parser.code_chunker import chunk_python_code, chunk_generic_code
from parser.dependency_analyzer import find_function_usage
from embeddings.vector_store import store_chunks, clear_collection
from retrieval.hybrid_retriever import hybrid_search
from generator.response_generator import generate_answer, ollama_available
from visualization.call_graph import build_call_graph, draw_call_graph


# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Codebase Intelligence",
    page_icon="🧠",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _language_label(file_name: str) -> str:
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


@st.cache_resource(show_spinner="Parsing and embedding repository…")
def load_and_embed(repo_path: str):
    """
    Load, parse, and embed the repo ONCE per unique path.
    st.cache_resource keeps this in memory across Streamlit reruns,
    so we never re-embed on every widget interaction.
    """
    docs = load_repository(repo_path)
    all_chunks = []

    for doc in docs:
        if doc["file_name"].endswith(".py"):
            chunks = chunk_python_code(doc["content"], doc["file_name"])
        else:
            chunks = chunk_generic_code(doc["content"], doc["file_name"])
        all_chunks.extend(chunks)

    # Clear old vectors and store fresh ones
    clear_collection()
    store_chunks(all_chunks)

    return docs, all_chunks


# ─────────────────────────────────────────────────────────────────────────────
#  Session state defaults
# ─────────────────────────────────────────────────────────────────────────────

if "query" not in st.session_state:
    st.session_state["query"] = ""

if "results" not in st.session_state:
    st.session_state["results"] = None

if "answer" not in st.session_state:
    st.session_state["answer"] = None


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 Codebase RAG")
    st.markdown("---")

    repo_path = st.text_input(
        "Repository path",
        value="sample_repo",
        help="Paste an absolute or relative path to any local code repository.",
    )

    # Validate before loading
    if not os.path.exists(repo_path):
        st.error("Path not found – please enter a valid directory.")
        st.stop()

    st.markdown("---")
    st.markdown("**Quick queries**")

    quick_queries = [
        ("🔐 Authentication",   "Where is authentication?"),
        ("🔑 Login usage",      "Where is login used?"),
        ("🗄️  Database",         "Where is database connection?"),
        ("🌐 API endpoints",    "Where are API endpoints defined?"),
    ]
    for label, q in quick_queries:
        if st.button(label, use_container_width=True):
            st.session_state["query"] = q
            st.session_state["results"] = None
            st.session_state["answer"] = None
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear search", use_container_width=True):
        st.session_state["query"] = ""
        st.session_state["results"] = None
        st.session_state["answer"] = None
        st.rerun()

    st.markdown("---")
    # Ollama status badge
    if ollama_available():
        st.success("🤖 Ollama connected – AI answers enabled")
    else:
        st.warning(
            "⚠️ Ollama not running\n\n"
            "Install [Ollama](https://ollama.com) and run "
            "`ollama pull llama3` for AI-generated answers."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Load repository (cached)
# ─────────────────────────────────────────────────────────────────────────────

docs, all_chunks = load_and_embed(repo_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Header + metrics
# ─────────────────────────────────────────────────────────────────────────────

st.title("🧠 Codebase Intelligence Platform")
st.caption("Multi-language software dependency analysis using Hybrid RAG")

col1, col2, col3, col4 = st.columns(4)

languages = {_language_label(d["file_name"]) for d in docs} - {"Unknown"}

col1.metric("📁 Files Loaded",    len(docs))
col2.metric("⚙️ Functions Found",  len(all_chunks))
col3.metric("🌐 Languages",        len(languages))
col4.metric(
    "🔍 Retrieval Signals",
    "5",
    help="Semantic · Keyword · Structural · Metadata · Name-Token",
)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
#  Query input + analyse button
# ─────────────────────────────────────────────────────────────────────────────

query = st.text_input(
    "Ask about your codebase",
    value=st.session_state.get("query", ""),
    placeholder='e.g. "Where is authentication?" or "What calls connectDB?"',
)

if st.button("🔍 Analyse Code", type="primary") and query.strip():
    st.session_state["query"] = query

    with st.spinner("Running hybrid retrieval…"):
        results = hybrid_search(query, all_chunks)
        answer = generate_answer(query, results)

    st.session_state["results"] = results
    st.session_state["answer"] = answer


# ─────────────────────────────────────────────────────────────────────────────
#  Results
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state["results"] is not None:
    results = st.session_state["results"]
    answer = st.session_state["answer"]

    # ── Answer box ────────────────────────────────────────────────────────────
    st.markdown("## 💬 Analysis Result")
    st.success(answer)

    # ── Call graph ────────────────────────────────────────────────────────────
    st.markdown("### 📊 Function Call Graph")
    graph = build_call_graph(all_chunks, st.session_state["query"], results)
    figure = draw_call_graph(graph)
    st.pyplot(figure)

    # ── Dependency impact ─────────────────────────────────────────────────────
    if results:
        top_fn = results[0].get("name", "")
        usage = find_function_usage(all_chunks, top_fn)
        if usage:
            st.markdown(f"### 🔗 Who calls `{top_fn}()`?")
            for item in usage:
                st.info(f"`{item['function']}()` in `{item['file']}`")

    st.markdown("---")

    # ── Top results table ─────────────────────────────────────────────────────
    st.markdown("### 📋 Top Retrieved Functions")

    tabs = st.tabs([
        f"`{r['name']}`" for r in results[:6]
    ] or ["No results"])

    for tab, chunk in zip(tabs, results[:6]):
        with tab:
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**File:** `{chunk['file']}`")
            c2.markdown(f"**Line:** {chunk.get('line', '?')}")
            c3.markdown(f"**Language:** {_language_label(chunk['file'])}")

            calls = chunk.get("calls", [])
            if isinstance(calls, str):
                import json
                try:
                    calls = json.loads(calls)
                except Exception:
                    calls = [c.strip() for c in calls.split(",") if c.strip()]

            if calls:
                st.markdown(
                    "**Calls:** " + " · ".join(f"`{c}`" for c in calls)
                )
            else:
                st.markdown("**Calls:** _none_")

            code = chunk.get("code", "")
            if code:
                lang = _language_label(chunk["file"]).lower()
                if lang == "c++":
                    lang = "cpp"
                st.code(code, language=lang if lang != "unknown" else "text")

    st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
#  Full codebase inspector  (collapsed by default)
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("📂 Full Codebase Inspector", expanded=False):
    lang_filter = st.selectbox(
        "Filter by language",
        ["All"] + sorted(languages),
    )

    displayed = [
        c for c in all_chunks
        if lang_filter == "All" or _language_label(c["file"]) == lang_filter
    ]

    st.caption(f"Showing {len(displayed)} of {len(all_chunks)} functions")

    for chunk in displayed:
        with st.container():
            col_a, col_b = st.columns([3, 1])
            col_a.markdown(f"**`{chunk['name']}()`** — `{chunk['file']}`")
            col_b.markdown(
                f"<span style='color:grey;font-size:12px'>"
                f"{_language_label(chunk['file'])} · line {chunk.get('line', '?')}"
                f"</span>",
                unsafe_allow_html=True,
            )

            calls = chunk.get("calls", [])
            if calls:
                st.caption("Calls: " + ", ".join(calls))

            if chunk.get("code"):
                lang = _language_label(chunk["file"]).lower()
                st.code(chunk["code"], language=lang if lang !=
                        "unknown" else "text")

            st.markdown("---")
