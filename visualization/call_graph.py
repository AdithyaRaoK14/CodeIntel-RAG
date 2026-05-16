"""
visualization/call_graph.py
────────────────────────────
Build and render a directed call graph for any function in the codebase.

Changes
───────
• Added draw_call_graph_interactive() — returns an HTML string using Pyvis.
  Interactive: users can drag nodes, zoom, hover for details.
• Static PNG (draw_call_graph) kept for the /export report.
• Replaced print() with proper logging.
"""

import json
import logging
import re
import textwrap

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

matplotlib.use("Agg")

log = logging.getLogger(__name__)

# Optional Pyvis import
try:
    from pyvis.network import Network
    _PYVIS_AVAILABLE = True
except ImportError:
    _PYVIS_AVAILABLE = False
    log.warning("pyvis not installed — interactive graphs disabled. "
                "Install with: pip install pyvis")


# ─────────────────────────────────────────────────────────────────────────────
#  Query helpers
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "where", "what", "show", "find", "tell", "about", "the", "is", "are",
    "does", "used", "called", "function", "method", "code", "how", "which",
    "file", "in", "of", "for", "to", "from", "all", "any",
}


def _extract_target(query: str) -> str | None:
    patterns = [
        r'where\s+is\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'what\s+calls\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\)',
        r'([a-zA-Z_][a-zA-Z0-9_]*)\s+function',
    ]
    for pat in patterns:
        m = re.search(pat, query, re.IGNORECASE)
        if m and m.group(1).lower() not in _STOPWORDS:
            return m.group(1)

    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]+\b', query)
    for w in words:
        if w.lower() not in _STOPWORDS:
            return w
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Graph builder  (shared by both renderers)
# ─────────────────────────────────────────────────────────────────────────────

def _node_label(name: str, file: str) -> str:
    short_file = file.split("/")[-1].split("\\")[-1]
    return f"{name}\n({short_file})"


def build_call_graph(
    chunks: list[dict],
    query: str,
    results: list[dict] | None = None,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    target_fn = _extract_target(query)

    if target_fn:
        exists = any(c["name"].lower() == target_fn.lower() for c in chunks)
        if not exists:
            target_fn = None

    if not target_fn and results:
        target_fn = results[0].get("name")

    if not target_fn:
        return graph

    target_fn_lower = target_fn.lower()
    target_nodes = [c for c in chunks if c["name"].lower() == target_fn_lower]

    if not target_nodes:
        return graph

    tc = target_nodes[0]
    t_label = _node_label(tc["name"], tc["file"])
    graph.add_node(t_label, role="target")

    # Callers → target
    for chunk in chunks:
        calls = chunk.get("calls", [])
        if isinstance(calls, str):
            try:
                calls = json.loads(calls)
            except Exception:
                calls = [c.strip() for c in calls.split(",") if c.strip()]

        if any(c.lower() == target_fn_lower for c in calls):
            src = _node_label(chunk["name"], chunk["file"])
            graph.add_node(src, role="caller")
            graph.add_edge(src, t_label)

    # target → callees
    target_calls = tc.get("calls", [])
    if isinstance(target_calls, str):
        try:
            target_calls = json.loads(target_calls)
        except Exception:
            target_calls = [c.strip()
                            for c in target_calls.split(",") if c.strip()]

    for callee_name in target_calls:
        callee_chunks = [c for c in chunks if c["name"].lower()
                         == callee_name.lower()]
        if callee_chunks:
            callee_label = _node_label(
                callee_chunks[0]["name"], callee_chunks[0]["file"])
        else:
            callee_label = f"{callee_name}\n(external)"
        graph.add_node(callee_label, role="callee")
        graph.add_edge(t_label, callee_label)

    return graph


# ─────────────────────────────────────────────────────────────────────────────
#  Static PNG renderer  (used by /export report)
# ─────────────────────────────────────────────────────────────────────────────

_ROLE_COLORS = {
    "target": "#FF8C00",
    "caller": "#4C9BE8",
    "callee": "#5CB85C",
}
_DEFAULT_COLOR = "#AAAAAA"


def draw_call_graph(graph: nx.DiGraph) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_facecolor("#1E1E2E")
    fig.patch.set_facecolor("#1E1E2E")

    if len(graph.nodes()) == 0:
        ax.text(0.5, 0.5,
                "No call graph available for this query.\n"
                "Try: \"Where is login used?\"",
                ha="center", va="center", fontsize=11, color="white",
                transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    if len(graph.nodes()) == 1:
        pos = {list(graph.nodes())[0]: (0, 0)}
    elif len(graph.nodes()) <= 3:
        pos = nx.shell_layout(graph)
    else:
        try:
            pos = nx.planar_layout(graph)
        except nx.NetworkXException:
            pos = nx.spring_layout(graph, seed=42, k=1.5)

    node_colors = [
        _ROLE_COLORS.get(graph.nodes[n].get("role"), _DEFAULT_COLOR)
        for n in graph.nodes()
    ]

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors,
                           node_size=1800, alpha=0.95)
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#CCCCCC",
                           arrows=True, arrowstyle="->", arrowsize=18,
                           width=1.5, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8,
                            font_color="white", font_weight="bold")

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor=_ROLE_COLORS["target"], label="Searched function"),
            Patch(facecolor=_ROLE_COLORS["caller"], label="Caller"),
            Patch(facecolor=_ROLE_COLORS["callee"], label="Callee"),
        ],
        loc="upper left", fontsize=8,
        facecolor="#2E2E3E", labelcolor="white",
    )

    ax.set_axis_off()
    plt.margins(0.25)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Interactive HTML renderer  (Pyvis)
# ─────────────────────────────────────────────────────────────────────────────

def draw_call_graph_interactive(graph: nx.DiGraph) -> str | None:
    """
    Render the call graph as an interactive HTML string using Pyvis.
    Users can drag nodes, zoom, and hover for details.

    Returns None if Pyvis is not installed.
    """
    if not _PYVIS_AVAILABLE:
        log.warning(
            "pyvis not available — returning None for interactive graph")
        return None

    if len(graph.nodes()) == 0:
        return None

    net = Network(
        height="420px",
        width="100%",
        bgcolor="#1E1E2E",
        font_color="white",
        directed=True,
        notebook=False,
        cdn_resources="remote",  # ← add this
    )

    role_colors = {
        "target": "#FF8C00",
        "caller": "#4C9BE8",
        "callee": "#5CB85C",
    }

    for node, data in graph.nodes(data=True):
        color = role_colors.get(data.get("role"), "#AAAAAA")
        # Clean label for display (replace \n with space)
        label = node.replace("\n", " ")
        net.add_node(
            node,
            label=label,
            color=color,
            size=28,
            font={"size": 11, "color": "white"},
            title=node,   # tooltip on hover
        )

    for src, dst in graph.edges():
        net.add_edge(src, dst, color="#CCCCCC", arrows="to", width=2)

    # Disable physics after initial layout for stability
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "stabilization": { "iterations": 100 },
        "barnesHut": { "gravitationalConstant": -3000, "springLength": 150 }
      },
      "interaction": { "hover": true, "tooltipDelay": 100 }
    }
    """)

    return net.generate_html()
