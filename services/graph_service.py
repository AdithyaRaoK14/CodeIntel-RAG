"""
services/graph_service.py
──────────────────────────
Call graph rendering + impact tree analysis extracted from api.py.
"""

from __future__ import annotations

import base64
import io
import logging

from parser.dependency_analyzer import get_full_dependency_tree
from visualization.call_graph import (
    build_call_graph, draw_call_graph, draw_call_graph_interactive,
)

log = logging.getLogger(__name__)


def get_graph_b64(query: str, results: list[dict], state) -> str | None:
    """Render the call graph as a base64-encoded PNG string."""
    try:
        import matplotlib.pyplot as plt
        graph  = build_call_graph(state.chunks, query, results)
        figure = draw_call_graph(graph)
        buf    = io.BytesIO()
        figure.savefig(buf, format="png", bbox_inches="tight",
                       facecolor="#1E1E2E", dpi=120)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        plt.close(figure)
        return b64
    except Exception as exc:
        log.warning("Could not render call graph: %s", exc)
        return None


def get_interactive_graph_html(query: str, results: list[dict], state) -> str | None:
    """Render the call graph as an interactive Pyvis HTML string."""
    graph = build_call_graph(state.chunks, query, results)
    return draw_call_graph_interactive(graph)


def get_impact_tree(function_name: str, state, max_depth: int = 3) -> dict:
    """Return the multi-hop impact tree for *function_name*."""
    return get_full_dependency_tree(state.chunks, function_name, max_depth=max_depth)
