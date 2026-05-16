"""
retrieval/reranker.py
──────────────────────
Blended hybrid + cross-encoder reranker.

Model choice
────────────
cross-encoder/ms-marco-MiniLM-L-6-v2 is kept over BAAI/bge-reranker-base.
Benchmarking showed bge-reranker-base hurt Hit@5 (0.88 vs 0.96) because it is
a general-purpose reranker not trained on code-like queries. ms-marco-MiniLM
was trained on MS MARCO which contains mixed text including code/technical
content, making it a better fit for function-name + code-snippet pairs.

Weight tuning
─────────────
CE_W=0.45, HYBRID_W=0.55 — balanced blend.
• CE_W=0.3  → Hit@5=0.96, Hit@1=0.72, MRR=0.808  (previous)
• CE_W=0.45 → target: lift Hit@3 without hurting Hit@5
• CE_W=0.7  → Hit@5=0.88 (regression) — CE dominates and its mistakes on
  short code snippets compound.

hybrid_score (not rank)
───────────────────────
hybrid_retriever.py attaches a normalised hybrid_score float to each result.
We use positional rank decay (1/(1+i)) as the hybrid prior rather than the
raw hybrid_score float — when adjacent candidates score similarly the
normalised values cluster near 1.0 and small CE deltas shuffle them
unpredictably, causing Hit@5 regression (0.96→0.80).
"""

import numpy as np
from sentence_transformers import CrossEncoder

_RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# FIX: raised from CE_W=0.3 to CE_W=0.45 now that hybrid ordering is clean.
# Earlier attempts at CE_W=0.45 hurt Hit@5 because generic names ("response",
# "view", "post") were landing at hybrid rank #1 and the CE was amplifying
# those errors. With the generic-name penalties and _is_test_artifact fix in
# place, hybrid rank #1 is now usually correct, so the CE can contribute more
# without overriding good ordering.
_CE_W = 0.45
_HYBRID_W = 0.55


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank chunks for query using a blended CE + hybrid score.

    Parameters
    ----------
    query  : raw user query string
    chunks : candidate list from hybrid_search() — each dict may carry a
             "hybrid_score" float in [0, 1] set by hybrid_retriever.py
    top_k  : number of results to return
    """
    if not chunks:
        return []

    n = len(chunks)
    pairs = [
        (query, f"{c['name']} in {c['file']}\n{c.get('code', '')[:500]}")
        for c in chunks
    ]

    # ── Cross-encoder scores ──────────────────────────────────────────────────
    ce_scores = np.array(_RERANKER.predict(pairs), dtype=np.float32)
    ce_min, ce_max = ce_scores.min(), ce_scores.max()
    if ce_max > ce_min:
        ce_scores = (ce_scores - ce_min) / (ce_max - ce_min)
    else:
        ce_scores = np.ones(n, dtype=np.float32)

    # ── Hybrid scores — positional rank decay (stable, not hybrid_score) ─────
    # Normalised hybrid_score was removed: when adjacent candidates score
    # similarly the normalised values cluster near 1.0 and small CE deltas
    # shuffle them unpredictably, causing Hit@5 regression (0.96→0.80).
    # 1/(1+i) decay gives a smooth prior that respects hybrid ordering without
    # amplifying noise from score normalisation.
    rank_scores = np.array([1.0 / (1.0 + i)
                           for i in range(n)], dtype=np.float32)
    rank_scores = rank_scores / rank_scores.max()

    # ── Blend ─────────────────────────────────────────────────────────────────
    final_scores = _CE_W * ce_scores + _HYBRID_W * rank_scores

    ranked = sorted(
        zip(final_scores.tolist(), chunks),
        key=lambda x: x[0],
        reverse=True,
    )
    return [chunk for _, chunk in ranked[:top_k]]
