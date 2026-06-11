"""Clause-aware hybrid retrieval (dense + lexical + metadata boost)."""
from __future__ import annotations

import re
from typing import Any

from clause_parse import is_article_clause_number

# 307절, 401 절, 제401절, "401" in technical questions
CLAUSE_IN_QUERY_RE = re.compile(
    r"(?:제?\s*)?(\d{3,4})\s*절|(\d{3,4})절|(?:^|\s)(\d{3,4})(?:\s*절|\s|$)",
    re.IGNORECASE,
)

TOKEN_RE = re.compile(r"[\w가-힣]+", re.UNICODE)

# Lower distance = better (Chroma cosine distance).
CLAUSE_EXACT_BOOST = 0.22
CLAUSE_IN_TEXT_BOOST = 0.08
LEXICAL_BOOST_SCALE = 0.12
REFERENCE_LINE_BOOST = 0.06


def extract_clause_hints(query: str) -> list[str]:
    hints: list[str] = []
    for groups in CLAUSE_IN_QUERY_RE.finditer(query):
        for g in groups.groups():
            if g and g.isdigit() and len(g) >= 3:
                hints.append(g)
    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for h in hints:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def _meta_clause(meta: dict) -> str:
    return str(meta.get("clause_number") or meta.get("article_number") or "").strip()


def lexical_overlap(query: str, document: str) -> float:
    q_tokens = {t for t in TOKEN_RE.findall(query.lower()) if len(t) > 1}
    if not q_tokens:
        return 0.0
    d_tokens = {t for t in TOKEN_RE.findall(document.lower()) if len(t) > 1}
    return len(q_tokens & d_tokens) / len(q_tokens)


def adjusted_distance(
    distance: float,
    *,
    query: str,
    document: str,
    meta: dict,
    clause_hints: list[str],
) -> float:
    score = float(distance)
    meta_clause = _meta_clause(meta)
    doc_head = (document or "")[:400]

    for hint in clause_hints:
        if meta_clause == hint:
            score -= CLAUSE_EXACT_BOOST
        elif hint in doc_head and is_article_clause_number(hint):
            score -= CLAUSE_IN_TEXT_BOOST
        # "401.의 규정을 준용" style cross-refs
        if f"{hint}." in doc_head or f"{hint}절" in doc_head:
            score -= REFERENCE_LINE_BOOST

    score -= LEXICAL_BOOST_SCALE * lexical_overlap(query, document)
    return score


def query_with_hybrid_ranking(
    collection,
    query: str,
    query_vector: list[float],
    *,
    top_k: int = 5,
    fetch_k: int | None = None,
) -> dict[str, Any]:
    """
    Over-fetch vector hits, rerank with clause + lexical boosts, return top_k.
    Optionally merge in metadata-filtered hits when clause hints exist.
    """
    clause_hints = extract_clause_hints(query)
    n_fetch = fetch_k or max(top_k * 6, 30)

    merged_ids: list[str] = []
    merged_dist: dict[str, float] = {}
    merged_meta: dict[str, dict] = {}
    merged_doc: dict[str, str] = {}

    def absorb(raw: dict) -> None:
        for cid, dist, meta, doc in zip(
            raw["ids"][0],
            raw["distances"][0],
            raw["metadatas"][0],
            raw["documents"][0],
        ):
            if cid in merged_dist:
                merged_dist[cid] = min(merged_dist[cid], float(dist))
            else:
                merged_ids.append(cid)
                merged_dist[cid] = float(dist)
                merged_meta[cid] = meta or {}
                merged_doc[cid] = doc or ""

    raw_vector = collection.query(query_embeddings=[query_vector], n_results=n_fetch)
    absorb(raw_vector)

    if clause_hints:
        for hint in clause_hints[:3]:
            try:
                filtered = collection.query(
                    query_embeddings=[query_vector],
                    n_results=min(15, n_fetch),
                    where={
                        "$or": [
                            {"clause_number": hint},
                            {"article_number": hint},
                        ]
                    },
                )
                absorb(filtered)
            except Exception:
                pass

    ranked = sorted(
        merged_ids,
        key=lambda cid: adjusted_distance(
            merged_dist[cid],
            query=query,
            document=merged_doc[cid],
            meta=merged_meta[cid],
            clause_hints=clause_hints,
        ),
    )[:top_k]

    return {
        "ids": [ranked],
        "distances": [[merged_dist[cid] for cid in ranked]],
        "metadatas": [[merged_meta[cid] for cid in ranked]],
        "documents": [[merged_doc[cid] for cid in ranked]],
        "clause_hints": clause_hints,
    }


def enrich_query_for_embedding(query: str, model_name: str) -> str:
    """Prepend clause hints for E5 query prefix (improves article-number queries)."""
    hints = extract_clause_hints(query)
    if not hints:
        return query
    prefix = " ".join(f"{h}절 {h}." for h in hints[:2])
    return f"{prefix} {query}".strip()
