"""Schema-aware 2-stage table retrieval with structured scoring and confidence gating."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from embedding_policy import embed_texts_local
from retrieval_search import _merge_where, enrich_query_for_embedding, safe_chroma_query
from table_normalize_lib import (
    best_entity_overlap,
    entity_matches,
    expand_entity_aliases,
    normalize_compact,
)
from table_query_parser import ParsedTableQuery, build_embed_query, parse_table_query
from table_rag_config import (
    CONFIDENCE_GATE_THRESHOLD,
    SCORING_BOOST_ROW_COL,
    SCORING_BOOST_ROW_COL_TOPIC,
    SCORING_CAPTION_AUX_MIN,
    SCORING_PENALTY_MISSING_COLUMN,
    SCORING_PENALTY_ROW_ONLY,
    SCORING_PENALTY_TOPIC_MISMATCH,
    SCORING_WEIGHTS,
    TABLE_SCHEMA_ROUTE_K,
    TABLE_SCHEMA_STAGE2_ROW_K,
)
from table_schema_lib import parse_schema_from_document

ROUTE_FETCH = 64
SUMMARY_FETCH = 48
SCHEMA_FETCH = 48
MARKDOWN_PER_TABLE = 1

CONFIDENCE_GATE_MESSAGE = (
    "관련 표 후보는 확인되었으나, 질문의 행/열 조건과 정확히 대응되는 셀을 확정하지 못했습니다. "
    "원문 표 확인이 필요합니다."
)

_WEIGHTS = SCORING_WEIGHTS


@dataclass
class TableScoreBreakdown:
    table_id: str
    vector_distance: float = 1.0
    caption_match: float = 0.0
    table_topic_match: float = 0.0
    column_match: float = 0.0
    row_entity_match: float = 0.0
    unit_match: float = 0.0
    keyword_match: float = 0.0
    combined_score: float = 0.0
    chunk_type: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _query_typed(
    collection,
    vector: list[float],
    *,
    chunk_types: list[str],
    n_results: int,
    doc_id: str | None = None,
    source: str | None = None,
    table_ids: list[str] | None = None,
) -> list[tuple[str, float, dict, str]]:
    if len(chunk_types) == 1:
        type_where: dict = {"chunk_type": chunk_types[0]}
    else:
        type_where = {"chunk_type": {"$in": chunk_types}}
    base_where = _merge_where(
        type_where,
        {"source": source.upper()} if source else None,
        {"doc_id": doc_id} if doc_id else None,
        {"table_id": {"$in": table_ids}} if table_ids else None,
    )
    raw = safe_chroma_query(
        collection,
        query_embeddings=[vector],
        n_results=n_results,
        where=base_where,
    )
    if not raw.get("ids") or not raw["ids"][0]:
        return []
    return list(
        zip(
            raw["ids"][0],
            raw["distances"][0],
            raw["metadatas"][0],
            raw["documents"][0],
        )
    )


def _score_caption(parsed: ParsedTableQuery, schema: dict, meta: dict, doc: str) -> float:
    caption = str(schema.get("caption") or meta.get("caption") or "")
    blob = f"{caption} {doc}"
    if not caption.strip():
        return 0.0
    hits = sum(1 for kw in parsed.keyword_terms[:8] if kw and kw in blob)
    return min(1.0, hits / max(1, min(4, len(parsed.keyword_terms[:8]))))


def _score_topics(parsed: ParsedTableQuery, schema: dict) -> float:
    topics = schema.get("table_topics") or []
    cap = str(schema.get("caption") or "")
    section = str(schema.get("section_title") or "")
    blob = f"{' '.join(topics)} {cap} {section}".lower()
    if not parsed.table_topic_candidates:
        return 0.0
    hits = 0
    for t in parsed.table_topic_candidates:
        tl = t.lower()
        if tl in blob or normalize_compact(t) in normalize_compact(blob):
            hits += 1
            continue
        for _key, aliases in (
            ("chemical_composition", ("화학", "chemical")),
            ("mechanical_property", ("기계", "mechanical", "항복", "인장")),
            ("inspection", ("정기검사", "inspection", "reporting")),
            ("test_material", ("시험재", "test_material", "용접")),
            ("note_lookup", ("비고", "note")),
        ):
            if tl == _key or any(a in tl for a in aliases):
                if _key.replace("_", "") in blob.replace("_", "") or any(a in blob for a in aliases):
                    hits += 1
                    break
    return min(1.0, hits / len(parsed.table_topic_candidates))


def _score_columns(parsed: ParsedTableQuery, schema: dict) -> float:
    if not parsed.column_entities:
        return 0.5  # neutral when no column constraint
    cols = schema.get("normalized_column_names") or schema.get("column_names") or []
    col_blob = " | ".join(str(c) for c in cols)
    raw_snippet = str(schema.get("_raw_snippet") or "")
    targets = [col_blob] + [str(c) for c in cols]
    if raw_snippet:
        targets.append(raw_snippet[:600])
    return best_entity_overlap(parsed.column_entities, targets)


def _caption_aux_score(parsed: ParsedTableQuery, caption_score: float, topic_score: float, column_score: float) -> float:
    """Caption alone should not dominate; require topic or column corroboration."""
    if caption_score <= 0:
        return 0.0
    corroboration = max(topic_score, column_score)
    if corroboration >= SCORING_CAPTION_AUX_MIN:
        return caption_score
    return caption_score * (corroboration / max(SCORING_CAPTION_AUX_MIN, 0.01)) * 0.35


def _apply_query_type_adjustments(
    parsed: ParsedTableQuery,
    bd: TableScoreBreakdown,
    schema: dict,
) -> float:
    """Boost/penalty rules for cell/column lookup ranking."""
    delta = 0.0
    topics = " ".join(schema.get("table_topics") or []).lower()
    q_topics = [t.lower() for t in parsed.table_topic_candidates]

    if parsed.query_type in ("cell_lookup", "column_lookup") and parsed.column_entities:
        if bd.column_match < 0.15:
            delta -= SCORING_PENALTY_MISSING_COLUMN
        if bd.row_entity_match >= 0.5 and bd.column_match < 0.15:
            delta -= SCORING_PENALTY_ROW_ONLY

    if parsed.row_entities and parsed.column_entities:
        if bd.row_entity_match >= 0.5 and bd.column_match >= 0.5:
            delta += SCORING_BOOST_ROW_COL
        if (
            bd.row_entity_match >= 0.5
            and bd.column_match >= 0.5
            and bd.table_topic_match >= 0.4
        ):
            delta += SCORING_BOOST_ROW_COL_TOPIC

    if q_topics and topics:
        chemistry_q = any("화학" in t or "chemical" in t for t in q_topics)
        chemistry_t = "chemical_composition" in topics or "화학성분" in topics
        lot_t = "lot_treatment" in topics or "열처리" in topics
        test_t = "test_material" in topics or "시험재" in topics
        mech_t = "mechanical_property" in topics or "기계적" in topics
        if chemistry_q and not chemistry_t:
            if lot_t or test_t:
                delta -= SCORING_PENALTY_TOPIC_MISMATCH
            elif mech_t and parsed.column_entities:
                delta -= SCORING_PENALTY_TOPIC_MISMATCH * 0.6
        if any("정기검사" in t or "inspection" in t for t in q_topics):
            if chemistry_t and "정기검사" not in topics:
                delta -= SCORING_PENALTY_TOPIC_MISMATCH * 0.5
    return delta


def _score_rows(parsed: ParsedTableQuery, schema: dict) -> float:
    if not parsed.row_entities:
        return 0.5
    rows = schema.get("normalized_row_entities") or schema.get("row_entities") or []
    return best_entity_overlap(parsed.row_entities, rows)


def _score_units(parsed: ParsedTableQuery, schema: dict) -> float:
    if not parsed.unit_candidates:
        return 0.5
    units = schema.get("units") or []
    if not units:
        return 0.0
    unit_blob = " ".join(units)
    hits = sum(1 for u in parsed.unit_candidates if u in unit_blob)
    return min(1.0, hits / len(parsed.unit_candidates))


def _score_keywords(parsed: ParsedTableQuery, schema: dict, doc: str) -> float:
    blob = f"{doc} {json.dumps(schema, ensure_ascii=False)[:400]}"
    hits = sum(1 for kw in parsed.keyword_terms[:10] if len(kw) >= 2 and kw in blob)
    return min(1.0, hits / max(1, min(6, len(parsed.keyword_terms[:10]))))


def _penalize_topic_mismatch(parsed: ParsedTableQuery, schema: dict) -> float:
    """Return penalty 0..SCORING_PENALTY_TOPIC_MISMATCH when table topic clearly conflicts with query."""
    topics = " ".join(schema.get("table_topics") or []).lower()
    penalty = 0.0
    q_topics = [t.lower() for t in parsed.table_topic_candidates]
    if any("화학" in t or "chemical" in t for t in q_topics):
        if "정기검사" in topics and "화학" not in topics:
            penalty += SCORING_PENALTY_TOPIC_MISMATCH
        if "용접" in topics or "시험재" in topics or "lot_treatment" in topics or "열처리로트" in topics:
            penalty += SCORING_PENALTY_TOPIC_MISMATCH * 0.85
    if any("정기검사" in t or "inspection" in t for t in q_topics):
        if "화학성분" in topics and "정기검사" not in topics:
            penalty += SCORING_PENALTY_TOPIC_MISMATCH * 0.55
    return min(SCORING_PENALTY_TOPIC_MISMATCH, penalty)


def score_table_candidate(
    parsed: ParsedTableQuery,
    *,
    vector_distance: float,
    meta: dict,
    document: str,
) -> TableScoreBreakdown:
    schema = parse_schema_from_document(document, meta)
    table_id = str(meta.get("table_id") or schema.get("table_id") or "")
    bd = TableScoreBreakdown(
        table_id=table_id,
        vector_distance=float(vector_distance),
        caption_match=_score_caption(parsed, schema, meta, document),
        table_topic_match=_score_topics(parsed, schema),
        column_match=_score_columns(parsed, schema),
        row_entity_match=_score_rows(parsed, schema),
        unit_match=_score_units(parsed, schema),
        keyword_match=_score_keywords(parsed, schema, document),
        chunk_type=str(meta.get("chunk_type") or ""),
        meta=dict(meta),
    )
    vec_sim = max(0.0, 1.0 - min(1.0, bd.vector_distance))
    caption_aux = _caption_aux_score(parsed, bd.caption_match, bd.table_topic_match, bd.column_match)
    combined = (
        _WEIGHTS["vector"] * vec_sim
        + _WEIGHTS["caption_match"] * caption_aux
        + _WEIGHTS["table_topic_match"] * bd.table_topic_match
        + _WEIGHTS["column_match"] * bd.column_match
        + _WEIGHTS["row_entity_match"] * bd.row_entity_match
        + _WEIGHTS["unit_match"] * bd.unit_match
        + _WEIGHTS["keyword_match"] * bd.keyword_match
        - _penalize_topic_mismatch(parsed, schema)
        + _apply_query_type_adjustments(parsed, bd, schema)
    )
    bd.combined_score = round(max(0.0, combined), 4)
    return bd


def _merge_grade_scan_candidates(
    collection,
    parsed: ParsedTableQuery,
    candidates: list[TableScoreBreakdown],
    *,
    doc_id: str | None,
    top_k: int = TABLE_SCHEMA_ROUTE_K,
) -> list[TableScoreBreakdown]:
    """Additive fallback: literal row-entity scan inside routed docs (pre-index schema gaps)."""
    if not parsed.row_entities:
        return candidates
    try:
        from table_first_retrieval import _scan_tables_for_grades
    except ImportError:
        return candidates

    scan_docs: list[str] = []
    if doc_id:
        scan_docs = [doc_id]
    else:
        for c in candidates:
            did = str((c.meta or {}).get("doc_id") or "")
            if did and did not in scan_docs:
                scan_docs.append(did)
    if not scan_docs:
        return candidates

    intent = "general_table"
    if any("화학" in t or "chemical" in t.lower() for t in parsed.table_topic_candidates):
        intent = "chemistry"
    elif any("정기검사" in t or "inspection" in t.lower() for t in parsed.table_topic_candidates):
        intent = "inspection"

    class _Slots:
        pass

    slots = _Slots()
    slots.intent = intent  # type: ignore[attr-defined]

    by_id = {c.table_id: c for c in candidates}
    scanned = _scan_tables_for_grades(collection, parsed.row_entities, scan_docs, slots=slots)
    needs_column = bool(parsed.column_entities) and parsed.query_type in ("cell_lookup", "column_lookup")
    for i, tid in enumerate(scanned):
        scan_score = round(0.82 - i * 0.035, 4)
        if tid in by_id:
            prev = by_id[tid]
            if needs_column and prev.column_match < 0.15:
                prev.row_entity_match = max(prev.row_entity_match, 0.55)
                prev.combined_score = round(max(prev.combined_score, scan_score * 0.55), 4)
            else:
                prev.row_entity_match = max(prev.row_entity_match, 0.9)
                prev.combined_score = round(max(prev.combined_score, scan_score), 4)
            continue
        row_match = 0.55 if needs_column else 0.9
        by_id[tid] = TableScoreBreakdown(
            table_id=tid,
            vector_distance=0.5,
            row_entity_match=row_match,
            table_topic_match=0.5 if parsed.table_topic_candidates else 0.0,
            combined_score=scan_score * (0.55 if needs_column else 1.0),
            meta={"table_id": tid, "doc_id": scan_docs[0]},
        )
    return sorted(by_id.values(), key=lambda x: (-x.combined_score, x.vector_distance))[:top_k]


def route_table_candidates(
    collection,
    question: str,
    model_name: str,
    parsed: ParsedTableQuery,
    *,
    doc_id: str | None = None,
    source: str | None = None,
    top_k: int = TABLE_SCHEMA_ROUTE_K,
    timing=None,
) -> list[TableScoreBreakdown]:
    embed_q = enrich_query_for_embedding(build_embed_query(parsed), model_name)
    vector = embed_texts_local([embed_q], model_name, for_query=True, timing=timing)[0]

    hits: list[tuple[str, float, dict, str]] = []
    for ctypes, n in (("table_schema", SCHEMA_FETCH), ("table_summary", SUMMARY_FETCH)):
        hits.extend(
            _query_typed(
                collection,
                vector,
                chunk_types=[ctypes],
                n_results=n,
                doc_id=doc_id,
                source=source,
            )
        )

    by_table: dict[str, TableScoreBreakdown] = {}
    for _cid, dist, meta, doc in hits:
        meta = meta or {}
        tid = str(meta.get("table_id") or "")
        if not tid:
            continue
        bd = score_table_candidate(parsed, vector_distance=float(dist), meta=meta, document=doc or "")
        prev = by_table.get(tid)
        if prev is None or bd.combined_score > prev.combined_score:
            by_table[tid] = bd

    ranked = sorted(by_table.values(), key=lambda x: (-x.combined_score, x.vector_distance))
    return _merge_grade_scan_candidates(collection, parsed, ranked, doc_id=doc_id, top_k=top_k)


def _row_column_match_score(parsed: ParsedTableQuery, text: str, meta: dict) -> float:
    row_ok = (
        best_entity_overlap(parsed.row_entities, [text]) if parsed.row_entities else 0.5
    )
    col_ok = (
        best_entity_overlap(parsed.column_entities, [text]) if parsed.column_entities else 0.5
    )
    if parsed.row_entities and parsed.column_entities:
        if row_ok < 0.2:
            return -0.25
        if col_ok < 0.2:
            return -0.20
        return 0.15 * row_ok + 0.15 * col_ok
    if parsed.row_entities and row_ok < 0.15:
        return -0.15
    if parsed.column_entities and col_ok < 0.15:
        return -0.15
    return 0.05


def fetch_stage2_chunks(
    collection,
    question: str,
    model_name: str,
    parsed: ParsedTableQuery,
    table_ids: list[str],
    *,
    doc_id: str | None = None,
    source: str | None = None,
    timing=None,
) -> list[tuple[str, float, dict, str, str]]:
    if not table_ids:
        return []

    embed_q = enrich_query_for_embedding(build_embed_query(parsed), model_name)
    vector = embed_texts_local([embed_q], model_name, for_query=True, timing=timing)[0]
    out: list[tuple[str, float, dict, str, str]] = []
    per_table_row: dict[str, int] = {}

    for chunk_type, per_limit in (("table_markdown", MARKDOWN_PER_TABLE), ("table_row", TABLE_SCHEMA_STAGE2_ROW_K)):
        hits = _query_typed(
            collection,
            vector,
            chunk_types=[chunk_type],
            n_results=min(len(table_ids) * per_limit * 4, 80),
            doc_id=doc_id,
            source=source,
            table_ids=table_ids,
        )
        rescored: list[tuple[str, float, dict, str, str]] = []
        for cid, dist, meta, doc in hits:
            adj = float(dist) - _row_column_match_score(parsed, doc or "", meta or {})
            if chunk_type == "table_row":
                adj -= 0.04
            rescored.append((cid, adj, meta or {}, doc or "", chunk_type))
        rescored.sort(key=lambda x: x[1])

        for cid, adj, meta, doc, ctype in rescored:
            tid = str(meta.get("table_id") or "")
            if tid not in table_ids:
                continue
            if ctype == "table_row":
                n = per_table_row.get(tid, 0)
                if n >= TABLE_SCHEMA_STAGE2_ROW_K:
                    continue
                per_table_row[tid] = n + 1
            out.append((cid, adj, meta, doc, ctype))
    return out


def compute_retrieval_confidence(
    parsed: ParsedTableQuery,
    candidates: list[TableScoreBreakdown],
    stage2: list[tuple[str, float, dict, str, str]],
) -> float:
    if not candidates:
        return 0.0
    top = candidates[0]
    conf = top.combined_score
    if len(candidates) > 1:
        gap = top.combined_score - candidates[1].combined_score
        conf += min(0.15, gap * 0.5)
    if parsed.row_entities and top.row_entity_match < 0.3:
        conf -= 0.20
    if parsed.column_entities and top.column_match < 0.3:
        conf -= 0.15
    if stage2:
        best_row = min((s[1] for s in stage2 if s[4] == "table_row"), default=1.0)
        conf += max(0.0, 0.12 - best_row * 0.1)
    return round(max(0.0, min(1.0, conf)), 3)


def _match_labels(parsed: ParsedTableQuery, stage2: list) -> tuple[str, str]:
    matched_row = ""
    matched_col = ""
    for _cid, _adj, _meta, doc, ctype in stage2:
        if ctype != "table_row":
            continue
        if not matched_row and parsed.row_entities:
            for re_ent in parsed.row_entities:
                if entity_matches(expand_entity_aliases(re_ent), doc or ""):
                    matched_row = re_ent
                    break
        if not matched_col and parsed.column_entities:
            for ce in parsed.column_entities:
                if entity_matches(expand_entity_aliases(ce), doc or ""):
                    matched_col = ce
                    break
    return matched_row, matched_col


def build_table_schema_raw(
    collection,
    question: str,
    model_name: str,
    *,
    top_k: int,
    doc_id: str | None = None,
    source: str | None = None,
    timing=None,
) -> dict[str, Any]:
    parsed = parse_table_query(question)
    candidates = route_table_candidates(
        collection,
        question,
        model_name,
        parsed,
        doc_id=doc_id,
        source=source,
        timing=timing,
    )
    table_ids = [c.table_id for c in candidates if c.table_id]
    table_score = {c.table_id: c.combined_score for c in candidates}

    embed_q = enrich_query_for_embedding(build_embed_query(parsed), model_name)
    vector = embed_texts_local([embed_q], model_name, for_query=True, timing=timing)[0]

    pool: dict[str, tuple[float, dict, str]] = {}

    route_hits = _query_typed(
        collection,
        vector,
        chunk_types=["table_schema", "table_summary"],
        n_results=ROUTE_FETCH,
        doc_id=doc_id,
        source=source,
        table_ids=table_ids[: max(TABLE_SCHEMA_ROUTE_K, 3)] if table_ids else None,
    )
    for cid, dist, meta, doc in route_hits:
        tid = str(meta.get("table_id") or "")
        combined = table_score.get(tid, 0.0)
        adj = float(dist) - combined * 0.85
        prev = pool.get(cid)
        if prev is None or adj < prev[0]:
            pool[cid] = (adj, meta or {}, doc or "")

    stage2 = fetch_stage2_chunks(
        collection,
        question,
        model_name,
        parsed,
        table_ids,
        doc_id=doc_id,
        source=source,
        timing=timing,
    )

    for cid, adj, meta, doc, _ctype in stage2:
        prev = pool.get(cid)
        if prev is None or adj < prev[0]:
            pool[cid] = (adj, meta, doc)

    confidence = compute_retrieval_confidence(parsed, candidates, stage2)
    matched_row, matched_col = _match_labels(parsed, stage2)
    selected_table_id = table_ids[0] if table_ids else ""

    debug = {
        "parsed_query": parsed.to_dict(),
        "selected_table_candidates": [c.to_dict() for c in candidates],
        "selected_table_id": selected_table_id,
        "matched_row": matched_row,
        "matched_column": matched_col,
        "retrieval_confidence": confidence,
        "passes_confidence_gate": confidence >= CONFIDENCE_GATE_THRESHOLD,
        "confidence_threshold": CONFIDENCE_GATE_THRESHOLD,
    }

    ranked = sorted(pool.items(), key=lambda x: x[1][0])[:top_k]
    return {
        "ids": [[cid for cid, _ in ranked]],
        "distances": [[score for _, (score, _, _) in ranked]],
        "metadatas": [[meta for _, (_, meta, _) in ranked]],
        "documents": [[doc for _, (_, _, doc) in ranked]],
        "table_schema_retrieval": True,
        "routed_table_ids": table_ids,
        "table_retrieval_debug": debug,
    }


def apply_confidence_gate(answer: str, debug: dict | None) -> str:
    if not debug:
        return answer
    if debug.get("passes_confidence_gate", True):
        return answer
    return CONFIDENCE_GATE_MESSAGE
