"""RAG answer generation with 1.2 structured Korean output."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embedding_policy import DEFAULT_EMBEDDING_PRESET, embed_texts_local, resolve_embedding_config
from rag_eval_lib import keyword_hits, load_chunk_text_map, load_questions
from retrieval_search import enrich_query_for_embedding, query_with_hybrid_ranking
from table_retrieval import (
    annotate_matched_columns,
    is_table_question,
    merge_table_aware_into_raw,
    query_table_chunks,
)
from meeting_outcome_retrieval import (
    is_meeting_outcome_question,
    merge_meeting_outcome_into_raw,
    query_meeting_outcome_chunks,
)
from meeting_outcome_answer import (
    build_meeting_outcome_system_prompt,
    build_meeting_outcome_user_prompt,
)
from retrieval_query_analysis import analyze_query
from answer_depth_guidance import (
    ANSWER_DENSITY_GUIDANCE,
    ANTI_REPETITION_GUIDANCE,
    CITATION_GUIDANCE,
    ENV_REGULATION_V01_HINT,
    EVIDENCE_DISPLAY_GUIDANCE,
    FORMAT_RULES,
    GOOD_BAD_EXAMPLES,
    RULE_LOOKUP_EVIDENCE_GUIDANCE,
    RULE_LOOKUP_GUIDANCE,
    SECTION2_OPERATIONAL_GUIDANCE,
    SECTION2_RULE_LOOKUP_GUIDANCE,
    SECTION3_FOLLOWUP_GUIDANCE,
    SECTION4_GUIDANCE,
    category_bullet_budget,
)

DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_BASE = "http://localhost:11434"
PILOT_REFERENCE_PATH = Path(__file__).resolve().parent.parent / "data/eval/pilot_validation_reference.jsonl"
_REFERENCE_CACHE: dict[str, dict] | None = None


def reference_for_question(row: dict, reference: dict | None = None) -> dict | None:
    global _REFERENCE_CACHE
    if reference is not None:
        return reference
    if _REFERENCE_CACHE is None:
        _REFERENCE_CACHE = (
            load_reference_answers(PILOT_REFERENCE_PATH) if PILOT_REFERENCE_PATH.exists() else {}
        )
    return _REFERENCE_CACHE.get(str(row.get("question_id", "")))

CATEGORY_GUIDANCE = {
    "trend_summary": (
        "최신 동향 요약: '## 1) 핵심 요약'에 bullet 7~10개, **각 bullet 2문장 이상**. "
        "3단 구조(논의·규제의미·업무영향). '논의되었습니다'로 끝내지 말 것. "
        "첫 3개 bullet은 최상위 결론/안건(**굵게**). 회의명·문서번호·수치 포함."
    ),
    "env_regulation": (
        "환경규제 대응: §1 bullet **5~7개**, 각 2문장 이상. "
        "GHG·MARPOL Annex VI·Net-Zero·CII·SEEMP·EEXI·연료·배출 데이터·리포팅. "
        "회의차수·문서번호·조항을 bullet에 명시. 규제 의미와 선사 실무 영향을 '따라서/그 결과'로 연결."
    ),
    "autonomous": (
        "자율운항: §1 bullet **5~7개**, 각 2문장 이상. "
        "MSC MASS Code·WG 결정·mandatory code 일정·goal-based 요건. "
        "mandatory/non-mandatory 일정은 날짜·회의차수와 함께 명시."
    ),
    "rule_lookup": (
        "단순 Rule 질문: '## 1) 핵심 요약'에 관련 Rule/Guidance 2~3 bullet. "
        "문서명·번호를 bullet 첫머리에 **굵게** 표기. "
        "각 bullet 2문장 이상 — scope·적용대상·주요 요건·예외를 context에서 구체 인용. "
        "**각 bullet 끝 citation [N] 필수.**"
    ),
}

QUESTION_HINTS: dict[str, str] = {
    "V01": (
        "MEPC 84 ISWG-GHG 20차(문서 MEPC 84/7/14). bullet마다 3단 구조(논의·규제의미·업무영향). "
        "IMO Net-Zero Framework, GFI compliance/reporting, SEEMP Guidelines, "
        "MARPOL Annex VI regulation 36·37, Fifth IMO GHG Study, LCA Guidelines. "
        "'논의되었습니다'로 끝내지 말고 따라서/그 결과로 영향까지. "
        "MEPC/ES.2 등 질문과 무관한 다른 회의는 넣지 말 것."
    ),
    "V02": "질문이 '3개 항목' 요약이므로 핵심 요약 상단 3 bullet을 TC 75 / C 134 / MEPC ES.2 등 3대 outcome 축으로 구분.",
    "V03": "MEPC 84-6-2 CII fleet report 2024. CII rating, reporting year 2024, fleet trend, operational 보고 영향.",
    "V04": "MSC 111-12 ISE Sub-Committee. alternative fuel safety, GHG 연계 안전규제 — MSC=안전 측면 우선.",
    "V05": "MSC 111-5 MASS WG Report. MASS Code 결정, mandatory code 일정, goal-based, degree of autonomy.",
    "V06": "DNV-CG-0264 Autonomous and remotely operated vessels. Smart/autonomous notation·guidance. **각 bullet 끝 [N] citation 필수.**",
    "V07": "LR Notice No.1 Section 15. low-flashpoint fuel, engines, alternative/dual fuel. **각 bullet 끝 [N] citation 필수.**",
}

SOURCE_ROLE = {
    "MEPC": "MEPC(해양환경보호위원회) — 배출·GHG·에너지효율·MARPOL 등 환경규제 본체 중심으로 답변.",
    "MSC": "MSC(해상안전위원회) — 환경규제와 연결되는 안전·운영·MASS·대체연료 안전 측면 중심으로 답변.",
    "DNV": "DNV Classification Rule/Guidance 인용.",
    "LR": "LR Rules/Notice 인용.",
    "KR": "KR Rules 인용.",
    "ABS": "ABS Rules/Guidance 인용.",
}


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    source: str
    file_name: str
    page_number: int | None
    clause_number: str
    element_type: str
    distance: float
    text: str
    matched_keywords: list[str] = field(default_factory=list)
    matched_topics: list[str] = field(default_factory=list)
    content_preview: str = ""
    chunk_type: str = ""
    table_id: str = ""
    caption: str = ""
    row_index: int | None = None
    matched_columns: list[str] = field(default_factory=list)
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    metadata_boost: float = 0.0
    source_priority_score: float = 0.0
    is_catalog_table: bool = False
    catalog_doc_candidates: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    question_id: str
    category: str
    question: str
    retrieval_sources: list[str]
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    retrieval_keyword_hits: int = 0
    retrieval_keyword_total: int = 0
    retrieval_source_hits: int = 0
    retrieval_hit_at_k: bool = False
    gold_doc_hit_at_k: bool = False
    gold_page_hit_at_k: bool = False
    gold_doc_rank: int | None = None
    retrieval_metrics: dict = field(default_factory=dict)
    retrieval_variant: str = "baseline"
    answer: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    error: str = ""


def load_unified_collection(unified_id: str, index_dir: Path, timing=None):
    from rag_resource_cache import load_unified_collection as _load

    return _load(unified_id, index_dir, timing=timing)


def _retrieved_from_fused_hits(
    fused_hits,
    *,
    question: str,
    chunks_dir: Path,
    preview_chars: int,
) -> list[RetrievedChunk]:
    chunk_text_cache: dict[str, dict[str, str]] = {}
    out: list[RetrievedChunk] = []
    for hit in fused_hits:
        meta = hit.meta or {}
        chunk_id = hit.chunk_id
        doc_id = str(meta.get("doc_id", ""))
        chunk_type = str(meta.get("chunk_type") or "")
        table_id = str(meta.get("table_id") or "")
        caption = str(meta.get("caption") or "")
        row_index_raw = meta.get("row_index")
        row_index = int(row_index_raw) if row_index_raw is not None and str(row_index_raw) != "" else None
        matched_cols = annotate_matched_columns(question, meta) if chunk_type else []
        if doc_id not in chunk_text_cache:
            text_map: dict[str, str] = {}
            for name in ("chunks.jsonl", "table_chunks.jsonl"):
                chunks_path = chunks_dir / doc_id / name
                if chunks_path.exists():
                    text_map.update(load_chunk_text_map(chunks_path))
            chunk_text_cache[doc_id] = text_map
        full_text = chunk_text_cache[doc_id].get(chunk_id) or hit.document or ""
        if len(full_text) > preview_chars:
            full_text = full_text[:preview_chars] + "\n...(truncated)"
        out.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source=str(meta.get("source", "")),
                file_name=str(meta.get("file_name", "")),
                page_number=meta.get("page_number"),
                clause_number=str(meta.get("clause_number") or meta.get("article_number") or ""),
                element_type=str(meta.get("element_type", "")),
                distance=float(hit.distance),
                text=full_text,
                chunk_type=chunk_type,
                table_id=table_id,
                caption=caption,
                row_index=row_index,
                matched_columns=matched_cols,
                dense_score=hit.dense_score,
                bm25_score=hit.bm25_score,
                rrf_score=hit.rrf_score,
                metadata_boost=hit.metadata_boost,
                source_priority_score=hit.source_priority_score,
                is_catalog_table=hit.is_catalog_table,
                catalog_doc_candidates=list(hit.catalog_candidates),
            )
        )
    return out


def retrieve_for_question(
    collection,
    model_name: str,
    row: dict,
    *,
    top_k: int,
    chunks_dir: Path,
    preview_chars: int = 4000,
    fetch_k: int | None = None,
    gold_doc_filter: bool | None = None,
    narrow_doc_id: str | None = None,
    unified_id: str | None = None,
    index_dir: Path | None = None,
    timing=None,
) -> list[RetrievedChunk]:
    question = str(row["question"])
    sources = list(row.get("retrieval_sources") or [])
    source_filter = sources[0] if len(sources) == 1 else None
    if not source_filter:
        from retrieval_query_analysis import detect_class_society_hint
        from retrieval_verification import effective_question_category

        society_hint = str(row.get("class_society_hint") or detect_class_society_hint(question))
        if society_hint:
            row["class_society_hint"] = society_hint
            category = str(row.get("category") or "") or effective_question_category(question, row)
            if category == "rule_lookup":
                source_filter = society_hint
    n_fetch = fetch_k or top_k
    if gold_doc_filter is None:
        use_gold = False
    else:
        use_gold = gold_doc_filter

    if narrow_doc_id:
        filter_doc_id = narrow_doc_id
    elif use_gold and row.get("gold_doc_id"):
        filter_doc_id = str(row["gold_doc_id"])
    else:
        filter_doc_id = None

    embed_query = enrich_query_for_embedding(question, model_name)
    vector = embed_texts_local([embed_query], model_name, for_query=True, timing=timing)[0]

    use_table_first = bool(row.get("_table_qa"))
    if use_table_first:
        from table_rag_config import use_table_schema_retrieval

        if use_table_schema_retrieval():
            from table_schema_retrieval import build_table_schema_raw

            raw = build_table_schema_raw(
                collection,
                question,
                model_name,
                top_k=n_fetch,
                doc_id=filter_doc_id,
                source=source_filter,
                timing=timing,
            )
        else:
            from table_first_retrieval import build_table_first_raw

            raw = build_table_first_raw(
                collection,
                question,
                model_name,
                top_k=n_fetch,
                doc_id=filter_doc_id,
                source=source_filter,
                timing=timing,
            )
        if raw.get("table_retrieval_debug"):
            row["_table_retrieval_debug"] = raw["table_retrieval_debug"]
    else:
        from retrieval_verification import effective_question_category

        category = str(row.get("category") or "") or effective_question_category(question, row)
        use_hybrid = row.get("_use_hybrid_bm25", True)
        legacy_cat = str(row.get("_eval_category") or row.get("category") or category)

        from meeting_category_profile import (
            build_meeting_retrieval_profile,
            uses_structured_meeting_answer,
        )

        if uses_structured_meeting_answer(row, legacy_category=legacy_cat) and use_hybrid is not False:
            from pathlib import Path as _Path

            from meeting_hybrid_retrieval import meeting_hybrid_search
            from hybrid_retrieval import get_bm25_index
            from rag_inprocess import DEFAULT_INDEX_DIR, DEFAULT_UNIFIED
            from rag_resource_cache import unified_index_fingerprint

            mprofile = build_meeting_retrieval_profile(
                question, row, legacy_category=legacy_cat
            )
            row["_meeting_retrieval_profile"] = mprofile.to_log_dict()
            row["_top_level_category"] = mprofile.top_level_category
            row["_internal_intent"] = mprofile.internal_intent

            uid = unified_id or str(row.get("unified_id") or DEFAULT_UNIFIED)
            idir = index_dir or _Path(str(row.get("index_dir") or DEFAULT_INDEX_DIR))
            fp = unified_index_fingerprint(uid, idir)
            bm25 = get_bm25_index(collection, uid, idir, fingerprint=fp)
            fused, log_payload = meeting_hybrid_search(
                collection,
                bm25,
                question,
                vector,
                profile=mprofile,
                fetch_k=max(n_fetch, 48),
                top_k=n_fetch,
                society=source_filter,
                doc_id=filter_doc_id,
                timing=timing,
            )
            row["_hybrid_retrieval_log"] = log_payload
            row["warning_flags"] = list(
                dict.fromkeys((row.get("warning_flags") or []) + log_payload.get("warning_flags", []))
            )
            if timing is not None and hasattr(timing, "mark"):
                timing.mark("t_context_build_start")
            out = _retrieved_from_fused_hits(
                fused,
                question=question,
                chunks_dir=chunks_dir,
                preview_chars=preview_chars,
            )
            if timing is not None and hasattr(timing, "mark"):
                timing.mark("t_context_build_end")
            return out

        if category == "rule_lookup" and use_hybrid is not False:
            from pathlib import Path as _Path

            from hybrid_retrieval import get_bm25_index, hybrid_rule_lookup_search
            from rag_inprocess import DEFAULT_INDEX_DIR, DEFAULT_UNIFIED
            from rag_resource_cache import unified_index_fingerprint

            uid = unified_id or str(row.get("unified_id") or DEFAULT_UNIFIED)
            idir = index_dir or _Path(str(row.get("index_dir") or DEFAULT_INDEX_DIR))
            fp = unified_index_fingerprint(uid, idir)
            bm25 = get_bm25_index(collection, uid, idir, fingerprint=fp)
            fused, log_payload = hybrid_rule_lookup_search(
                collection,
                bm25,
                question,
                vector,
                fetch_k=max(n_fetch, 40),
                top_k=n_fetch,
                society=source_filter,
                doc_id=filter_doc_id,
                timing=timing,
                hard_source_filter=bool(source_filter or row.get("_hard_society_filter")),
            )
            row["_hybrid_retrieval_log"] = log_payload
            row["warning_flags"] = list(
                dict.fromkeys((row.get("warning_flags") or []) + log_payload.get("warning_flags", []))
            )
            if timing is not None and hasattr(timing, "mark"):
                timing.mark("t_context_build_start")
            out = _retrieved_from_fused_hits(
                fused,
                question=question,
                chunks_dir=chunks_dir,
                preview_chars=preview_chars,
            )
            if timing is not None and hasattr(timing, "mark"):
                timing.mark("t_context_build_end")
            return out

        raw = query_with_hybrid_ranking(
            collection,
            question,
            vector,
            top_k=n_fetch,
            fetch_k=max(n_fetch * 10, 80),
            source=source_filter,
            doc_id=filter_doc_id,
            timing=timing,
        )

        if is_table_question(question):
            table_by_type = query_table_chunks(
                collection,
                question,
                model_name,
                top_k=max(6, n_fetch // 2),
                doc_id=filter_doc_id,
                source=source_filter,
                timing=timing,
            )
            if any(table_by_type.values()):
                raw = merge_table_aware_into_raw(raw, table_by_type, top_k=n_fetch, question=question)

    if is_meeting_outcome_question(question, row):
        signals = analyze_query(question)
        meeting_hits = query_meeting_outcome_chunks(
            collection,
            question,
            model_name,
            signals,
            top_k=max(8, n_fetch // 2),
            doc_id=filter_doc_id,
            source=source_filter,
            timing=timing,
        )
        if meeting_hits:
            sig = analyze_query(question)
            raw = merge_meeting_outcome_into_raw(
                raw,
                meeting_hits,
                top_k=n_fetch,
                min_outcome_chunks=2,
                topic_specific=bool(sig.topics),
                question=question,
            )

    if len(sources) > 1:
        if timing is not None and hasattr(timing, "mark"):
            timing.mark("t_metadata_filter_start")
        raw_all = query_with_hybrid_ranking(
            collection,
            question,
            vector,
            top_k=n_fetch * 2,
            fetch_k=max(n_fetch * 15, 100),
            timing=timing,
        )
        ids = raw_all["ids"][0]
        metas = raw_all["metadatas"][0]
        dists = raw_all["distances"][0]
        docs = raw_all["documents"][0]
        allowed = {s.upper() for s in sources}
        filtered = [
            (i, d, m, doc)
            for i, d, m, doc in zip(ids, dists, metas, docs)
            if str((m or {}).get("source", "")).upper() in allowed
        ][:n_fetch]
        if filtered:
            raw = {
                "ids": [[x[0] for x in filtered]],
                "distances": [[x[1] for x in filtered]],
                "metadatas": [[x[2] for x in filtered]],
                "documents": [[x[3] for x in filtered]],
            }
        if timing is not None and hasattr(timing, "mark"):
            timing.mark("t_metadata_filter_end")

    if timing is not None and hasattr(timing, "mark"):
        timing.mark("t_context_build_start")

    chunk_text_cache: dict[str, dict[str, str]] = {}
    out: list[RetrievedChunk] = []
    for chunk_id, distance, meta, doc in zip(
        raw["ids"][0],
        raw["distances"][0],
        raw["metadatas"][0],
        raw["documents"][0],
    ):
        meta = meta or {}
        doc_id = str(meta.get("doc_id", ""))
        chunk_type = str(meta.get("chunk_type") or "")
        table_id = str(meta.get("table_id") or "")
        caption = str(meta.get("caption") or "")
        row_index_raw = meta.get("row_index")
        row_index = int(row_index_raw) if row_index_raw is not None and str(row_index_raw) != "" else None
        matched_cols = annotate_matched_columns(question, meta) if chunk_type else []
        if doc_id not in chunk_text_cache:
            text_map: dict[str, str] = {}
            for name in ("chunks.jsonl", "table_chunks.jsonl"):
                chunks_path = chunks_dir / doc_id / name
                if chunks_path.exists():
                    text_map.update(load_chunk_text_map(chunks_path))
            chunk_text_cache[doc_id] = text_map
        full_text = chunk_text_cache[doc_id].get(chunk_id) or doc or ""
        if len(full_text) > preview_chars:
            full_text = full_text[:preview_chars] + "\n...(truncated)"
        out.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source=str(meta.get("source", "")),
                file_name=str(meta.get("file_name", "")),
                page_number=meta.get("page_number"),
                clause_number=str(meta.get("clause_number") or meta.get("article_number") or ""),
                element_type=str(meta.get("element_type", "")),
                distance=float(distance),
                text=full_text,
                chunk_type=chunk_type,
                table_id=table_id,
                caption=caption,
                row_index=row_index,
                matched_columns=matched_cols,
            )
        )
    if timing is not None and hasattr(timing, "mark"):
        timing.mark("t_context_build_end")
    return out


def supplement_gold_pages_for_llm(
    pool: list[RetrievedChunk],
    row: dict,
    chunks_dir: Path,
    *,
    preview_chars: int = 4000,
) -> list[RetrievedChunk]:
    """Inject substantive chunks from eval gold_pages (often exec-summary) missing from vector hits."""
    from rag_eval_lib import load_chunks
    from retrieval_chunk_quality import is_thin_chunk, substantive_len

    doc_id = str(row.get("gold_doc_id") or "")
    pages = row.get("gold_pages") or []
    if not doc_id or not pages:
        return pool

    chunks_path = chunks_dir / doc_id / "chunks.jsonl"
    if not chunks_path.exists():
        return pool

    existing_ids = {c.chunk_id for c in pool}
    file_name = next((c.file_name for c in pool if c.doc_id == doc_id and c.file_name), "")
    source = str(
        row.get("gold_source")
        or next((c.source for c in pool if c.doc_id == doc_id), "")
        or (row.get("retrieval_sources") or ["MEPC"])[0]
    )

    best_by_page: dict[int, dict] = {}
    keywords = [str(k) for k in (row.get("expected_keywords") or [])]
    for ch in load_chunks(chunks_path):
        try:
            page = int(ch.get("page_number", -1))
        except (TypeError, ValueError):
            continue
        if page not in pages:
            continue
        text = str(ch.get("text") or "")
        if is_thin_chunk(text, min_chars=100):
            continue
        kw_hits, _ = keyword_hits(text, keywords) if keywords else (0, 0)
        score = substantive_len(text) + kw_hits * 400
        prev = best_by_page.get(page)
        if prev:
            prev_text = str(prev.get("text") or "")
            prev_kw, _ = keyword_hits(prev_text, keywords) if keywords else (0, 0)
            prev_score = substantive_len(prev_text) + prev_kw * 400
            if score <= prev_score:
                continue
        best_by_page[page] = ch

    if not best_by_page:
        return pool

    out = list(pool)
    for page in sorted(pages):
        ch = best_by_page.get(int(page))
        if not ch:
            continue
        chunk_id = str(ch.get("chunk_id", ""))
        if not chunk_id or chunk_id in existing_ids:
            continue
        full_text = str(ch.get("text") or "")
        if len(full_text) > preview_chars:
            full_text = full_text[:preview_chars] + "\n...(truncated)"
        out.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source=source,
                file_name=file_name or str(ch.get("file_name") or ""),
                page_number=int(page),
                clause_number=str(ch.get("clause_number") or ch.get("article_number") or ""),
                element_type=str(ch.get("element_type", "")),
                distance=0.04,
                text=full_text,
            )
        )
        existing_ids.add(chunk_id)
    return out


def score_retrieval(
    row: dict, retrieved: list[RetrievedChunk], top_k: int, *, eval_k: int = 5
) -> tuple[int, int, int, bool, bool, bool, int | None, dict]:
    from rag_retrieval_metrics import compute_retrieval_metrics, resolve_gold_pages

    metrics = compute_retrieval_metrics(row, retrieved, top_k=top_k, eval_k=eval_k)
    gold_pages = resolve_gold_pages(row)
    gold_page_hit = metrics.gold_page_set_hit_at_5 if gold_pages else False
    if not gold_pages:
        gp = row.get("gold_page")
        gold_doc = str(row.get("gold_doc_id") or "")
        if gold_doc and gp is not None:
            for c in retrieved[:top_k]:
                if c.doc_id == gold_doc and c.page_number == int(gp):
                    gold_page_hit = True
                    break

    hit_at_k = metrics.gold_doc_hit_at_5 if row.get("gold_doc_id") else (
        metrics.best_keyword_hits >= max(2, (metrics.keyword_total + 1) // 2)
        if metrics.keyword_total
        else False
    )
    return (
        metrics.best_keyword_hits,
        metrics.keyword_total,
        metrics.source_hits_in_top_k,
        hit_at_k,
        metrics.gold_doc_hit_at_5,
        gold_page_hit,
        metrics.gold_doc_rank,
        metrics.to_dict(),
    )


def _rule_lookup_section4_prompt(category: str) -> str:
    return SECTION4_GUIDANCE


def build_context_block(retrieved: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for i, c in enumerate(retrieved, start=1):
        cite = f"[{i}] source={c.source} | doc={c.file_name or c.doc_id} | p{c.page_number}"
        if c.clause_number:
            cite += f" | clause={c.clause_number}"
        blocks.append(f"{cite}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def build_system_prompt(row: dict) -> str:
    if is_meeting_outcome_question(str(row.get("question", "")), row):
        return build_meeting_outcome_system_prompt(row)
    from question_classifier import classify_question_category

    category = str(row.get("category") or "").strip()
    if category not in CATEGORY_GUIDANCE:
        category = classify_question_category(str(row.get("question", "")), row)
    sources = row.get("retrieval_sources") or []
    role_lines = [SOURCE_ROLE.get(str(s), "") for s in sources if SOURCE_ROLE.get(str(s))]
    bullet_min, bullet_max, priority = category_bullet_budget(category, row)
    qid = str(row.get("question_id", ""))
    hint = QUESTION_HINTS.get(qid, "")
    section2 = (
        SECTION2_RULE_LOOKUP_GUIDANCE
        if category == "rule_lookup"
        else SECTION2_OPERATIONAL_GUIDANCE
    )
    rule_block = RULE_LOOKUP_GUIDANCE if category == "rule_lookup" else ""
    evidence_block = RULE_LOOKUP_EVIDENCE_GUIDANCE if category == "rule_lookup" else ""
    if category == "rule_lookup":
        section1_bullets = (
            f"- bullet {bullet_min}~{bullet_max}개 — **문서별 고유 내용** (동일 결론 문장 반복 금지)\n"
            "- 각 bullet 1~2문장 + 회의/문서/조항 명시 + citation [N]"
        )
    else:
        section1_bullets = (
            f"- bullet {bullet_min}~{bullet_max}개 (상단 {priority}개 최우선 — **굵게**)\n"
            "- 각 bullet **최소 2문장**, 3단 구조(논의·규제의미·업무영향)\n"
            "- 각 bullet 끝 citation [번호]\n"
            "- 서로 다른 문서·안건별 bullet 분리"
        )

    return f"""당신은 해운사 규제·선급 Rule 전문 조력자입니다.
반드시 한국어로 답변합니다. 제공된 검색 근거(context)에 없는 사실·날짜·수치·결의번호는 추측하지 마세요.
{FORMAT_RULES}
{EVIDENCE_DISPLAY_GUIDANCE}
{ANTI_REPETITION_GUIDANCE}
{CITATION_GUIDANCE}
{evidence_block}
{ANSWER_DENSITY_GUIDANCE}
{rule_block}

카테고리: {category}
{CATEGORY_GUIDANCE.get(category, '')}
{chr(10).join(role_lines)}
{f"질문별 지침: {hint}" if hint else ""}

## 1) 핵심 요약
{section1_bullets}

{section2}

{SECTION3_FOLLOWUP_GUIDANCE}

{_rule_lookup_section4_prompt(category)}

작성 규칙:
- context에 없으면 "## 3)"에 "검색 결과 내 확인 불가 — (이유)" 기록
- 여러 문서 검색 시 문서명 명시; 단일 문서 편중 시 ## 3) [해석 근거] 태그
- 한글·영문·숫자·기호만 사용
{GOOD_BAD_EXAMPLES}
"""


def build_user_prompt(
    row: dict,
    context: str,
    reference: dict | None = None,
    retrieved: list[RetrievedChunk] | None = None,
) -> str:
    if is_meeting_outcome_question(str(row.get("question", "")), row):
        return build_meeting_outcome_user_prompt(row, context)
    keywords = row.get("expected_keywords") or []
    topics = row.get("expected_topics") or []
    coverage = ""
    if keywords or topics:
        parts = []
        if keywords:
            parts.append(f"키워드(가능한 범위에서 반영): {', '.join(keywords)}")
        if topics:
            parts.append(f"토픽(가능한 범위에서 반영): {', '.join(topics)}")
        coverage = "\n" + "\n".join(parts) + "\n(키워드가 context에 없으면 억지로 넣지 마세요.)\n"

    ref_block = ""
    if reference:
        qc = reference.get("quality_criteria")
        must = reference.get("must_cover")
        outline = reference.get("example_answer_outline")
        parts = []
        if qc:
            parts.append(f"[품질 기준] {qc}")
        if must:
            parts.append(f"[반드시 다룰 주제 — context에 있을 때 각각 별도 bullet] {', '.join(must)}")
        if outline:
            topic_lines = []
            for ln in str(outline).splitlines():
                ln = ln.strip()
                if ln.startswith("- "):
                    topic = ln[2:].split("[")[0].strip()
                    if topic and not topic.startswith("##"):
                        topic_lines.append(topic)
            if topic_lines:
                parts.append(
                    "[주제 체크리스트 — 아래 각 항목을 context 근거로 2문장 bullet 작성, 문장 복사 금지]\n"
                    + "\n".join(f"  · {t}" for t in topic_lines[:12])
                )
        if parts:
            ref_block = "\n" + "\n".join(parts) + "\n"

    structure_hint = ""
    qid = str(row.get("question_id", ""))
    if qid == "V01":
        structure_hint = ENV_REGULATION_V01_HINT

    from question_classifier import classify_question_category

    cat = str(row.get("category") or "").strip()
    if cat not in CATEGORY_GUIDANCE:
        cat = classify_question_category(str(row.get("question", "")), row)

    if cat == "rule_lookup":
        from rule_lookup_context import citation_doc_manifest

        manifest = citation_doc_manifest(retrieved or [])
        depth_reminder = (
            "\n**중요 (Rule/Guidance 조회):**\n"
            "- §1: **인용 허용 문서 목록의 file_name별** 서로 다른 scope·notation·요건만\n"
            "- §2: **1~2 bullet** 통합 실무 조치만 (§1 문서를 다시 나열하지 말 것)\n"
            "- citation [N]의 **file_name과 bullet 문서명이 일치**해야 함\n"
            "- 목록에 없는 RP/CG/RU 번호·placeholder 출력 **금지**\n"
            "- **§4는 작성하지 말 것** (시스템 자동 생성)\n"
            f"\n{manifest}\n"
        )
    else:
        depth_reminder = (
            "\n**중요:**\n"
            f"- ## 1) 최소 {int(row.get('answer_bullets_min', 5))}개 bullet, ## 2) 최소 3개 bullet\n"
            "- 각 bullet **2~3문장 보고서형** — 3요소(논의·개정·요구 / 규제 의미 / 업무 영향) 중 **최소 2개**\n"
            "- **§1·§2 bullet마다 문장 끝 citation [N] 필수** (context 청크 번호와 일치)\n"
            "- `↔`·쉼표 키워드 나열로 끝내지 말고 **완전한 문장**으로\n"
            "- **동일·유사 문장 반복 금지** — §1↔§2 복사, bullet 간 같은 결론 문구 반복 금지\n"
            "- ## 3) [미확정 규제]/[해석 근거]/[선급별 상이 요구] 태그\n"
        )

    output_scope = (
        "위 context만 사용하여 **§1~§3만** Markdown으로 작성하세요 (§4 출력 금지)."
        if cat == "rule_lookup"
        else "위 context만 사용하여 1.2 형식 4섹션 Markdown 답변을 작성하세요."
    )

    return f"""질문:
{row['question']}
{coverage}{ref_block}{structure_hint}{depth_reminder}
검색 근거 (context):
{context}

{output_scope}
추론 과정이나 서두 설명 없이, 바로 '## 1) 핵심 요약'부터 출력하세요."""


def _ollama_chat_payload(
    model: str,
    system: str,
    user: str,
    *,
    stream: bool,
    temperature: float = 0.1,
    num_predict: int = 2500,
    num_ctx: int = 16384,
) -> bytes:
    return json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": 0.92,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "repeat_penalty": 1.12,
                "keep_alive": "30m",
            },
        }
    ).encode("utf-8")


def call_ollama_chat(
    model: str,
    system: str,
    user: str,
    base_url: str,
    timeout: int = 600,
    *,
    temperature: float = 0.15,
    num_predict: int = 2500,
) -> str:
    payload = _ollama_chat_payload(
        model, system, user, stream=False, temperature=temperature, num_predict=num_predict
    )
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str(data.get("message", {}).get("content", "")).strip()


def call_ollama_chat_stream(
    model: str,
    system: str,
    user: str,
    base_url: str,
    timeout: int = 300,
    *,
    temperature: float = 0.1,
) -> Iterator[str]:
    payload = _ollama_chat_payload(model, system, user, stream=True, temperature=temperature)
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line:
                continue
            data = json.loads(line)
            chunk = data.get("message", {}).get("content", "")
            if chunk:
                yield chunk
            if data.get("done"):
                break


def call_ollama_chat_timed(
    model: str,
    system: str,
    user: str,
    base_url: str,
    timeout: int = 900,
    *,
    temperature: float = 0.15,
    num_predict: int = 2500,
    num_ctx: int = 16384,
    timing=None,
    on_token=None,
) -> str:
    """Stream Ollama chat and record t_llm_request_start / t_first_token / t_llm_response_end."""
    if timing is not None and hasattr(timing, "mark"):
        if "t_llm_request_start" not in timing.monotonic:
            timing.mark("t_llm_request_start")
    payload = _ollama_chat_payload(
        model,
        system,
        user,
        stream=True,
        temperature=temperature,
        num_predict=num_predict,
        num_ctx=num_ctx,
    )
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    parts: list[str] = []
    first_token = False
    ollama_stream_meta: dict[str, Any] = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line:
                continue
            data = json.loads(line)
            chunk = data.get("message", {}).get("content", "")
            if chunk:
                if (
                    not first_token
                    and timing is not None
                    and hasattr(timing, "mark")
                    and "t_first_token" not in timing.monotonic
                ):
                    timing.mark("t_first_token")
                    first_token = True
                parts.append(chunk)
                if on_token is not None:
                    on_token(chunk)
            if data.get("done"):
                ollama_stream_meta = {
                    "load_duration_ms": round((data.get("load_duration") or 0) / 1e6, 1),
                    "prompt_eval_duration_ms": round(
                        (data.get("prompt_eval_duration") or 0) / 1e6, 1
                    ),
                    "eval_duration_ms": round((data.get("eval_duration") or 0) / 1e6, 1),
                    "eval_count": data.get("eval_count"),
                }
                break
    if timing is not None and hasattr(timing, "mark"):
        if "t_first_token" not in timing.monotonic:
            timing.mark("t_first_token")
        timing.mark("t_llm_response_end")
        if hasattr(timing, "set_ollama_meta"):
            timing.set_ollama_meta(ollama_stream_meta)
    return "".join(parts).strip()


def check_ollama_model(base_url: str, model: str, timeout: int = 5) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        names = {m.get("name", "") for m in data.get("models", [])}
        if model in names or f"{model}:latest" in names:
            return True, ""
        available = ", ".join(sorted(names)[:8])
        return False, f"모델 '{model}' 없음. `ollama pull {model}` 실행. (설치됨: {available}...)"
    except Exception as exc:
        return False, f"Ollama 연결 실패 ({base_url}): {exc}"


def call_openai_chat(model: str, system: str, user: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str(data["choices"][0]["message"]["content"]).strip()


def generate_extractive_answer(row: dict, retrieved: list[RetrievedChunk]) -> str:
    """Structured fallback when LLM is unavailable — formats retrieval hits into 1.2 sections."""
    category = str(row.get("category", ""))
    bullet_max = int(row.get("answer_bullets_max", 7))
    lines = [
        "## 1) 핵심 요약",
        "",
    ]
    for i, c in enumerate(retrieved[: min(bullet_max, 8)], start=1):
        title = c.file_name or c.doc_id
        snippet = c.text.replace("\n", " ").strip()[:220]
        lines.append(f"- [{c.source}] {title} (p{c.page_number}): {snippet}")
    lines.extend(
        [
            "",
            "## 2) 선박 운항/업무 영향",
            "",
            "- 검색된 회의/Rule 문서를 근거로 운항·보고·검사·설계 요건 변경 가능성을 검토해야 합니다.",
            "- 선급·국적국 규정과의 정합성 확인이 필요합니다.",
            "",
            "## 3) 추후 확인 필요사항",
            "",
            "- 본 답변은 검색 근거 자동 요약(extractive)이며, LLM 생성 답변이 아닙니다.",
            "- 미확정 안건·워킹그룹 진행 중 사항은 공식 MEPC/MSC report 및 선급 최신 Notice 확인 필요.",
            "",
            "## 4) 관련 선급 Rule / Guidance",
            "",
        ]
    )
    rule_chunks = [c for c in retrieved if c.source in {"DNV", "LR", "KR", "ABS"}]
    if rule_chunks:
        for c in rule_chunks[:3]:
            lines.append(f"- {c.source}: {c.file_name or c.doc_id} (p{c.page_number})")
    elif category == "rule_lookup":
        lines.append("- 검색 범위 내 Rule/Guidance 청크 참조 (상단 핵심 요약 bullet)")
    else:
        lines.append("- 본 검색은 IMO 회의 자료만 포함됨. 선급 Rule/Guidance는 별도 검색 필요.")
    return "\n".join(lines)


def iter_generate_answer(
    row: dict,
    retrieved: list[RetrievedChunk],
    *,
    model: str,
    ollama_base: str,
    temperature: float = 0.15,
    reference: dict | None = None,
):
    """Token iterator for Streamlit st.write_stream (keeps websocket alive)."""
    if not retrieved:
        return
    system = build_system_prompt(row)
    user = build_user_prompt(
        row, build_context_block(retrieved), reference=reference_for_question(row, reference), retrieved=retrieved
    )
    yield from call_ollama_chat_stream(
        model, system, user, ollama_base, temperature=temperature
    )


def _apply_profile_to_config(config: RetrievalRunConfig, profile) -> None:
    config.retrieval_profile_id = profile.profile_id
    config.retrieval_profile_label = profile.label_ko
    config.retrieval_profile_notes = list(profile.notes)


def _execute_retrieval_core(
    row: dict,
    collection,
    embed_model: str,
    chunks_dir: Path,
    *,
    eval_constrained_mode: bool = False,
    gold_doc_filter: bool | None = None,
    question_mode: str | None = None,
    top_k: int = 8,
    fetch_k: int = 40,
    use_diversity_rerank: bool = True,
    max_chunks_per_doc: int | None = None,
    max_chunks_per_page: int = 1,
    max_docs: int = 4,
    eval_k: int = 5,
    timing=None,
):
    from retrieval_chunk_quality import llm_context_target_k, refine_chunks_for_llm
    from retrieval_diversity import DiversityConfig, diversity_rerank
    from retrieval_question_profile import build_retrieval_profile
    from retrieval_verification import RetrievalRunConfig, resolve_retrieval_run_config
    from rag_retrieval_metrics import compute_retrieval_metrics
    from multi_doc_summary import assign_global_citations, discover_document_candidates

    question = str(row.get("question", ""))
    if row.get("category") and "_eval_category" not in row:
        row["_eval_category"] = str(row["category"])
    profile = build_retrieval_profile(
        question,
        row,
        ui_top_k=top_k,
        ui_fetch_k=fetch_k,
        ui_max_docs=max_docs,
        ui_max_chunks_per_doc=max_chunks_per_doc or 3,
        ui_max_chunks_per_page=max_chunks_per_page,
        ui_use_rerank=use_diversity_rerank,
        eval_constrained=eval_constrained_mode,
    )
    category = profile.question_category
    row["category"] = category
    from retrieval_query_analysis import detect_class_society_hint

    row["class_society_hint"] = detect_class_society_hint(question)

    from meeting_category_profile import uses_structured_meeting_answer

    legacy_row_cat = str(row.get("category") or category)
    structured_meeting = uses_structured_meeting_answer(row, legacy_category=legacy_row_cat)

    if profile.answer_mode == "meeting_outcome" and not eval_constrained_mode and not structured_meeting:
        pool = retrieve_for_question(
            collection,
            embed_model,
            row,
            top_k=profile.fetch_k,
            fetch_k=profile.fetch_k,
            chunks_dir=chunks_dir,
            gold_doc_filter=False,
            timing=timing,
        )
        if timing is not None and hasattr(timing, "mark"):
            timing.mark("t_context_build_start")
        cfg = DiversityConfig(
            max_chunks_per_doc=profile.max_chunks_per_doc,
            max_chunks_per_page=profile.max_chunks_per_page,
            max_docs=profile.max_docs,
        )
        selected = diversity_rerank(
            pool,
            top_k=profile.top_k,
            category=category,
            question_mode="broad",
            config=cfg,
        )
        llm_k = llm_context_target_k(row, max(profile.top_k, 12))
        retrieved = refine_chunks_for_llm(selected, pool, row=row, target_k=llm_k)
        if timing is not None and hasattr(timing, "mark"):
            timing.mark("t_context_build_end")
        config = RetrievalRunConfig(
            question_mode=profile.question_mode,
            question_category=category,
            broad_summary_mode=False,
            answer_mode="meeting_outcome",
            use_gold_filter=False,
            eval_constrained_mode=False,
            top_k=profile.top_k,
            fetch_k=profile.fetch_k,
            max_chunks_per_doc=profile.max_chunks_per_doc,
            max_chunks_per_page=profile.max_chunks_per_page,
            max_docs=profile.max_docs,
            use_diversity_rerank=profile.use_diversity_rerank,
        )
        _apply_profile_to_config(config, profile)
        eval_mode = "open"
        metrics = compute_retrieval_metrics(
            row, retrieved, top_k=config.top_k, eval_k=eval_k, eval_mode=eval_mode
        ).to_dict()
        return config, pool, retrieved, metrics, [], [], category

    if profile.broad_summary and not structured_meeting:
        pool = retrieve_for_question(
            collection,
            embed_model,
            row,
            top_k=profile.fetch_k,
            fetch_k=profile.fetch_k,
            chunks_dir=chunks_dir,
            gold_doc_filter=False,
            timing=timing,
        )
        if timing is not None and hasattr(timing, "mark"):
            timing.mark("t_context_build_start")
        doc_groups, pipe_warnings = discover_document_candidates(
            pool,
            top_docs=min(10, profile.max_docs),
            max_chunks_per_doc=profile.max_chunks_per_doc,
            min_unique_docs=3,
        )
        retrieved = assign_global_citations(doc_groups)
        if timing is not None and hasattr(timing, "mark"):
            timing.mark("t_context_build_end")
        config = RetrievalRunConfig(
            question_mode=profile.question_mode,
            question_category=category,
            broad_summary_mode=True,
            answer_mode="multi_doc_summary",
            use_gold_filter=False,
            eval_constrained_mode=False,
            top_k=profile.top_k,
            fetch_k=profile.fetch_k,
            max_chunks_per_doc=profile.max_chunks_per_doc,
            max_chunks_per_page=profile.max_chunks_per_page,
            max_docs=len(doc_groups),
            use_diversity_rerank=False,
        )
        _apply_profile_to_config(config, profile)
        eval_mode = "open"
        metrics = compute_retrieval_metrics(
            row, retrieved, top_k=len(retrieved), eval_k=eval_k, eval_mode=eval_mode
        ).to_dict()
        return config, pool, retrieved, metrics, doc_groups, pipe_warnings, category

    config = resolve_retrieval_run_config(
        row,
        eval_constrained_mode=eval_constrained_mode,
        gold_doc_filter=gold_doc_filter,
        question_mode=question_mode or profile.question_mode,
        top_k=profile.top_k,
        fetch_k=profile.fetch_k,
        max_chunks_per_doc=profile.max_chunks_per_doc,
        max_chunks_per_page=profile.max_chunks_per_page,
        max_docs=profile.max_docs,
        use_diversity_rerank=profile.use_diversity_rerank,
    )
    config.question_category = category
    config.broad_summary_mode = False
    config.answer_mode = profile.answer_mode
    if structured_meeting:
        config.answer_mode = "structured_meeting"
    _apply_profile_to_config(config, profile)

    pool = retrieve_for_question(
        collection,
        embed_model,
        row,
        top_k=config.fetch_k if config.use_diversity_rerank else config.top_k,
        fetch_k=config.fetch_k if config.use_diversity_rerank else None,
        chunks_dir=chunks_dir,
        gold_doc_filter=config.use_gold_filter,
        narrow_doc_id=config.narrow_doc_id,
        timing=timing,
    )
    if config.supplement_gold_pages:
        pool = supplement_gold_pages_for_llm(pool, row, chunks_dir)

    if category == "rule_lookup":
        from rule_lookup_answer import filter_pool_for_rule_lookup

        row["_pool_before_society_filter"] = list(pool)
        pool = filter_pool_for_rule_lookup(pool)

    if category == "rule_lookup" and row.get("class_society_hint"):
        from rag_society_filter import filter_pool_for_society, society_hard_filter_enabled

        pool, had = filter_pool_for_society(
            pool,
            str(row["class_society_hint"]),
            hard=society_hard_filter_enabled(row),
        )
        if society_hard_filter_enabled(row) and not had:
            row.setdefault("warning_flags", []).append("society_evidence_insufficient")

    if timing is not None and hasattr(timing, "mark"):
        timing.mark("t_context_build_start")

    if config.use_diversity_rerank:
        if timing is not None and hasattr(timing, "mark"):
            timing.mark("t_rerank_start")
        cfg = DiversityConfig(
            max_chunks_per_doc=config.max_chunks_per_doc,
            max_chunks_per_page=config.max_chunks_per_page,
            max_docs=config.max_docs,
        )
        selected = diversity_rerank(
            pool,
            top_k=config.top_k,
            category=category,
            question_mode=config.question_mode,
            config=cfg,
            class_society_hint=str(row.get("class_society_hint") or ""),
        )
        if timing is not None and hasattr(timing, "mark"):
            timing.mark("t_rerank_end")
    else:
        selected = pool[: config.top_k]

    llm_k = llm_context_target_k(row, config.top_k)
    retrieved = refine_chunks_for_llm(selected, pool, row=row, target_k=llm_k)
    if category == "rule_lookup":
        from rule_lookup_context import enrich_rule_lookup_chunks

        retrieved = enrich_rule_lookup_chunks(
            retrieved, pool, chunks_dir=chunks_dir, row=row
        )
    if timing is not None and hasattr(timing, "mark"):
        timing.mark("t_context_build_end")
    eval_mode = "constrained" if config.eval_constrained_mode else "open"
    metrics = compute_retrieval_metrics(
        row, retrieved, top_k=config.top_k, eval_k=eval_k, eval_mode=eval_mode
    ).to_dict()
    return config, pool, retrieved, metrics, [], [], category


def run_retrieval_only(
    row: dict,
    collection,
    embed_model: str,
    *,
    chunks_dir: Path,
    top_k: int = 8,
    fetch_k: int = 40,
    use_diversity_rerank: bool = True,
    max_chunks_per_doc: int | None = None,
    max_chunks_per_page: int = 1,
    max_docs: int = 4,
    eval_k: int = 5,
    eval_constrained_mode: bool = False,
    gold_doc_filter: bool | None = None,
    question_mode: str | None = None,
    timing=None,
) -> dict[str, Any]:
    """Retrieval + metrics + verification metadata (no LLM)."""
    from retrieval_verification import (
        build_evidence_table,
        build_retrieval_trace,
        build_verification_summary,
        compute_must_cover_coverage,
        get_must_cover_items,
        hybrid_score_lookup,
    )

    if timing is not None and hasattr(timing, "mark") and "t_retrieval_start" not in timing.monotonic:
        timing.mark("t_retrieval_start")

    config, pool, retrieved, metrics, doc_groups, pipe_warnings, category = _execute_retrieval_core(
        row,
        collection,
        embed_model,
        chunks_dir,
        eval_constrained_mode=eval_constrained_mode,
        gold_doc_filter=gold_doc_filter,
        question_mode=question_mode,
        top_k=top_k,
        fetch_k=fetch_k,
        use_diversity_rerank=use_diversity_rerank,
        max_chunks_per_doc=max_chunks_per_doc,
        max_chunks_per_page=max_chunks_per_page,
        max_docs=max_docs,
        eval_k=eval_k,
        timing=timing,
    )
    if timing is not None and hasattr(timing, "mark"):
        timing.mark("t_retrieval_end")
    from question_classifier import category_label_ko

    must_cover = get_must_cover_items(row)
    must_rows = compute_must_cover_coverage(must_cover, retrieved)
    summary = build_verification_summary(
        config=config,
        retrieved=retrieved,
        pool=pool,
        must_cover_rows=must_rows,
        metrics=metrics,
        row=row,
    )
    summary["question_category_label"] = category_label_ko(category)
    summary["final_doc_count"] = len(doc_groups) if doc_groups else summary.get("unique_doc_count", 0)
    summary["pipeline_warnings"] = pipe_warnings
    trace = build_retrieval_trace(
        row=row,
        config=config,
        pool=pool,
        retrieved=retrieved,
        metrics=metrics,
    )
    doc_groups_json = [
        {
            "doc_id": dg.doc_id,
            "file_name": dg.file_name,
            "source": dg.source,
            "meeting": dg.meeting,
            "pages": dg.pages,
            "citation_ids": dg.citation_ids,
            "chunk_ids": [c.chunk_id for c in dg.chunks],
        }
        for dg in doc_groups
    ]
    return {
        "question_id": row.get("question_id"),
        "category": category,
        "question": row.get("question"),
        "retrieved": retrieved,
        "retrieval_pool": pool,
        "retrieval_metrics": metrics,
        "retrieval_config": config.to_dict(),
        "answer_mode": config.answer_mode,
        "question_category": category,
        "question_category_label": category_label_ko(category),
        "broad_summary_mode": config.broad_summary_mode,
        "doc_groups": doc_groups_json,
        "pipeline_warnings": pipe_warnings,
        "evidence_table": build_evidence_table(
            retrieved,
            score_lookup=hybrid_score_lookup(row.get("_hybrid_retrieval_log")),
        ),
        "must_cover_coverage": must_rows,
        "verification_summary": summary,
        "trace": trace,
        "timing_metrics": timing.compute_metrics() if timing is not None and hasattr(timing, "compute_metrics") else None,
        "table_retrieval_debug": row.get("_table_retrieval_debug"),
        "pool_before_society_filter": row.get("_pool_before_society_filter"),
    }


def chunks_in_citation_order(
    pool: list[RetrievedChunk],
    doc_groups: list[dict] | None,
) -> list[RetrievedChunk]:
    """Rebuild [1]..[N] order used in multi-doc LLM context."""
    if not doc_groups or not pool:
        return pool
    by_id = {c.chunk_id: c for c in pool}
    ordered: list[RetrievedChunk] = []
    for dg in doc_groups:
        for chunk_id in dg.get("chunk_ids") or []:
            c = by_id.get(chunk_id)
            if c is not None:
                ordered.append(c)
    return ordered if ordered else pool


def build_answer_verification(
    row: dict,
    retrieved: list[RetrievedChunk],
    answer: str,
    *,
    config_dict: dict | None = None,
    pool: list[RetrievedChunk] | None = None,
    metrics: dict | None = None,
    doc_groups: list | None = None,
) -> dict[str, Any]:
    from retrieval_verification import (
        RetrievalRunConfig,
        build_answer_citation_mapping,
        build_evidence_table,
        build_retrieval_trace,
        build_verification_summary,
        compute_must_cover_coverage,
        get_must_cover_items,
        hybrid_score_lookup,
        resolve_retrieval_run_config,
    )

    if config_dict:
        config = RetrievalRunConfig(**config_dict)
    else:
        config = resolve_retrieval_run_config(row)
    must_rows = compute_must_cover_coverage(get_must_cover_items(row), retrieved, answer)
    cite_list = retrieved
    if config.answer_mode == "multi_doc_summary" and pool:
        cite_list = chunks_in_citation_order(pool, doc_groups) if doc_groups else pool
    summary = build_verification_summary(
        config=config,
        retrieved=retrieved,
        pool=pool or retrieved,
        answer=answer,
        must_cover_rows=must_rows,
        metrics=metrics,
        row=row,
    )
    if str(config.question_category or row.get("category") or "") == "rule_lookup":
        from rule_lookup_context import rule_lookup_answer_warnings

        summary.setdefault("warnings", [])
        summary["warnings"].extend(rule_lookup_answer_warnings(answer, cite_list))
        for note in row.get("_rule_lookup_repair_notes") or []:
            summary["warnings"].append(f"rule_lookup repair: {note}")
    return {
        "evidence_table": build_evidence_table(
            retrieved,
            answer,
            score_lookup=hybrid_score_lookup(row.get("_hybrid_retrieval_log")),
        ),
        "answer_citation_mapping": build_answer_citation_mapping(answer, cite_list),
        "must_cover_coverage": must_rows,
        "verification_summary": summary,
        "trace": build_retrieval_trace(
            row=row,
            config=config,
            pool=pool or retrieved,
            retrieved=retrieved,
            metrics=metrics or {},
            answer=answer,
        ),
    }


def generate_multi_document_answer(
    row: dict,
    pool: list[RetrievedChunk],
    *,
    category: str,
    doc_groups: list | None = None,
    retrieved: list[RetrievedChunk] | None = None,
    provider: str = "ollama",
    model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base: str = DEFAULT_OLLAMA_BASE,
    temperature: float = 0.15,
    multi_doc_strategy: str = "single_pass",
    max_llm_docs: int = 6,
    timing=None,
    on_token=None,
) -> tuple[str, list[RetrievedChunk], list[str]]:
    import sys
    from multi_doc_summary import DocGroup, run_multi_doc_summary

    llm_started = False
    stream_synthesis = {"allow": False}

    def call_llm(system: str, user: str, num_predict: int = 2000) -> str:
        nonlocal llm_started
        llm_started = True
        use_timing = timing if stream_synthesis["allow"] else None
        token_cb = on_token if stream_synthesis["allow"] and on_token else None
        return call_ollama_chat_timed(
            model,
            system,
            user,
            ollama_base,
            temperature=min(temperature, 0.1),
            num_predict=num_predict,
            num_ctx=16384,
            timeout=900,
            timing=use_timing,
            on_token=token_cb,
        )

    def on_progress(step: str) -> None:
        stream_synthesis["allow"] = "synthesis" in step.lower()
        sys.stderr.write(f"MULTI_DOC_PROGRESS: {step}\n")
        sys.stderr.flush()

    chunk_by_id: dict[str, RetrievedChunk] = {}
    for c in pool + (retrieved or []):
        chunk_by_id[c.chunk_id] = c

    precomputed = None
    if doc_groups:
        precomputed = []
        for dg in doc_groups:
            if isinstance(dg, DocGroup):
                precomputed.append(dg)
                continue
            chunks = [chunk_by_id[cid] for cid in dg.get("chunk_ids", []) if cid in chunk_by_id]
            if not chunks:
                chunks = [
                    chunk_by_id[c.chunk_id]
                    for c in (retrieved or [])
                    if c.doc_id == dg.get("doc_id")
                ][:3]
            if not chunks:
                continue
            precomputed.append(
                DocGroup(
                    doc_id=str(dg.get("doc_id", "")),
                    file_name=str(dg.get("file_name", "")),
                    source=str(dg.get("source", "")),
                    meeting=dg.get("meeting"),
                    chunks=chunks,
                    citation_ids=list(dg.get("citation_ids") or []),
                )
            )

    result = run_multi_doc_summary(
        row,
        pool,
        category=category,
        call_llm=call_llm,
        doc_groups=precomputed,
        strategy=multi_doc_strategy,
        max_llm_docs=max_llm_docs,
        on_progress=on_progress,
    )
    return result.answer, result.retrieved, result.warnings


def generate_answer(
    row: dict,
    retrieved: list[RetrievedChunk],
    *,
    provider: str,
    model: str,
    ollama_base: str,
    allow_extractive_fallback: bool = True,
    stream: bool = False,
    temperature: float = 0.15,
    reference: dict | None = None,
    answer_mode: str = "standard_rag",
    pool: list[RetrievedChunk] | None = None,
    category: str | None = None,
    doc_groups: list | None = None,
    multi_doc_strategy: str = "single_pass",
    max_llm_docs: int = 6,
    timing=None,
    on_token=None,
) -> tuple[str, str, str]:
    debug = row.get("_table_retrieval_debug")
    if debug and row.get("_table_qa"):
        from table_schema_retrieval import apply_confidence_gate

        if not debug.get("passes_confidence_gate", True):
            return apply_confidence_gate("", debug), "confidence_gate", "none"

    if not retrieved and not pool:
        return "", provider, model
    if answer_mode == "multi_doc_summary" and pool:
        try:
            answer, _, _ = generate_multi_document_answer(
                row,
                pool,
                category=category or str(row.get("category", "trend_summary")),
                doc_groups=doc_groups,
                retrieved=retrieved,
                provider=provider,
                model=model,
                ollama_base=ollama_base,
                temperature=temperature,
                multi_doc_strategy=multi_doc_strategy,
                max_llm_docs=max_llm_docs,
                timing=timing,
                on_token=on_token,
            )
            return answer, provider, model
        except Exception:
            if not allow_extractive_fallback:
                raise
            return generate_extractive_answer(row, retrieved or pool[:10]), "extractive", "retrieval-only"
    if not retrieved:
        return "", provider, model
    cat = str(category or row.get("category") or "").strip()
    if cat not in CATEGORY_GUIDANCE:
        from question_classifier import classify_question_category

        cat = classify_question_category(str(row.get("question", "")), row)
    if cat == "rule_lookup":
        from rule_lookup_retrieval_log import save_rule_lookup_run_log
        from rule_lookup_structured_answer import build_rule_lookup_structured_answer

        warnings = list(row.get("warning_flags") or [])
        answer, ans_warnings = build_rule_lookup_structured_answer(
            retrieved,
            question=str(row.get("question") or ""),
            pool=pool,
            warning_flags=warnings,
        )
        row["warning_flags"] = list(dict.fromkeys(warnings + ans_warnings))
        row["_answer_generation"] = {
            "answer_source": "structured_template",
            "llm_used": False,
            "llm_call_function": None,
            "llm_prompt_chars": 0,
            "llm_context_chunks": len(retrieved),
            "llm_output_chars": len(answer or ""),
            "llm_grounded_check_pass": True,
            "fallback_reason": None,
        }
        log = row.get("_hybrid_retrieval_log") or {}
        try:
            save_rule_lookup_run_log(
                question=str(row.get("question") or ""),
                row=row,
                category=cat,
                dense_results=log.get("dense_results") or [],
                bm25_results=log.get("bm25_results") or [],
                fused_results=log.get("fused_results") or [],
                retrieved=retrieved,
                answer=answer,
                warning_flags=row["warning_flags"],
            )
        except Exception:
            pass
        return answer, "rule_guidance_lookup", "none"

    legacy_cat = str(row.get("_eval_category") or row.get("category") or cat)
    from meeting_category_profile import build_meeting_retrieval_profile, uses_structured_meeting_answer

    if uses_structured_meeting_answer(row, legacy_category=legacy_cat):
        from meeting_structured_answer import build_meeting_structured_answer

        mprofile = build_meeting_retrieval_profile(
            str(row.get("question") or ""),
            row,
            legacy_category=legacy_cat,
        )
        warnings = list(row.get("warning_flags") or [])
        ctx = (pool or retrieved)[:40]
        answer, ans_warnings, meta = build_meeting_structured_answer(
            ctx,
            question=str(row.get("question") or ""),
            row=row,
            profile=mprofile,
            warning_flags=warnings,
        )
        row["warning_flags"] = list(dict.fromkeys(warnings + ans_warnings))
        row["_meeting_answer_meta"] = meta
        row["_top_level_category"] = mprofile.top_level_category
        row["_internal_intent"] = mprofile.internal_intent
        return answer, "structured_meeting", "none"

    llm_temp = min(temperature, 0.1) if cat == "rule_lookup" else temperature
    system = build_system_prompt(row)
    user = build_user_prompt(
        row, build_context_block(retrieved), reference=reference_for_question(row, reference), retrieved=retrieved
    )
    try:
        if provider == "openai":
            answer = call_openai_chat(model, system, user)
        elif stream:
            answer = "".join(
                call_ollama_chat_stream(model, system, user, ollama_base, temperature=llm_temp)
            )
        else:
            answer = call_ollama_chat_timed(
                model,
                system,
                user,
                ollama_base,
                temperature=llm_temp,
                num_ctx=16384,
                timing=timing,
                on_token=on_token,
            )
        if cat == "rule_lookup":
            from rule_lookup_answer import finalize_rule_lookup_answer

            answer, repair_notes = finalize_rule_lookup_answer(answer, retrieved)
            row["_rule_lookup_repair_notes"] = repair_notes
        return answer, provider, model
    except Exception:
        if allow_extractive_fallback:
            return generate_extractive_answer(row, retrieved), "extractive", "retrieval-only"
        raise


def load_reference_answers(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out[str(row["question_id"])] = row
    return out


def run_rag_pipeline(
    row: dict,
    collection,
    embed_model: str,
    *,
    chunks_dir: Path,
    top_k: int = 8,
    fetch_k: int = 40,
    use_diversity_rerank: bool = True,
    max_chunks_per_doc: int | None = None,
    max_chunks_per_page: int = 1,
    max_docs: int = 4,
    eval_constrained_mode: bool = False,
    gold_doc_filter: bool | None = None,
    provider: str = "ollama",
    model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base: str = DEFAULT_OLLAMA_BASE,
    skip_llm: bool = False,
    stream: bool = False,
    temperature: float = 0.15,
    trace_log_path: Path | None = None,
) -> dict[str, Any]:
    from retrieval_verification import append_retrieval_trace_log

    config, pool, retrieved, metrics, doc_groups, pipe_warnings, category = _execute_retrieval_core(
        row,
        collection,
        embed_model,
        chunks_dir,
        eval_constrained_mode=eval_constrained_mode,
        gold_doc_filter=gold_doc_filter,
        top_k=top_k,
        fetch_k=fetch_k,
        use_diversity_rerank=use_diversity_rerank,
        max_chunks_per_doc=max_chunks_per_doc,
        max_chunks_per_page=max_chunks_per_page,
        max_docs=max_docs,
    )
    answer = ""
    llm_provider = provider
    llm_model = model
    error = ""
    verification: dict[str, Any] = {}
    if not skip_llm:
        try:
            answer, llm_provider, llm_model = generate_answer(
                row,
                retrieved,
                provider=provider,
                model=model,
                ollama_base=ollama_base,
                stream=stream,
                temperature=temperature,
                answer_mode=config.answer_mode,
                pool=pool,
                category=category,
                doc_groups=doc_groups,
            )
        except Exception as exc:
            error = str(exc)
    verification = build_answer_verification(
        row,
        retrieved,
        answer,
        config_dict=config.to_dict(),
        pool=pool,
        metrics=metrics,
    )
    if trace_log_path and (answer or skip_llm):
        entry = verification.get("trace", {})
        entry["llm_provider"] = llm_provider
        entry["llm_model"] = llm_model
        entry["error"] = error
        append_retrieval_trace_log(trace_log_path, entry)

    return {
        "question_id": row.get("question_id"),
        "category": row.get("category"),
        "question": row.get("question"),
        "retrieved": retrieved,
        "retrieval_pool": pool,
        "retrieval_metrics": metrics,
        "retrieval_config": config.to_dict(),
        "answer": answer,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "error": error,
        **verification,
    }


def format_result_markdown(result: ValidationResult, *, variant: str | None = None) -> str:
    label = variant or result.retrieval_variant or "baseline"
    m = result.retrieval_metrics or {}
    lines = [
        f"# {result.question_id} [{result.category}] ({label})",
        "",
        f"**질문:** {result.question}",
        "",
        "## Retrieval metrics",
        f"- source_hit@5: {'YES' if m.get('source_hit_at_5') else 'NO'}",
        f"- gold_doc_hit@5: {'YES' if m.get('gold_doc_hit_at_5') else 'NO'}"
        + (f" (rank {m.get('gold_doc_rank')})" if m.get("gold_doc_rank") else ""),
        f"- doc_recall@5 (open): {'YES' if m.get('doc_recall_at_5') else 'NO'}",
        f"- page_recall@5: {'YES' if m.get('page_recall_at_5') else 'NO'}",
        f"- unique_docs@k: {m.get('unique_doc_count', 0)}",
        f"- gold_page_set_hit@5: {'YES' if m.get('gold_page_set_hit_at_5') else 'NO'}"
        + (f" (pages {m.get('gold_pages')})" if m.get("gold_pages") else ""),
        f"- topic_hit@k: {'YES' if m.get('topic_hit_at_k') else 'NO'}",
        f"- keyword_coverage: {m.get('keyword_coverage', 0):.1%}",
        f"- boundary_error_rate: {m.get('boundary_error_rate', 0):.1%}",
        f"- duplicate_doc_ratio: {m.get('duplicate_doc_ratio', 0):.1%}",
        f"- duplicate_page_ratio: {m.get('duplicate_page_ratio', 0):.1%}",
        "",
        "## Retrieval (legacy)",
        f"- keyword hits: {result.retrieval_keyword_hits}/{result.retrieval_keyword_total}",
        f"- source hits in top-k: {result.retrieval_source_hits}",
        f"- hit@k: {'YES' if result.retrieval_hit_at_k else 'NO'}",
        "",
        "### Top chunks",
    ]
    for i, c in enumerate(result.retrieved[:8], start=1):
        kw = ", ".join(c.matched_keywords) if c.matched_keywords else "-"
        topics = ", ".join(c.matched_topics) if c.matched_topics else "-"
        lines.extend(
            [
                f"#### {i}. `{c.file_name or c.doc_id}`",
                f"- source={c.source} | p{c.page_number} | dist={c.distance:.4f} | chunk={c.chunk_id}",
                f"- matched_keywords: {kw}",
                f"- matched_topics: {topics}",
                f"- preview: {c.content_preview or _preview(c.text)}",
                "",
            ]
        )
    lines.extend(["## Generated Answer", ""])
    if result.error:
        lines.append(f"*(generation error: {result.error})*")
    lines.append(result.answer or "*(no answer)*")
    return "\n".join(lines)


def _preview(text: str, limit: int = 500) -> str:
    from rag_retrieval_metrics import content_preview

    return content_preview(text, limit)
