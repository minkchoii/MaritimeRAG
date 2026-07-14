"""Meeting outcome question detection, query expansion, and two-stage retrieval."""
from __future__ import annotations

import re
from typing import Any

from embedding_policy import embed_texts_local
from imo_doc_classify import (
    asks_broad_session_outcome,
    classify_imo_filename,
    meeting_outcome_scope,
)
from imo_doc_registry import DEFAULT_CORPUS, load_corpus_rows, priority_doc_ids
from meeting_summary_context import (
    TARGET_SCOPE_WHOLE_SESSION,
    meeting_summary_source_tier,
    resolve_meeting_summary_context,
)
from retrieval_query_analysis import (
    IMO_SESSION_RE,
    QuerySignals,
    analyze_query,
    is_meeting_outcome_question,
    topic_agenda_prefixes,
)
from retrieval_search import _merge_where, safe_chroma_query

MEETING_SESSION_RE = IMO_SESSION_RE

OUTCOME_INTENT_PATTERNS = (
    r"주요\s*결과",
    r"\b결과\b",
    r"\boutcome\b",
    r"\bsummary\b",
    r"key\s*outcomes?",
    r"\badopted\b",
    r"\bapproved\b",
    r"\bdecision\b",
    r"결정\s*사항",
    r"채택",
    r"승인",
    r"요약해",
    r"정리해",
)

COMPARISON_PATTERNS = (
    r"비교",
    r"compare",
    r"versus",
    r"\bvs\.?\b",
    r"차이",
    r"대비",
)

MEETING_BOOST_KEYWORDS = (
    "summary report",
    "key outcomes",
    "key outcome",
    "outcome",
    "adopted",
    "approved",
    "maritime safety committee",
    "mass code",
    "igc code",
    "ammonia",
)

OUTCOME_CHUNK_TERMS = (
    "outcome",
    "adopted",
    "approved",
    "resolution",
    "decision",
    "key outcomes",
    "summary report",
    "executive summary",
)

MEETING_DOC_BOOST = {
    "session_report": 0.16,
    "session_outcome": 0.10,
    "reference_outcome": 0.0,
    "working_group_report": 0.08,
    "agenda_report": 0.06,
}

SESSION_FINAL_REPORT_BOOST = 0.36
REFERENCE_BODY_OUTCOME_PENALTY = 0.32
MEETING_SUMMARY_REFERENCE_PENALTY = 0.45
MEETING_SUMMARY_AGENDA_PENALTY = 0.28
BROAD_WG_REPORT_PENALTY = 0.14

MEETING_KEYWORD_BOOST = 0.05
MEETING_OUTCOME_CHUNK_BOOST = 0.15
OTHER_SESSION_PENALTY = 0.22
OLD_SESSION_PENALTY = 0.18


def detect_meeting_outcome_question(question: str, row: dict | None = None) -> bool:
    return is_meeting_outcome_question(question, row)


def is_comparison_question(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q, re.I) for p in COMPARISON_PATTERNS)


def parse_outcome_item_count(question: str, row: dict | None = None) -> int:
    if row and row.get("outcome_item_count"):
        try:
            return max(1, int(row["outcome_item_count"]))
        except (TypeError, ValueError):
            pass
    m = re.search(r"(\d+)\s*개\s*(?:항목|개)", question)
    if m:
        return max(1, int(m.group(1)))
    m = re.search(r"(\d+)\s*(?:items?|points?|bullets?)", question, re.I)
    if m:
        return max(1, int(m.group(1)))
    return 3


def _session_label(body: str, num: int) -> str:
    return f"{body.upper()} {num}"


def _committee_long_name(body: str) -> str:
    if body.upper() == "MSC":
        return "Maritime Safety Committee"
    if body.upper() == "MEPC":
        return "Marine Environment Protection Committee"
    return body.upper()


def expand_meeting_outcome_queries(question: str, signals: QuerySignals | None = None) -> list[str]:
    """Expand user question into internal retrieval queries."""
    sig = signals or analyze_query(question)
    expansions: list[str] = [question.strip()]
    lower = question.lower()

    for body, num in sig.session_codes:
        label = _session_label(body, num)
        long_name = _committee_long_name(body)
        expansions.extend(
            [
                f"{label} draft report maritime safety committee session key outcomes",
                f"{label} summary report key outcomes",
                f"{long_name} {num}th session report adopted approved",
                f"IMO {label} key decisions",
            ]
        )
        if body == "MSC" and num == 111:
            expansions.extend(
                [
                    "MSC 111 WP.1 Draft Report Maritime Safety Committee 111th session",
                    "MSC 111 report of the Maritime Safety Committee on its 111th session",
                ]
            )
        if asks_broad_session_outcome(question, topics=sig.topics):
            expansions.extend(
                [
                    f"{long_name} draft report on its {num}th session",
                    f"{label} session report resolutions decisions adopted",
                ]
            )
        elif "mass" in sig.topics or "mass code" in lower:
            expansions.append(f"{label} MASS Code adopted mandatory")
        if "igc" in lower or "igc code" in lower:
            expansions.append(f"{label} IGC Code amendments adopted")
        if "ammonia" in lower or "암모니아" in question:
            expansions.append(f"{label} ammonia fuel ship guidelines adopted")

    seen: set[str] = set()
    out: list[str] = []
    for term in expansions:
        key = term.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(term.strip())
    return out


def enrich_meeting_outcome_query(question: str, model_name: str) -> str:
    """Embedding query with meeting-outcome expansions."""
    from retrieval_search import _enrich_query_standard

    signals = analyze_query(question)
    if not signals.meeting_outcome_question:
        return _enrich_query_standard(question)
    parts = expand_meeting_outcome_queries(question, signals)[:8]
    base = _enrich_query_standard(question)
    return f"{' '.join(parts)} {base}".strip()


def meeting_outcome_metadata_adjustment(
    *,
    meta: dict,
    document: str,
    signals: QuerySignals,
    question: str = "",
    is_comparison: bool = False,
) -> tuple[float, float]:
    """Return (boost, penalty) for meeting-outcome ranking."""
    if not signals.meeting_outcome_question:
        return 0.0, 0.0

    boost = 0.0
    penalty = 0.0
    fname = str(meta.get("file_name") or meta.get("doc_id") or "").lower()
    file_name = str(meta.get("file_name") or "")
    doc_head = (document or "")[:1200].lower()
    combined = f"{fname} {doc_head}"
    doc_type = classify_imo_filename(file_name)
    scope = meeting_outcome_scope(file_name)
    broad_session = asks_broad_session_outcome(question, topics=tuple(signals.topics))
    summary_ctx = resolve_meeting_summary_context(question)
    summary_intent = summary_ctx.target_scope == TARGET_SCOPE_WHOLE_SESSION

    if scope == "session_final_report":
        boost += SESSION_FINAL_REPORT_BOOST
    elif doc_type in MEETING_DOC_BOOST:
        boost += MEETING_DOC_BOOST[doc_type]

    if summary_ctx.apply_session_final_priority:
        tier = meeting_summary_source_tier(file_name, ctx=summary_ctx)
        if tier == 0:
            boost += 0.20
        elif tier == 1:
            boost += 0.14
        elif tier >= 3:
            penalty += MEETING_SUMMARY_REFERENCE_PENALTY
        if summary_ctx.apply_reference_penalties and (
            "strategic plan" in fname or "fal 50" in fname or "fal.50" in fname
        ):
            penalty += 0.25
    elif summary_ctx.preferred_doc_hints:
        if any(h in fname for h in summary_ctx.preferred_doc_hints):
            boost += 0.18

    if broad_session:
        if scope == "reference_body_outcome":
            penalty += REFERENCE_BODY_OUTCOME_PENALTY
            if summary_ctx.apply_reference_penalties:
                penalty += MEETING_SUMMARY_REFERENCE_PENALTY
        elif scope == "working_group_report" and not signals.topics:
            penalty += BROAD_WG_REPORT_PENALTY
        elif scope == "session_final_report":
            boost += 0.10
        elif summary_ctx.apply_reference_penalties and re.search(
            r"\b(?:msc|mepc)\s*\d{1,3}[-/]2(?:[-/]|$)", fname
        ):
            penalty += MEETING_SUMMARY_AGENDA_PENALTY
    elif scope == "reference_body_outcome" and doc_type == "reference_outcome":
        boost += 0.04

    if not broad_session and ("summary report" in combined or "key outcomes" in combined):
        boost += 0.10

    for body, num in signals.session_codes:
        label = f"{body.lower()} {num}"
        dash_label = f"{body.lower()}-{num}"
        slash_label = f"{body.lower()}/{num}"
        if label in fname or dash_label in fname or slash_label in fname:
            boost += 0.12 if broad_session and scope == "session_final_report" else 0.10
        for prefix in topic_agenda_prefixes(signals):
            if prefix in fname:
                boost += 0.14
        if not broad_session:
            for kw in MEETING_BOOST_KEYWORDS:
                if kw in combined:
                    boost += MEETING_KEYWORD_BOOST

    if broad_session and scope == "session_final_report":
        for term in ("resolution", "decision", "adopted", "approved", "key outcomes"):
            if term in doc_head:
                boost += 0.03

    if is_comparison:
        return boost, penalty

    for body, num in signals.session_codes:
        target = body.upper()
        label = f"{body.lower()} {num}"
        if target == "MSC":
            if re.search(r"\bmepc\s*\d{1,3}\b", fname) and label not in fname:
                penalty += OTHER_SESSION_PENALTY
        elif target == "MEPC":
            if re.search(r"\bmsc\s*\d{1,3}\b", fname) and label not in fname:
                penalty += OTHER_SESSION_PENALTY

        for m in re.finditer(r"\b(msc|mepc)\s*(\d{1,3})\b", fname):
            other_body, other_num = m.group(1).upper(), int(m.group(2))
            if other_body == target and other_num < num and f"{target.lower()} {num}" not in fname:
                penalty += OLD_SESSION_PENALTY

        if re.search(r"\btc\s*\d{1,3}\b", fname) and label not in fname and "outcome" not in fname:
            penalty += OTHER_SESSION_PENALTY * 0.5

    return boost, penalty


def _file_matches_session(file_name: str, signals: QuerySignals) -> bool:
    fn = (file_name or "").lower()
    for body, num in signals.session_codes:
        if f"{body.lower()} {num}" in fn or f"{body.lower()}-{num}" in fn:
            return True
    return False


def meeting_priority_doc_ids(signals: QuerySignals, *, question: str = "", limit: int = 20) -> list[str]:
    """Doc_ids for session outcome/summary/report documents."""
    if not signals.session_codes:
        return []

    broad = asks_broad_session_outcome(question, topics=tuple(signals.topics))
    agenda_items: tuple[int, ...] | None = None
    if "mass" in signals.topics:
        agenda_items = (5,)
    elif "igc" in signals.topics:
        agenda_items = (14,)
    elif "alt_fuel" in signals.topics:
        agenda_items = (12,)

    if broad:
        preferred: tuple[str, ...] | None = ("session_report",)
    elif agenda_items:
        preferred = None
    else:
        preferred = (
            "session_report",
            "session_outcome",
            "working_group_report",
            "agenda_report",
        )

    rows = load_corpus_rows(str(DEFAULT_CORPUS))
    ids: list[str] = []
    if broad:
        for row in rows:
            fn = str(row.get("file_name", ""))
            doc_id = str(row.get("doc_id", ""))
            if doc_id and meeting_outcome_scope(fn) == "session_final_report" and _file_matches_session(fn, signals):
                ids.append(doc_id)
        ids = list(dict.fromkeys(ids))

    ids.extend(
        d
        for d in priority_doc_ids(signals, preferred_types=preferred, agenda_items=agenda_items, limit=limit)
        if d not in ids
    )
    if agenda_items and len(ids) < limit // 2:
        for d in priority_doc_ids(signals, preferred_types=preferred, limit=limit):
            if d not in ids:
                ids.append(d)

    extra: list[tuple[int, str]] = []
    for row in rows:
        file_name = str(row.get("file_name", ""))
        file_lower = file_name.lower()
        doc_id = str(row.get("doc_id", ""))
        if not doc_id or not _file_matches_session(file_name, signals):
            continue
        scope = meeting_outcome_scope(file_name)
        score = 0
        if broad:
            if scope == "session_final_report":
                score += 250
            elif scope == "reference_body_outcome":
                score += 10
            elif scope == "working_group_report":
                score += 20
            elif "report of the" in file_lower:
                score += 45
            else:
                score += 25
        else:
            if scope == "reference_body_outcome":
                score += 90
            elif "summary report" in file_lower:
                score += 120
            elif "report of the" in file_lower:
                score += 80
            elif "outcome" in file_lower:
                score += 70
            else:
                score += 40
        if score:
            extra.append((score, doc_id))
    extra.sort(key=lambda x: (-x[0], x[1]))
    seen = set(ids)
    for _, doc_id in extra:
        if doc_id not in seen:
            seen.add(doc_id)
            ids.append(doc_id)
        if len(ids) >= limit:
            break
    return ids[:limit]


def _is_outcome_like_chunk(
    meta: dict,
    document: str,
    *,
    question: str = "",
    broad_session: bool | None = None,
) -> bool:
    file_name = str(meta.get("file_name") or "")
    fname = file_name.lower()
    text = (document or "")[:800].lower()
    combined = f"{fname} {text}"
    scope = meeting_outcome_scope(file_name)
    if broad_session is None:
        broad_session = asks_broad_session_outcome(question)
    if broad_session and scope == "reference_body_outcome":
        return False
    if scope == "session_final_report":
        return True
    if classify_imo_filename(file_name) in {"session_report"} and broad_session:
        return True
    if not broad_session and classify_imo_filename(file_name) in {"session_outcome", "session_report"}:
        return True
    return any(term in combined for term in OUTCOME_CHUNK_TERMS)


def query_meeting_outcome_chunks(
    collection,
    question: str,
    model_name: str,
    signals: QuerySignals,
    *,
    top_k: int = 10,
    doc_id: str | None = None,
    source: str | None = None,
    timing=None,
) -> list[tuple[str, float, dict, str]]:
    """Two-stage: meeting docs first, then outcome/adopted chunks within them."""
    if not signals.meeting_outcome_question:
        return []

    base_where = _merge_where(
        {"source": source.upper()} if source else None,
        {"doc_id": doc_id} if doc_id else None,
    )
    priority_ids = meeting_priority_doc_ids(signals, question=question)
    expansions = expand_meeting_outcome_queries(question, signals)

    pool: dict[str, tuple[float, dict, str]] = {}

    def absorb(raw: dict, boost: float = 0.0) -> None:
        if not raw.get("ids") or not raw["ids"][0]:
            return
        for cid, dist, meta, doc in zip(
            raw["ids"][0],
            raw["distances"][0],
            raw["metadatas"][0],
            raw["documents"][0],
        ):
            adj = float(dist) - boost
            prev = pool.get(cid)
            if prev is None or adj < prev[0]:
                pool[cid] = (adj, meta or {}, doc or "")

    # Stage 1 — meeting summary/outcome documents
    embed_main = enrich_meeting_outcome_query(question, model_name)
    vector_main = embed_texts_local([embed_main], model_name, for_query=True, timing=timing)[0]

    # For topic-specific questions, run a focused sub-query first.
    if signals.topics:
        topic_terms: list[str] = []
        if "mass" in signals.topics:
            topic_terms.append("MASS Code adopted mandatory goal-based")
        if "igc" in signals.topics:
            topic_terms.append("IGC Code amendments adopted consolidated draft")
        if "alt_fuel" in signals.topics or "ammonia" in question.lower() or "암모니아" in question:
            topic_terms.append("ammonia fuel ship safety alternative fuel guidelines adopted")
        if topic_terms:
            topic_vec = embed_texts_local(
                [" ".join(topic_terms)], model_name, for_query=True, timing=timing
            )[0]
            topic_ids = meeting_priority_doc_ids(signals, question=question, limit=12)
            if topic_ids:
                try:
                    absorb(
                        safe_chroma_query(
                            collection,
                            query_embeddings=[topic_vec],
                            n_results=min(top_k * 3, 30),
                            where=_merge_where(base_where, {"doc_id": {"$in": topic_ids[:12]}}),
                        ),
                        boost=MEETING_OUTCOME_CHUNK_BOOST + 0.08,
                    )
                except Exception:
                    pass

    try:
        absorb(
            safe_chroma_query(
                collection,
                query_embeddings=[vector_main],
                n_results=min(top_k * 4, 40),
                where=base_where,
            ),
            boost=0.0,
        )
    except Exception:
        pass

    if priority_ids:
        batch = priority_ids[:16]
        try:
            absorb(
                safe_chroma_query(
                    collection,
                    query_embeddings=[vector_main],
                    n_results=min(top_k * 3, 36),
                    where=_merge_where(base_where, {"doc_id": {"$in": batch}}),
                ),
                boost=MEETING_OUTCOME_CHUNK_BOOST,
            )
        except Exception:
            for pid in batch[:10]:
                try:
                    absorb(
                        safe_chroma_query(
                            collection,
                            query_embeddings=[vector_main],
                            n_results=8,
                            where=_merge_where(base_where, {"doc_id": pid}),
                        ),
                        boost=MEETING_OUTCOME_CHUNK_BOOST,
                    )
                except Exception:
                    pass

    # Stage 2 — outcome/adopted/approved focused sub-query within meeting docs
    outcome_query = " ".join(
        list(OUTCOME_CHUNK_TERMS[:6])
        + [_session_label(b, n) for b, n in signals.session_codes[:2]]
    )
    vector_outcome = embed_texts_local([outcome_query], model_name, for_query=True, timing=timing)[0]
    stage2_where = base_where
    if priority_ids:
        stage2_where = _merge_where(base_where, {"doc_id": {"$in": priority_ids[:12]}})
    try:
        absorb(
            safe_chroma_query(
                collection,
                query_embeddings=[vector_outcome],
                n_results=min(top_k * 3, 30),
                where=stage2_where,
            ),
            boost=MEETING_OUTCOME_CHUNK_BOOST + 0.04,
        )
    except Exception:
        pass

    # Additional expansion queries (LR/DNV summary reports etc.)
    for exp in expansions[1:4]:
        try:
            vec = embed_texts_local([exp], model_name, for_query=True, timing=timing)[0]
            absorb(
                safe_chroma_query(
                    collection,
                    query_embeddings=[vec],
                    n_results=min(top_k * 2, 20),
                    where=base_where,
                ),
                boost=0.06,
            )
        except Exception:
            pass

    ranked = sorted(pool.items(), key=lambda x: x[1][0])[: top_k * 2]
    return [(cid, score, meta, doc) for cid, (score, meta, doc) in ranked]


def merge_meeting_outcome_into_raw(
    baseline_raw: dict,
    meeting_hits: list[tuple[str, float, dict, str]],
    *,
    top_k: int | None = None,
    min_outcome_chunks: int = 2,
    topic_specific: bool = False,
    question: str = "",
) -> dict:
    """Merge meeting-outcome hits into baseline pool via score boost."""
    broad_session = asks_broad_session_outcome(question) and not topic_specific
    pool: dict[str, tuple[float, dict, str]] = {}
    for cid, dist, meta, doc in zip(
        baseline_raw.get("ids", [[]])[0],
        baseline_raw.get("distances", [[]])[0],
        baseline_raw.get("metadatas", [[]])[0],
        baseline_raw.get("documents", [[]])[0],
    ):
        pool[cid] = (float(dist), meta or {}, doc or "")

    for cid, dist, meta, doc in meeting_hits:
        extra_boost = 0.08 if _is_outcome_like_chunk(meta, doc, question=question, broad_session=broad_session) else 0.04
        adj = float(dist) - extra_boost
        prev = pool.get(cid)
        if prev is None or adj < prev[0]:
            pool[cid] = (adj, meta or {}, doc or "")

    ranked = sorted(pool.items(), key=lambda x: x[1][0])
    if top_k is not None:
        min_required = 0 if topic_specific else min_outcome_chunks
        outcome_ids = [
            cid
            for cid, (_, meta, doc) in ranked
            if _is_outcome_like_chunk(meta, doc, question=question, broad_session=broad_session)
        ]
        if outcome_ids and min_required > 0:
            top_ids = {cid for cid, _ in ranked[:top_k]}
            missing = [cid for cid in outcome_ids if cid not in top_ids][:min_required]
            if missing:
                keep = ranked[: max(top_k - len(missing), 1)]
                tail = [(cid, pool[cid]) for cid in missing if cid in pool]
                merged_ids = {cid for cid, _ in keep}
                for cid, item in ranked:
                    if cid in merged_ids:
                        continue
                    if len(keep) + len(tail) >= top_k:
                        break
                    if cid in missing:
                        tail.append((cid, item))
                ranked = keep + [x for x in tail if x[0] not in merged_ids]
                ranked = sorted(ranked, key=lambda x: x[1][0])[:top_k]
            else:
                ranked = ranked[:top_k]
        else:
            ranked = ranked[:top_k]

    return {
        "ids": [[cid for cid, _ in ranked]],
        "distances": [[score for _, (score, _, _) in ranked]],
        "metadatas": [[meta for _, (_, meta, _) in ranked]],
        "documents": [[doc for _, (_, _, doc) in ranked]],
        "meeting_outcome_aware": True,
    }


def meeting_doc_recall_at_k(
    retrieved: list[Any],
    row: dict,
    k: int,
) -> bool:
    """True if a meeting summary/outcome doc for the target session is in top-k."""
    gold_doc = str(row.get("gold_doc_id") or "")
    gold_docs = row.get("gold_doc_ids") or []
    if isinstance(gold_docs, str):
        gold_docs = [gold_docs]
    targets = {gold_doc} if gold_doc else set()
    targets.update(str(d) for d in gold_docs if d)

    signals = analyze_query(str(row.get("question", "")))
    for chunk in retrieved[:k]:
        if chunk.doc_id in targets:
            return True
        fname = (chunk.file_name or "").lower()
        doc_type = classify_imo_filename(chunk.file_name or "")
        if doc_type not in {"session_outcome", "session_report"}:
            continue
        for body, num in signals.session_codes:
            if f"{body.lower()} {num}" in fname or f"{body.lower()}-{num}" in fname:
                return True
    return False
