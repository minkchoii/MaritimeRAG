"""
V07 Rule/Guidance answer: before/after alt-fuel clause coverage comparison.

  python scripts/37_rule_lookup_v07_compare.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rag_answer_lib import load_questions, load_unified_collection, retrieve_for_question
from rag_inprocess import DEFAULT_CHUNKS_DIR, DEFAULT_INDEX_DIR, DEFAULT_UNIFIED
from retrieval_verification import (
    build_evidence_table,
    compute_must_cover_coverage,
    get_must_cover_items,
    hybrid_score_lookup,
    parse_citation_ids,
)
from rule_lookup_alt_fuel import compute_alt_fuel_must_cover
from rule_lookup_retrieval_log import save_rule_lookup_run_log
from rule_lookup_structured_answer import (
    build_rule_lookup_legacy_answer,
    build_rule_lookup_structured_answer,
)

PRIOR_CONTENT_MARKERS = [
    "Section 15",
    "low-flashpoint",
    "dual fuel",
    "alternative fuel",
    "fuel storage",
    "fuel supply",
    "engine",
    "IGF",
    "IGC",
    "survey",
    "2025",
]


def _serialize_ranked(items: list[dict], *, limit: int) -> list[dict]:
    out: list[dict] = []
    for row in items[:limit]:
        out.append(
            {
                "file_name": row.get("file_name"),
                "page": row.get("page_number") or row.get("page"),
                "score": row.get("score") or row.get("bm25_score"),
                "rrf_score": row.get("rrf_score"),
                "chunk_id": row.get("chunk_id"),
            }
        )
    return out


def _content_presence(text: str) -> dict[str, bool]:
    low = (text or "").lower()
    return {m: m.lower() in low for m in PRIOR_CONTENT_MARKERS}


def main() -> None:
    qpath = Path("data/eval/pilot_validation_questions.jsonl")
    rows = [q for q in load_questions(qpath) if q.get("question_id") == "V07"]
    if not rows:
        raise SystemExit("V07 not found")
    row = rows[0]
    row["category"] = "rule_lookup"
    row["class_society_hint"] = "LR"

    collection, embed_model, _ = load_unified_collection(DEFAULT_UNIFIED, DEFAULT_INDEX_DIR)
    pool = retrieve_for_question(
        collection,
        embed_model,
        row,
        top_k=56,
        fetch_k=56,
        chunks_dir=DEFAULT_CHUNKS_DIR,
        unified_id=DEFAULT_UNIFIED,
        index_dir=DEFAULT_INDEX_DIR,
    )
    retrieved = pool[:8]
    question = str(row["question"])
    log = row.get("_hybrid_retrieval_log") or {}
    hybrid_log = log

    legacy = build_rule_lookup_legacy_answer(retrieved)
    improved, warnings = build_rule_lookup_structured_answer(
        retrieved,
        question=question,
        pool=pool,
        warning_flags=list(row.get("warning_flags") or []),
    )

    must_items = get_must_cover_items(row)
    must_chunks = compute_must_cover_coverage(must_items, list(pool or retrieved))
    must_answer = compute_must_cover_coverage(must_items, list(pool or retrieved), improved)
    alt_must = compute_alt_fuel_must_cover(list(pool or retrieved), improved)
    citations_used = sorted(parse_citation_ids(improved))

    out_dir = Path("data/processed/logs/rule_lookup_hybrid")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{ts}_V07_alt_fuel_compare.json"

    payload = {
        "question_id": "V07",
        "question": question,
        "legacy_answer": legacy,
        "improved_answer": improved,
        "improved_warning_flags": warnings,
        "must_cover_keyword_coverage": alt_must,
        "must_cover_standard": must_answer,
        "bm25_query": log.get("bm25_query"),
        "bm25_query_tokens": log.get("bm25_query_tokens"),
        "bm25_term_checks": log.get("bm25_term_checks"),
        "dense_top20": log.get("dense_top20") or log.get("dense_results", [])[:20],
        "bm25_top20": log.get("bm25_top20") or log.get("bm25_results", [])[:20],
        "rrf_top10": log.get("fused_top10") or log.get("fused_results", [])[:10],
        "hybrid_warning_flags": log.get("warning_flags", []),
        "evidence_bm25_scores": [
            r.get("bm25_score")
            for r in build_evidence_table(
                retrieved,
                improved,
                score_lookup=hybrid_score_lookup(hybrid_log),
            )
        ],
        "citations_in_answer": citations_used,
        "content_presence": {
            "legacy": _content_presence(legacy),
            "improved": _content_presence(improved),
        },
        "checks": {
            "improved_has_section15": "section 15" in improved.lower(),
            "improved_has_low_flashpoint": "low-flashpoint" in improved.lower() or "저인화점" in improved,
            "improved_has_dual_fuel": "dual fuel" in improved.lower(),
            "improved_has_clause_candidate_label": "관련 조항 후보" in improved,
            "improved_section4_has_main_content": "- 주요 내용:" in improved,
            "legacy_content_retained": all(
                _content_presence(improved).get(k, False)
                for k in ("Section 15", "low-flashpoint", "dual fuel", "engine")
                if _content_presence(legacy).get(k, False)
            ),
            "bm25_called": bool(log.get("bm25_top20") or log.get("bm25_results")),
            "bm25_term_checks_all_true": all(log.get("bm25_term_checks", {}).values())
            if log.get("bm25_term_checks")
            else False,
            "evidence_has_bm25_score": any(
                r.get("bm25_score") is not None
                for r in build_evidence_table(
                    retrieved,
                    improved,
                    score_lookup=hybrid_score_lookup(hybrid_log),
                )
            ),
            "must_cover_answer_yes": sum(
                1 for r in alt_must if r["included_in_answer"] == "Yes"
            ),
        },
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    save_rule_lookup_run_log(
        question=question,
        row=row,
        category="rule_lookup",
        dense_results=log.get("dense_results", []),
        bm25_results=log.get("bm25_results", []),
        fused_results=log.get("fused_results", []),
        retrieved=retrieved,
        answer=improved,
        warning_flags=warnings,
        tag="v07_alt_fuel",
    )

    print(f"\nWrote {out_path}")
    print(
        json.dumps(
            {
                "question_id": "V07",
                "checks": payload["checks"],
                "must_cover_in_answer": [
                    r["must_cover"]
                    for r in alt_must
                    if r["included_in_answer"] == "Yes"
                ],
                "citations_in_answer": citations_used,
                "improved_warning_flags": warnings,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
