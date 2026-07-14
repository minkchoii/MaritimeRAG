"""
V06 Rule/Guidance answer: legacy vs improved comparison.

  python scripts/36_rule_lookup_v06_compare.py
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
from rule_lookup_retrieval_log import save_rule_lookup_run_log
from rule_lookup_structured_answer import (
    build_rule_lookup_legacy_answer,
    build_rule_lookup_structured_answer,
    build_rule_lookup_ungrouped_answer,
)


def _top_files(chunks, k=5):
    seen = []
    for c in chunks:
        fn = str(c.file_name or "")
        if fn and fn not in seen:
            seen.append(fn)
        if len(seen) >= k:
            break
    return seen


def main() -> None:
    qpath = Path("data/eval/pilot_validation_questions.jsonl")
    rows = [q for q in load_questions(qpath) if q.get("question_id") == "V06"]
    if not rows:
        raise SystemExit("V06 not found")
    row = rows[0]
    row["category"] = "rule_lookup"
    row["class_society_hint"] = "DNV"

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

    legacy = build_rule_lookup_legacy_answer(retrieved)
    ungrouped = build_rule_lookup_ungrouped_answer(
        retrieved, question=question, pool=pool
    )
    improved, warnings = build_rule_lookup_structured_answer(
        retrieved,
        question=question,
        pool=pool,
        warning_flags=list(row.get("warning_flags") or []),
    )

    out_dir = Path("data/processed/logs/rule_lookup_hybrid")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{ts}_V06_grouped_compare.json"

    payload = {
        "question_id": "V06",
        "question": question,
        "top5_file_names": _top_files(pool, 5),
        "legacy_answer": legacy,
        "ungrouped_answer": ungrouped,
        "grouped_answer": improved,
        "grouped_warning_flags": warnings,
        "checks": {
            "grouped_section4_entry_count": improved.count("\n- 유형:"),
            "ungrouped_section4_entry_count": ungrouped.count("\n- 유형:"),
            "grouped_lists_ru_ou_individually_in_s1": sum(
                1 for code in ("0102", "0103", "0104") if f"**DNV-RU-OU-{code}" in improved
            ),
            "grouped_has_ru_ou_group": "RU-OU 계열" in improved or "RU-OU/" in improved,
            "cg0264_confirmed": "DNV-CG-0264" in improved and "확정 여부: 확정" in improved,
            "ru_ou_not_confirmed_in_section4": not bool(
                re.search(r"RU-OU[\s\S]*?확정 여부: 확정", improved)
            ),
            "negative_applicability_flagged": "negative_applicability_clause" in warnings,
            "candidate_not_confirmed": "candidate_not_confirmed" in warnings,
            "too_many_candidates": "too_many_candidates" in warnings,
        },
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    save_rule_lookup_run_log(
        question=question,
        row=row,
        category="rule_lookup",
        dense_results=(row.get("_hybrid_retrieval_log") or {}).get("dense_results", []),
        bm25_results=(row.get("_hybrid_retrieval_log") or {}).get("bm25_results", []),
        fused_results=(row.get("_hybrid_retrieval_log") or {}).get("fused_results", []),
        retrieved=retrieved,
        answer=improved,
        warning_flags=warnings,
        tag="v06_improved",
    )

    print(f"\nWrote {out_path}")
    # Windows console may not support all Unicode; summary only on stdout.
    print(
        json.dumps(
            {
                "question_id": payload["question_id"],
                "checks": payload["checks"],
                "grouped_warning_flags": payload["grouped_warning_flags"],
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
