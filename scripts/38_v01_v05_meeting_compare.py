"""
V01–V05 meeting category: structured answer + hybrid retrieval comparison.

  python scripts/38_v01_v05_meeting_compare.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from meeting_category_profile import (
    TOP_LEVEL_LABELS_KO,
    build_meeting_retrieval_profile,
    resolve_top_level_category,
)
from rag_answer_lib import (
    build_context_block,
    generate_extractive_answer,
    load_questions,
    load_unified_collection,
    run_retrieval_only,
)
from rag_inprocess import DEFAULT_CHUNKS_DIR, DEFAULT_INDEX_DIR, DEFAULT_UNIFIED
from meeting_structured_answer import build_meeting_structured_answer

QIDS = ("V01", "V02", "V03", "V04", "V05")


def _legacy_answer(row, retrieved) -> str:
    return generate_extractive_answer(row, retrieved)


def main() -> None:
    qpath = Path("data/eval/pilot_validation_questions.jsonl")
    all_rows = {q["question_id"]: q for q in load_questions(qpath) if q.get("question_id") in QIDS}
    collection, embed_model, _ = load_unified_collection(DEFAULT_UNIFIED, DEFAULT_INDEX_DIR)

    results: list[dict] = []
    for qid in QIDS:
        row = dict(all_rows[qid])
        legacy_cat = str(row.get("category") or "")
        out = run_retrieval_only(
            row,
            collection,
            embed_model,
            chunks_dir=DEFAULT_CHUNKS_DIR,
            top_k=8,
            fetch_k=56,
        )
        retrieved = out["retrieved"]
        pool = out.get("retrieval_pool") or retrieved
        log = row.get("_hybrid_retrieval_log") or {}
        mprofile = build_meeting_retrieval_profile(
            str(row["question"]), row, legacy_category=legacy_cat
        )
        legacy_ans = _legacy_answer(row, retrieved)
        improved, warnings, meta = build_meeting_structured_answer(
            pool[:40],
            question=str(row["question"]),
            row=row,
            profile=mprofile,
            warning_flags=list(row.get("warning_flags") or []),
        )
        top = resolve_top_level_category(legacy_cat)
        coverage = meta.get("coverage_check") or {}
        coverage_pass = all(coverage.values()) if coverage else False

        results.append(
            {
                "question_id": qid,
                "question": row["question"],
                "legacy_category": legacy_cat,
                "top_level_category": top,
                "top_level_label_ko": TOP_LEVEL_LABELS_KO.get(top, top),
                "internal_intent": mprofile.internal_intent,
                "retrieval_profile": mprofile.profile_id,
                "use_dense": mprofile.use_dense,
                "use_bm25": mprofile.use_bm25,
                "use_rrf": mprofile.use_rrf,
                "use_source_tier": mprofile.use_source_tier,
                "coverage_pass": coverage_pass,
                "coverage_check": coverage,
                "warning_flags": warnings,
                "bm25_top3": (log.get("bm25_top20") or [])[:3],
                "fused_top3": (log.get("fused_top10") or [])[:3],
                "legacy_answer_excerpt": legacy_ans[:600],
                "improved_answer": improved,
                "quality_note": _quality_note(qid, improved, legacy_ans),
            }
        )
        print(f"{qid}: coverage_pass={coverage_pass} warnings={warnings[:4]}")

    out_dir = Path("data/processed/logs/meeting_structured")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{ts}_V01_V05_compare.json"
    out_path.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


def _quality_note(qid: str, improved: str, legacy: str) -> str:
    notes: list[str] = []
    if "## 1)" in improved and "## 4)" in improved:
        notes.append("4섹션 형식 적용")
    if len(improved) > len(legacy) * 0.8:
        notes.append("내용 밀도 개선")
    if qid == "V02" and improved.count("\n- ") >= 3:
        notes.append("다항목 요약")
    if "관련 조항 후보" in improved or "추가 확인" in improved:
        notes.append("확정/후보 구분")
    return "; ".join(notes) if notes else "—"


if __name__ == "__main__":
    main()
