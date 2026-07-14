"""
V01–V05 route audit: retrieval routing, topics, coverage per question.

  python scripts/39_v01_v05_route_audit.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from meeting_category_profile import build_meeting_retrieval_profile
from meeting_structured_answer import build_meeting_structured_answer
from meeting_topic_cluster import _topic_id_for_text
from rag_answer_lib import load_questions, load_unified_collection, run_retrieval_only
from rag_inprocess import DEFAULT_CHUNKS_DIR, DEFAULT_INDEX_DIR, DEFAULT_UNIFIED

QIDS = ("V01", "V02", "V03", "V04", "V05")


def _topic_label(chunk_or_hit: dict) -> str:
    text = chunk_or_hit.get("text") or chunk_or_hit.get("document") or ""
    return _topic_id_for_text(str(text))


def _hits_with_topic(items: list[dict], pool_by_id: dict) -> list[dict]:
    out = []
    for h in items[:10]:
        cid = h.get("artifact") or h.get("chunk_id") or ""
        chunk = pool_by_id.get(cid, {})
        text = getattr(chunk, "text", "") if chunk else ""
        out.append(
            {
                "file_name": (h.get("file_name") or "")[:70],
                "page": h.get("page"),
                "topic": _topic_id_for_text(str(text)),
                "chunk_id": cid[:40],
            }
        )
    return out


def _audit_row(row: dict, collection, embed_model) -> dict:
    legacy_cat = str(row.get("category") or "")
    out = run_retrieval_only(
        row,
        collection,
        embed_model,
        chunks_dir=DEFAULT_CHUNKS_DIR,
        top_k=8,
        fetch_k=56,
    )
    pool = out.get("retrieval_pool") or out["retrieved"]
    log = row.get("_hybrid_retrieval_log") or {}
    mprofile = build_meeting_retrieval_profile(
        str(row["question"]), row, legacy_category=legacy_cat
    )
    answer, warnings, meta = build_meeting_structured_answer(
        pool[:40],
        question=str(row["question"]),
        row=row,
        profile=mprofile,
        warning_flags=list(row.get("warning_flags") or []),
    )
    pool_by_id = {str(getattr(c, "chunk_id", "")): c for c in pool}

    dense_top = log.get("dense_top20") or []
    bm25_top = log.get("bm25_top20") or []
    fused_top = log.get("fused_top10") or []

    dense_topics = []
    for h in dense_top[:10]:
        cid = h.get("chunk_id", "")
        c = pool_by_id.get(cid)
        dense_topics.append(
            {
                "file": (h.get("file_name") or "")[:60],
                "page": h.get("page"),
                "topic": _topic_id_for_text(str(getattr(c, "text", "") if c else "")),
            }
        )
    bm25_topics = []
    for h in bm25_top[:10]:
        cid = h.get("chunk_id", "")
        c = pool_by_id.get(cid)
        bm25_topics.append(
            {
                "file": (h.get("file_name") or "")[:60],
                "page": h.get("page"),
                "topic": _topic_id_for_text(str(getattr(c, "text", "") if c else "")),
            }
        )
    fused_topics = []
    for h in fused_top[:10]:
        cid = h.get("chunk_id", "")
        c = pool_by_id.get(cid)
        fused_topics.append(
            {
                "file": (h.get("file_name") or "")[:60],
                "page": h.get("page"),
                "topic": _topic_id_for_text(str(getattr(c, "text", "") if c else "")),
            }
        )

    coverage = meta.get("coverage_check") or {}
    return {
        "question_id": row.get("question_id"),
        "question": row.get("question"),
        "top_level_category": mprofile.top_level_category,
        "internal_intent": mprofile.internal_intent,
        "applied_query_expansion": (log.get("applied_query_expansion") or log.get("bm25_query") or "")[:200],
        "dense_top10": dense_topics,
        "bm25_top10": bm25_topics,
        "fused_top10": fused_topics,
        "final_used_citations": meta.get("used_citations"),
        "detected_topics": meta.get("detected_topics"),
        "coverage_check": coverage,
        "coverage_pass": all(coverage.values()) if coverage else False,
        "warning_flags": warnings,
        "answer_section1": answer.split("## 2)")[0] if "## 2)" in answer else answer[:800],
    }


def main() -> None:
    qpath = Path("data/eval/pilot_validation_questions.jsonl")
    rows = {q["question_id"]: q for q in load_questions(qpath) if q.get("question_id") in QIDS}
    collection, embed_model, _ = load_unified_collection(DEFAULT_UNIFIED, DEFAULT_INDEX_DIR)

    audits = [_audit_row(dict(rows[qid]), collection, embed_model) for qid in QIDS]

    print("\n=== V01~V05 Route Audit ===\n")
    header = (
        "| QID | top_level | intent | coverage | topics | warnings |"
    )
    print(header)
    print("|" + "---|" * 6)
    for a in audits:
        topics = ",".join(a.get("detected_topics") or [])[:40]
        warns = ",".join((a.get("warning_flags") or [])[:3])
        print(
            f"| {a['question_id']} | {a['top_level_category'][:18]} | {a['internal_intent']} | "
            f"{a['coverage_pass']} | {topics} | {warns or '-'} |"
        )

    for a in audits:
        print(f"\n--- {a['question_id']}: {a['question']} ---")
        print(f"intent={a['internal_intent']} expansion={a['applied_query_expansion'][:100]}...")
        print("fused_top10 topics:", [x["topic"] for x in a["fused_top10"][:5]])
        print("detected_topics:", a["detected_topics"])
        print("coverage:", a["coverage_check"])
        print("section1 excerpt written to log file")

    out_dir = Path("data/processed/logs/meeting_structured")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{ts}_V01_V05_route_audit.json"
    path.write_text(json.dumps({"audits": audits}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
