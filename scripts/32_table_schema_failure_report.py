"""Per-question failure analysis for table_schema retrieval regression set."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rag_answer_lib import load_unified_collection
from rag_eval_lib import load_questions
from table_query_parser import parse_table_query
from table_schema_retrieval import route_table_candidates, score_table_candidate
from table_schema_lib import parse_schema_from_document


def _classify_failure(
    *,
    gold_table_id: str,
    candidates: list,
    parsed,
    gold_rank: int | None,
) -> list[str]:
    reasons: list[str] = []
    if not gold_table_id:
        return reasons
    cand_ids = [c.table_id for c in candidates]
    if gold_table_id not in cand_ids:
        reasons.append("missing_gold")
        return reasons
    gold_bd = next(c for c in candidates if c.table_id == gold_table_id)
    if gold_rank and gold_rank > 1:
        top = candidates[0]
        if gold_bd.column_match < 0.15 and parsed.column_entities:
            reasons.append("parsing")
        if gold_bd.row_entity_match < 0.3 and parsed.row_entities:
            reasons.append("parsing")
        if top.row_entity_match >= 0.8 and top.column_match < 0.2 and parsed.column_entities:
            if "lot_treatment" in " ".join(
                (parse_schema_from_document("", top.meta or {}).get("table_topics") or [])
            ):
                reasons.append("pseudo_table")
        if not reasons:
            if top.table_id != gold_table_id:
                reasons.append("scoring")
        if not parsed.row_entities and not parsed.column_entities:
            reasons.append("query_parser")
    return list(dict.fromkeys(reasons))


def _candidate_detail(parsed, meta: dict, document: str, vector_distance: float) -> dict:
    bd = score_table_candidate(
        parsed,
        vector_distance=vector_distance,
        meta=meta,
        document=document,
    )
    vec_sim = round(max(0.0, 1.0 - min(1.0, bd.vector_distance)), 4)
    return {
        "table_id": bd.table_id,
        "page": (meta or {}).get("page"),
        "caption": (meta or {}).get("caption"),
        "vector_score": vec_sim,
        "caption_match": bd.caption_match,
        "topic_match": bd.table_topic_match,
        "column_match": bd.column_match,
        "row_match": bd.row_entity_match,
        "unit_match": bd.unit_match,
        "keyword_match": bd.keyword_match,
        "final_score": bd.combined_score,
    }


def analyze_question(collection, model: str, row: dict, *, top_n: int = 10) -> dict:
    question = str(row["question"])
    gold_tid = str(row.get("gold_table_id") or "")
    parsed = parse_table_query(question)
    candidates = route_table_candidates(collection, question, model, parsed, top_k=top_n)

    gold_rank = None
    for i, c in enumerate(candidates, 1):
        if c.table_id == gold_tid:
            gold_rank = i
            break

    selected_tid = candidates[0].table_id if candidates else ""
    top_candidates = [c.to_dict() for c in candidates[:top_n]]

    failure_reasons = _classify_failure(
        gold_table_id=gold_tid,
        candidates=candidates,
        parsed=parsed,
        gold_rank=gold_rank,
    )

    return {
        "qid": row.get("qid"),
        "question": question,
        "gold_table_id": gold_tid,
        "selected_table_id": selected_tid,
        "gold_table_rank": gold_rank,
        "parsed_query": parsed.to_dict(),
        "top10_table_candidates": top_candidates,
        "failure_reasons": failure_reasons,
        "category": row.get("category"),
        "question_type": row.get("question_type"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Table schema retrieval failure report.")
    parser.add_argument("--questions", type=Path, default=Path("data/eval/table_schema_regression.jsonl"))
    parser.add_argument("--collection-id", type=str, default="kr_tables_v1")
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/logs/table_schema_failure_report.json"),
    )
    args = parser.parse_args()

    root = _SCRIPT_DIR.parent
    questions_path = args.questions if args.questions.is_absolute() else root / args.questions
    index_dir = args.index_dir if args.index_dir.is_absolute() else root / args.index_dir
    out_path = args.out if args.out.is_absolute() else root / args.out

    questions = load_questions(questions_path)
    collection, model, _ = load_unified_collection(args.collection_id, index_dir)

    report_items = [analyze_question(collection, model, row, top_n=args.top_n) for row in questions]
    by_reason: dict[str, list[str]] = {}
    for item in report_items:
        for reason in item.get("failure_reasons") or []:
            by_reason.setdefault(reason, []).append(str(item.get("qid")))

    payload = {
        "collection_id": args.collection_id,
        "n_questions": len(questions),
        "items": report_items,
        "failure_summary": {k: {"count": len(v), "qids": v} for k, v in by_reason.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["failure_summary"], ensure_ascii=False, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
