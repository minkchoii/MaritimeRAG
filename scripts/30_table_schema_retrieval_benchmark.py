"""Benchmark: table_schema 2-stage retrieval vs legacy table-first / baseline."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rag_answer_lib import load_unified_collection, retrieve_for_question
from rag_eval_lib import load_questions
from table_query_parser import parse_table_query
from table_schema_retrieval import build_table_schema_raw, route_table_candidates


def _table_id_match(retrieved, gold_table_id: str, k: int = 10) -> bool:
    if not gold_table_id:
        return False
    for c in retrieved[:k]:
        if c.table_id == gold_table_id:
            return True
    return False


def _table_rank(retrieved, gold_table_id: str, k: int = 10) -> int | None:
    for i, c in enumerate(retrieved[:k], 1):
        if c.table_id == gold_table_id:
            return i
    return None


def _row_match(debug: dict | None, gold_row: str) -> bool | None:
    if not gold_row:
        return None
    if not debug:
        return False
    mr = str(debug.get("matched_row") or "")
    return gold_row in mr or mr in gold_row


def _column_match(debug: dict | None, gold_col: str) -> bool | None:
    if not gold_col:
        return None
    if not debug:
        return False
    mc = str(debug.get("matched_column") or "")
    return gold_col in mc or mc in gold_col


def retrieve_with_mode(
    collection,
    model: str,
    row: dict,
    *,
    mode: str,
    top_k: int,
    chunks_dir: Path,
):
    r = dict(row)
    r["_table_qa"] = True
    if mode == "schema":
        os.environ["MARITIME_TABLE_SCHEMA_RETRIEVAL"] = "1"
    elif mode == "legacy_first":
        os.environ["MARITIME_TABLE_SCHEMA_RETRIEVAL"] = "0"
    else:
        r.pop("_table_qa", None)
        os.environ["MARITIME_TABLE_SCHEMA_RETRIEVAL"] = "0"
    hits = retrieve_for_question(
        collection,
        model,
        r,
        top_k=top_k,
        fetch_k=max(top_k * 3, 30),
        chunks_dir=chunks_dir,
        gold_doc_filter=False,
    )
    debug = r.get("_table_retrieval_debug")
    return hits, debug


def main() -> None:
    parser = argparse.ArgumentParser(description="Table schema retrieval benchmark.")
    parser.add_argument("--questions", type=Path, default=Path("data/eval/table_schema_regression.jsonl"))
    parser.add_argument("--collection-id", type=str, default="kr_tables")
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("data/processed/logs/table_schema_benchmark.json"))
    args = parser.parse_args()

    questions = load_questions(args.questions)
    collection, model, _ = load_unified_collection(args.collection_id, args.index_dir)

    results: list[dict] = []
    summary: dict[str, dict] = {}

    for mode in ("baseline", "legacy_first", "schema"):
        n_table = 0
        ranks: list[int] = []
        conf_fail_ok = 0
        conf_fail_n = 0
        for row in questions:
            hits, debug = retrieve_with_mode(
                collection, model, row, mode=mode, top_k=args.top_k, chunks_dir=args.chunks_dir
            )
            gold_tid = str(row.get("gold_table_id") or "")
            match = _table_id_match(hits, gold_tid, args.top_k)
            rank = _table_rank(hits, gold_tid, args.top_k)
            if gold_tid:
                n_table += int(match)
                if rank:
                    ranks.append(rank)
            item = {
                "qid": row.get("qid"),
                "category": row.get("category"),
                "mode": mode,
                "table_id_match": match,
                "selected_table_rank": rank,
                "row_match": _row_match(debug, str(row.get("gold_row_key") or "")),
                "column_match": _column_match(debug, str(row.get("gold_column") or "")),
                "confidence": (debug or {}).get("retrieval_confidence"),
                "passes_gate": (debug or {}).get("passes_confidence_gate"),
            }
            if mode == "schema" and debug:
                item["top_candidates"] = [
                    {"table_id": c.get("table_id"), "combined": c.get("combined_score")}
                    for c in (debug.get("selected_table_candidates") or [])[:5]
                ]
                if not debug.get("passes_confidence_gate") and not match:
                    conf_fail_ok += 1
                if not debug.get("passes_confidence_gate"):
                    conf_fail_n += 1
            results.append(item)

        summary[mode] = {
            "table_id_match_rate": round(n_table / max(1, len(questions)), 3),
            "mean_rank_when_hit": round(sum(ranks) / len(ranks), 2) if ranks else None,
            "confidence_fail_correctness": round(conf_fail_ok / max(1, conf_fail_n), 3) if conf_fail_n else None,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Export ranking example for first chemistry question
    sample_q = next((q for q in questions if q.get("category") == "chemistry"), questions[0])
    parsed = parse_table_query(str(sample_q["question"]))
    cands = route_table_candidates(collection, str(sample_q["question"]), model, parsed)
    ranking_path = args.out.parent / "table_candidate_ranking_example.json"
    ranking_path.write_text(
        json.dumps(
            {
                "question": sample_q["question"],
                "parsed_query": parsed.to_dict(),
                "candidates": [c.to_dict() for c in cands],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {ranking_path}")


if __name__ == "__main__":
    main()
