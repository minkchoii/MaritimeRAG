"""Diagnose cell lookup failures: parsing vs retrieval vs generation."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rag_answer_lib import RetrievedChunk, load_unified_collection, retrieve_for_question
from rag_eval_lib import load_questions
from table_extract_lib import load_table_chunks

TOKEN_RE = re.compile(r"[\w가-힣○]+", re.UNICODE)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _contains(hay: str, needle: str) -> bool:
    if not needle:
        return True
    return _norm(needle) in _norm(hay)


def load_tables_map(tables_path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not tables_path.exists():
        return out
    with tables_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tid = str(rec.get("table_id") or "")
            if tid:
                out[tid] = rec
    return out


def load_row_chunks(chunks_path: Path, table_id: str) -> list[dict]:
    if not chunks_path.exists():
        return []
    prefix = f"{table_id}__row_"
    rows: list[dict] = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if str(rec.get("chunk_type")) == "table_row" and str(rec.get("table_id")) == table_id:
                rows.append(rec)
    rows.sort(key=lambda r: int(r.get("row_index") or 0))
    return rows


def find_gold_row_in_table(table: dict, row_key: str) -> dict | None:
    if not row_key:
        return None
    rows = (table.get("table_json") or {}).get("rows") or []
    cols = table.get("column_names") or []
    row_col = cols[0] if cols else "구역"
    for row in rows:
        label = str(row.get(row_col, "") or row.get("항목", ""))
        if _contains(label, row_key):
            return row
    return None


def analyze_parsing(table: dict, row: dict, row_key: str, column: str, gold_answer: str) -> dict:
    cols = table.get("column_names") or []
    issues = table.get("parse_issues") or []
    result = {
        "gold_table_exists": bool(table),
        "gold_row_in_table_json": row is not None,
        "gold_column_exists": bool(not column or column in cols),
        "gold_cell_in_table_json": False,
        "gold_cell_in_row_chunk": False,
        "parse_issues": issues,
        "columns": cols,
    }
    if row and column:
        cell = str(row.get(column, ""))
        result["gold_cell_in_table_json"] = _contains(cell, gold_answer) or gold_answer in cell
        result["table_json_cell"] = cell
    elif row and not column:
        joined = " ".join(str(v) for v in row.values())
        result["gold_cell_in_table_json"] = _contains(joined, gold_answer)
        result["table_json_cell"] = joined[:200]

    row_chunks = load_row_chunks(
        Path("data/processed/chunks") / str(table.get("doc_id", "")) / "table_chunks.jsonl",
        str(table.get("table_id", "")),
    )
    for ch in row_chunks:
        if row_key and not _contains(ch.get("text", ""), row_key):
            continue
        if _contains(ch.get("text", ""), gold_answer):
            result["gold_cell_in_row_chunk"] = True
            result["matching_row_chunk_id"] = ch.get("chunk_id")
            break
        rd = ch.get("row_data") or {}
        if column and _contains(str(rd.get(column, "")), gold_answer):
            result["gold_cell_in_row_chunk"] = True
            result["matching_row_chunk_id"] = ch.get("chunk_id")
            result["row_data_cell"] = rd.get(column)
            break

    return result


def analyze_retrieval(retrieved: list[RetrievedChunk], row: dict) -> dict:
    gold_table = str(row.get("gold_table_id") or "")
    gold_page = int(row.get("gold_page") or -1)
    row_key = str(row.get("gold_row_key") or "")
    column = str(row.get("gold_column") or "")
    gold_answer = str(row.get("gold_answer") or "")

    table_hit_rank = None
    row_hit_rank = None
    cell_hit_rank = None
    for i, c in enumerate(retrieved, start=1):
        if table_hit_rank is None and (c.table_id == gold_table or (
            c.chunk_type.startswith("table") and c.page_number == gold_page
        )):
            table_hit_rank = i
        if row_hit_rank is None and c.chunk_type == "table_row":
            if row_key and _contains(c.text, row_key):
                row_hit_rank = i
        if cell_hit_rank is None and _contains(c.text, gold_answer):
            if not row_key or _contains(c.text, row_key):
                if not column or _contains(c.text, column):
                    cell_hit_rank = i

    return {
        "table_hit_rank": table_hit_rank,
        "row_hit_rank": row_hit_rank,
        "cell_hit_rank": cell_hit_rank,
        "top5": [
            {
                "rank": i,
                "chunk_type": c.chunk_type or c.element_type,
                "table_id": c.table_id,
                "chunk_id": c.chunk_id,
                "preview": (c.text or "")[:120],
            }
            for i, c in enumerate(retrieved[:5], start=1)
        ],
    }


def classify_failure(parsing: dict, retrieval: dict, cell_exact: bool) -> str:
    if cell_exact:
        return "success"
    if not parsing.get("gold_cell_in_table_json"):
        return "parsing"
    if not parsing.get("gold_cell_in_row_chunk"):
        return "parsing"
    if retrieval.get("cell_hit_rank") is None:
        if retrieval.get("row_hit_rank") is None and retrieval.get("table_hit_rank") is None:
            return "retrieval"
        return "retrieval"
    return "retrieval_or_ranking"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose table cell lookup failures.")
    parser.add_argument("--questions", type=Path, default=Path("data/eval/table_questions.jsonl"))
    parser.add_argument("--tables-dir", type=Path, default=Path("data/processed/tables"))
    parser.add_argument("--collection-id", type=str, default="kr_1_2025")
    parser.add_argument("--qid", type=str, default=None, help="Diagnose single question id")
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("data/processed/logs/table_cell_lookup_diagnose.json"))
    args = parser.parse_args()

    questions = [q for q in load_questions(args.questions) if q.get("question_type") == "cell_lookup"]
    if args.qid:
        questions = [q for q in load_questions(args.questions) if str(q.get("qid")) == args.qid]
        if not questions:
            raise SystemExit(f"Unknown qid: {args.qid}")
    doc_id = str(questions[0].get("gold_doc_id") if questions else "kr_1_2025")
    tables_map = load_tables_map(args.tables_dir / doc_id / "tables.jsonl")

    collection, model_name, _ = load_unified_collection(args.collection_id, args.index_dir)
    chunks_dir = Path("data/processed/chunks")

    results: list[dict] = []
    print("| qid | cell_match | failure_type | table_json | row_chunk | table@k | row@k | cell@k |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")

    for row in questions:
        qid = str(row.get("qid"))
        gold_table_id = str(row.get("gold_table_id"))
        table = tables_map.get(gold_table_id, {})
        gold_row = find_gold_row_in_table(table, str(row.get("gold_row_key") or ""))

        parsing = analyze_parsing(
            table,
            gold_row,
            str(row.get("gold_row_key") or ""),
            str(row.get("gold_column") or ""),
            str(row.get("gold_answer") or ""),
        )

        retrieved = retrieve_for_question(
            collection,
            model_name,
            row,
            top_k=args.top_k,
            chunks_dir=chunks_dir,
            gold_doc_filter=True,
        )
        retrieval = analyze_retrieval(retrieved, row)

        cell_in_top = retrieval.get("cell_hit_rank") is not None and retrieval["cell_hit_rank"] <= args.top_k
        failure = classify_failure(parsing, retrieval, cell_in_top)

        entry = {
            "qid": qid,
            "question": row.get("question"),
            "gold_table_id": gold_table_id,
            "gold_row_key": row.get("gold_row_key"),
            "gold_column": row.get("gold_column"),
            "gold_answer": row.get("gold_answer"),
            "parsing": parsing,
            "retrieval": retrieval,
            "failure_type": failure,
            "cell_exact_in_topk": cell_in_top,
        }
        results.append(entry)

        print(
            f"| {qid} | {'Y' if cell_in_top else 'N'} | {failure} | "
            f"{'Y' if parsing['gold_cell_in_table_json'] else 'N'} | "
            f"{'Y' if parsing['gold_cell_in_row_chunk'] else 'N'} | "
            f"{retrieval['table_hit_rank'] or '-'} | {retrieval['row_hit_rank'] or '-'} | "
            f"{retrieval['cell_hit_rank'] or '-'} |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
