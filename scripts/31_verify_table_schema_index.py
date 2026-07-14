"""Post-reindex verification: chunk counts, AH32 routing, benchmark @1/@3/@10."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rag_answer_lib import load_unified_collection, retrieve_for_question
from rag_eval_lib import load_questions
from table_query_parser import parse_table_query
from table_schema_lib import parse_schema_from_document

GOLD_P17 = "kr_kr_rules_2_2025_6bdd7355_p0017_t013"
P19_TABLE = "kr_kr_rules_2_2025_6bdd7355_p0019_t015"
P20_TABLE = "kr_kr_rules_2_2025_6bdd7355_p0020_t016"
AH32_Q = "AH 32 고장력강 황(S) 허용 한도는?"


def _load_table_schema(collection, table_id: str) -> dict:
    raw = collection.get(where={"table_id": table_id}, include=["documents", "metadatas"])
    for i, meta in enumerate(raw.get("metadatas") or []):
        if (meta or {}).get("chunk_type") == "table_schema":
            return parse_schema_from_document(raw["documents"][i], meta)
    if raw.get("documents"):
        return parse_schema_from_document(raw["documents"][0], raw["metadatas"][0])
    return {}


def p17_schema_checks(collection) -> dict:
    schema = _load_table_schema(collection, GOLD_P17)
    cols = schema.get("normalized_column_names") or schema.get("column_names") or []
    rows = schema.get("normalized_row_entities") or schema.get("row_entities") or []
    topics = schema.get("table_topics") or []
    return {
        "table_id": GOLD_P17,
        "has_s_column": "S" in cols or any("S" == str(c).upper() for c in cols),
        "column_names": cols,
        "topic_is_chemical_composition": "chemical_composition" in topics or "화학성분" in topics,
        "table_topics": topics,
        "has_ah32_row": "AH32" in rows or any("AH32" in str(r).upper().replace(" ", "") for r in rows),
        "row_entities_sample": rows[:12],
        "caption": schema.get("caption"),
    }


def ah32_competitor_breakdown(collection, model: str, candidates: list) -> dict:
    """Score breakdown for p.19/p.20 and other AH32-related competitors."""
    parsed = parse_table_query(AH32_Q)
    out: dict[str, dict] = {}
    for tid in (P19_TABLE, P20_TABLE):
        schema = _load_table_schema(collection, tid)
        if not schema:
            out[tid] = {"status": "not_found"}
            continue
        bd = next((c for c in candidates if c.table_id == tid), None)
        out[tid] = {
            "page": schema.get("page"),
            "caption": schema.get("caption"),
            "table_topics": schema.get("table_topics"),
            "column_names": schema.get("column_names"),
            "route_rank": next((i for i, c in enumerate(candidates, 1) if c.table_id == tid), None),
            "score_breakdown": bd.to_dict() if bd else None,
            "why_demoted": (
                "not_in_route_top"
                if bd is None
                else f"topic={bd.table_topic_match:.2f} col={bd.column_match:.2f} row={bd.row_entity_match:.2f} final={bd.combined_score:.3f}"
            ),
        }
    return out


def count_chunk_types(collection) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for ct in ("table_summary", "table_markdown", "table_row", "table_schema"):
        try:
            raw = collection.get(where={"chunk_type": ct}, include=[])
            counts[ct] = len(raw["ids"])
        except Exception as exc:
            counts[ct] = -1
            counts[f"{ct}_error"] = str(exc)[:120]
    return dict(counts)


def ah32_top10_report(collection, model: str, *, chunks_dir: Path) -> dict:
    row = {"question": AH32_Q, "_table_qa": True}
    os.environ["MARITIME_TABLE_SCHEMA_RETRIEVAL"] = "1"
    hits = retrieve_for_question(
        collection,
        model,
        row,
        top_k=10,
        fetch_k=30,
        chunks_dir=chunks_dir,
        gold_doc_filter=False,
    )
    debug = row.get("_table_retrieval_debug") or {}
    parsed = parse_table_query(AH32_Q)
    candidates = route_table_candidates(collection, AH32_Q, model, parsed)

    cand_by_id = {c.table_id: c.to_dict() for c in candidates}

    top10 = []
    for i, h in enumerate(hits[:10], 1):
        bd = cand_by_id.get(h.table_id or "", {})
        top10.append(
            {
                "rank": i,
                "table_id": h.table_id,
                "page": h.page_number,
                "caption": h.caption,
                "chunk_type": h.chunk_type,
                "distance": round(float(h.distance), 4),
                "score_breakdown": {
                    "combined": bd.get("combined_score"),
                    "vector": bd.get("vector_distance"),
                    "caption_match": bd.get("caption_match"),
                    "table_topic_match": bd.get("table_topic_match"),
                    "column_match": bd.get("column_match"),
                    "row_entity_match": bd.get("row_entity_match"),
                    "unit_match": bd.get("unit_match"),
                    "keyword_match": bd.get("keyword_match"),
                },
            }
        )

    ranks = [i for i, h in enumerate(hits, 1) if h.table_id == GOLD_P17]
    route_ranks = [i for i, c in enumerate(candidates, 1) if c.table_id == GOLD_P17]

    return {
        "question": AH32_Q,
        "gold_table_id": GOLD_P17,
        "parsed_query": parsed.to_dict(),
        "p17_schema_checks": p17_schema_checks(collection),
        "competitor_breakdown": ah32_competitor_breakdown(collection, model, candidates),
        "retrieval_confidence": debug.get("retrieval_confidence"),
        "selected_table_id": debug.get("selected_table_id"),
        "top10_hits": top10,
        "p17_in_retrieval_top1": 1 in ranks,
        "p17_in_retrieval_top3": any(r <= 3 for r in ranks),
        "p17_in_retrieval_top5": any(r <= 5 for r in ranks),
        "p17_in_retrieval_top10": any(r <= 10 for r in ranks),
        "p17_retrieval_rank": ranks[0] if ranks else None,
        "p17_route_rank": route_ranks[0] if route_ranks else None,
        "route_top5": [c.to_dict() for c in candidates[:5]],
    }


def _table_id_match_at_k(retrieved, gold_table_id: str, k: int) -> bool:
    for c in retrieved[:k]:
        if c.table_id == gold_table_id:
            return True
    return False


def benchmark_at_k(
    collection,
    model: str,
    questions: list[dict],
    *,
    mode: str,
    chunks_dir: Path,
    ks: tuple[int, ...] = (1, 3, 10),
) -> dict:
    metrics = {f"table_id_match@{k}": 0 for k in ks}
    n = 0
    for row in questions:
        gold = str(row.get("gold_table_id") or "")
        if not gold:
            continue
        n += 1
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
            top_k=max(ks),
            fetch_k=max(ks) * 3,
            chunks_dir=chunks_dir,
            gold_doc_filter=False,
        )
        for k in ks:
            metrics[f"table_id_match@{k}"] += int(_table_id_match_at_k(hits, gold, k))
    for k in ks:
        metrics[f"table_id_match@{k}"] = round(metrics[f"table_id_match@{k}"] / max(1, n), 3)
    metrics["n_with_gold"] = n
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-id", default="kr_tables")
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--questions", type=Path, default=Path("data/eval/table_schema_regression.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/logs/table_schema_verify.json"))
    args = parser.parse_args()

    collection, model, _ = load_unified_collection(args.collection_id, args.index_dir)

    chunk_counts = count_chunk_types(collection)
    ah32 = ah32_top10_report(collection, model, chunks_dir=args.chunks_dir)

    questions = load_questions(args.questions)
    bench = {}
    for mode in ("baseline", "legacy_first", "schema"):
        bench[mode] = benchmark_at_k(collection, model, questions, mode=mode, chunks_dir=args.chunks_dir)

    report = {
        "chunk_type_counts": chunk_counts,
        "table_schema_total": chunk_counts.get("table_schema", 0),
        "ah32_report": ah32,
        "benchmark": bench,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Chunk type counts (kr_tables) ===")
    for ct in ("table_summary", "table_markdown", "table_row", "table_schema"):
        print(f"  {ct}: {chunk_counts.get(ct, 0)}")
    print(f"\n=== AH32 p.17 placement ===")
    print(f"  retrieval rank: {ah32.get('p17_retrieval_rank')}")
    print(f"  route rank: {ah32.get('p17_route_rank')}")
    print(f"  top1={ah32.get('p17_in_retrieval_top1')} top3={ah32.get('p17_in_retrieval_top3')} top5={ah32.get('p17_in_retrieval_top5')}")
    checks = ah32.get("p17_schema_checks") or {}
    print(f"  p17 S column: {checks.get('has_s_column')}  topic chemical: {checks.get('topic_is_chemical_composition')}  AH32 row: {checks.get('has_ah32_row')}")
    print("\n=== Benchmark table_id_match ===")
    for mode, m in bench.items():
        print(f"  {mode}: @1={m.get('table_id_match@1')} @3={m.get('table_id_match@3')} @10={m.get('table_id_match@10')} (n={m.get('n_with_gold')})")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
