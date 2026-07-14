"""Table QA benchmark: table_recall, row_recall, cell match, citation match."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from embedding_policy import DEFAULT_EMBEDDING_PRESET, embed_texts_local, resolve_embedding_config
from rag_answer_lib import RetrievedChunk, load_unified_collection, retrieve_for_question
from rag_eval_lib import load_chunk_text_map, load_questions
from retrieval_search import enrich_query_for_embedding, query_with_hybrid_ranking
from table_retrieval import is_table_question

TOKEN_RE = re.compile(r"[\w가-힣]+", re.UNICODE)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _contains(hay: str, needle: str) -> bool:
    if not needle:
        return True
    return _norm(needle) in _norm(hay)


def _table_recall_at_k(retrieved: list[RetrievedChunk], row: dict, k: int) -> bool:
    gold_table = str(row.get("gold_table_id") or "")
    gold_page = int(row.get("gold_page") or -1)
    for c in retrieved[:k]:
        if gold_table and c.table_id == gold_table:
            return True
        ct = (c.chunk_type or "").startswith("table")
        if ct and c.doc_id == row.get("gold_doc_id") and c.page_number == gold_page:
            if gold_table and c.table_id and c.table_id != gold_table:
                continue
            return True
    return False


def _row_recall_at_k(retrieved: list[RetrievedChunk], row: dict, k: int) -> bool:
    row_key = str(row.get("gold_row_key") or "")
    if not row_key:
        return _table_recall_at_k(retrieved, row, k)
    for c in retrieved[:k]:
        if c.chunk_type != "table_row":
            continue
        if c.table_id and row.get("gold_table_id") and c.table_id != row.get("gold_table_id"):
            continue
        if _contains(c.text, row_key):
            return True
    return False


def _cell_exact_match(retrieved: list[RetrievedChunk], row: dict, k: int) -> bool:
    answer = str(row.get("gold_answer") or "")
    col = str(row.get("gold_column") or "")
    row_key = str(row.get("gold_row_key") or "")
    for c in retrieved[:k]:
        text = c.text or ""
        if row_key and not _contains(text, row_key):
            continue
        if col and not _contains(text, col):
            continue
        if _contains(text, answer):
            return True
    return False


def _citation_match(retrieved: list[RetrievedChunk], row: dict, k: int) -> bool:
    gold_page = int(row.get("gold_page") or -1)
    gold_doc = str(row.get("gold_doc_id") or "")
    for c in retrieved[:k]:
        if c.doc_id != gold_doc:
            continue
        if c.page_number == gold_page:
            return True
        if c.table_id and c.table_id == str(row.get("gold_table_id") or ""):
            return True
    return False


def _answer_grounded_simple(retrieved: list[RetrievedChunk], row: dict, k: int) -> bool:
    gold = str(row.get("gold_answer") or "")
    if not gold:
        return _table_recall_at_k(retrieved, row, k)
    tokens = [t for t in TOKEN_RE.findall(gold) if len(t) > 1]
    if not tokens:
        return _cell_exact_match(retrieved, row, k)
    for c in retrieved[:k]:
        text = _norm(c.text)
        if all(t.lower() in text for t in tokens[:3]):
            return True
    return False


def retrieve_baseline_only(
    collection,
    model_name: str,
    row: dict,
    *,
    top_k: int,
    chunks_dir: Path,
) -> list[RetrievedChunk]:
    """Baseline retrieval without table-aware merge (for before/after compare)."""
    from rag_answer_lib import retrieve_for_question as _full

    # Temporarily bypass table merge by patching is_table_question - use direct hybrid
    question = str(row["question"])
    embed_query = enrich_query_for_embedding(question, model_name)
    vector = embed_texts_local([embed_query], model_name, for_query=True)[0]
    filter_doc_id = str(row.get("gold_doc_id") or "") or None
    raw = query_with_hybrid_ranking(
        collection,
        question,
        vector,
        top_k=top_k,
        fetch_k=max(top_k * 10, 80),
        doc_id=filter_doc_id,
    )
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
        if doc_id not in chunk_text_cache:
            cp = chunks_dir / doc_id / "chunks.jsonl"
            chunk_text_cache[doc_id] = load_chunk_text_map(cp) if cp.exists() else {}
        full_text = chunk_text_cache[doc_id].get(chunk_id) or doc or ""
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
                chunk_type=str(meta.get("chunk_type") or ""),
                table_id=str(meta.get("table_id") or ""),
            )
        )
    return out


def evaluate_row(
    retrieved: list[RetrievedChunk],
    row: dict,
    k: int,
) -> dict:
    return {
        "table_recall@k": int(_table_recall_at_k(retrieved, row, k)),
        "row_recall@k": int(_row_recall_at_k(retrieved, row, k)),
        "cell_exact_match": int(_cell_exact_match(retrieved, row, k)),
        "citation_match": int(_citation_match(retrieved, row, k)),
        "answer_grounded_simple": int(_answer_grounded_simple(retrieved, row, k)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Table QA retrieval benchmark.")
    parser.add_argument("--questions", type=Path, default=Path("data/eval/table_questions.jsonl"))
    parser.add_argument("--collection-id", type=str, default="kr_1_2025")
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--embedding-preset", type=str, default=DEFAULT_EMBEDDING_PRESET)
    parser.add_argument("--compare-baseline", action="store_true", default=True)
    parser.add_argument("--out", type=Path, default=Path("data/processed/logs/table_qa_benchmark.json"))
    args = parser.parse_args()

    questions = load_questions(args.questions)
    collection, model_name, _manifest = load_unified_collection(args.collection_id, args.index_dir)
    embed_config = resolve_embedding_config(args.embedding_preset, None)
    if not model_name:
        model_name = str(embed_config["model"])

    results: list[dict] = []
    agg = {
        "table_recall@k": 0,
        "row_recall@k": 0,
        "cell_exact_match": 0,
        "citation_match": 0,
        "answer_grounded_simple": 0,
    }
    baseline_agg = dict(agg)

    print(f"| qid | question_type | table_recall@{args.top_k} | row_recall@{args.top_k} | "
          f"cell_exact_match | citation_match | answer |")
    print(f"| --- | ------------- | -------------: | -----------: | "
          f"---------------: | -------------: | ------ |")

    for row in questions:
        qid = str(row.get("qid") or row.get("question_id") or "")
        qtype = str(row.get("question_type") or "")
        eval_row = {"qid": qid, "question": row.get("question"), "question_type": qtype}

        retrieved = retrieve_for_question(
            collection,
            model_name,
            row,
            top_k=args.top_k,
            chunks_dir=args.chunks_dir,
            gold_doc_filter=True,
        )
        metrics = evaluate_row(retrieved, row, args.top_k)
        eval_row.update(metrics)
        eval_row["table_aware"] = is_table_question(str(row.get("question") or ""))
        eval_row["top_chunk_types"] = [c.chunk_type or c.element_type for c in retrieved[:5]]

        if args.compare_baseline:
            baseline = retrieve_baseline_only(
                collection, model_name, row, top_k=args.top_k, chunks_dir=args.chunks_dir
            )
            bmetrics = evaluate_row(baseline, row, args.top_k)
            eval_row["baseline_table_recall@k"] = bmetrics["table_recall@k"]
            eval_row["baseline_row_recall@k"] = bmetrics["row_recall@k"]
            eval_row["baseline_cell_exact_match"] = bmetrics["cell_exact_match"]
            for k in agg:
                baseline_agg[k] += bmetrics[k]

        for k in agg:
            agg[k] += metrics[k]

        answer_preview = str(row.get("gold_answer") or "")[:40]
        print(
            f"| {qid} | {qtype} | {metrics['table_recall@k']} | {metrics['row_recall@k']} | "
            f"{metrics['cell_exact_match']} | {metrics['citation_match']} | {answer_preview} |"
        )
        results.append(eval_row)

    n = max(len(questions), 1)
    summary = {k: round(v / n, 3) for k, v in agg.items()}
    out_payload = {
        "collection_id": args.collection_id,
        "top_k": args.top_k,
        "n_questions": len(questions),
        "summary": summary,
        "baseline_summary": {k: round(v / n, 3) for k, v in baseline_agg.items()} if args.compare_baseline else {},
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSummary (table-aware): {summary}")
    if args.compare_baseline:
        print(f"Summary (baseline):    {out_payload['baseline_summary']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
