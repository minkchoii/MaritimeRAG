"""
RAG validation: automatic retrieval scoring + manual spot-check workbook.

Examples:
  python scripts/13_rag_validate.py --doc-id kr_1_2025 --mode auto
  python scripts/13_rag_validate.py --doc-id kr_1_2025 --mode spot --spot-top-k 3
  python scripts/13_rag_validate.py --doc-id kr_1_2025 --mode both
  python scripts/13_rag_validate.py --doc-id kr_1_2025 --mode spot --question-id KR1_Q054
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from embedding_policy import DEFAULT_EMBEDDING_PRESET, embed_texts_local, resolve_embedding_config
from rag_eval_lib import (
    evaluate_question,
    load_chunk_text_map,
    load_manifest,
    load_questions,
    load_chunks,
    run_retrieval_eval,
)


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def write_spot_check_report(
    *,
    path: Path,
    doc_id: str,
    results: list,
    chunk_texts: dict[str, str],
    spot_top_k: int,
    preview_chars: int,
) -> None:
    lines = [
        f"Manual spot-check workbook — {doc_id}",
        f"Top-{spot_top_k} retrieved chunks per question (read and mark O/X by hand).",
        "",
        "Legend:",
        "  [AUTO HIT]  = gold chunk appeared in top-k (retrieval OK)",
        "  [AUTO MISS] = gold chunk not in top-k",
        "  [GOLD] tag on a chunk = matches gold_chunk_ids or page+clause",
        "",
        "After reading top-3 chunks, record:",
        "  retrieval_ok: Y/N  — was the right regulation retrieved?",
        "  answer_ok: Y/N     — can you answer the question from these chunks?",
        "",
        "=" * 72,
    ]

    for r in results:
        auto = "HIT" if r.hit_at_k else "MISS"
        lines.append("")
        lines.append(f"=== {r.question_id} [AUTO {auto}] ===")
        lines.append(f"Q: {r.question}")
        lines.append(
            f"Gold: {r.page_band} | p{r.gold_page} | clause={r.gold_clause or '-'} | "
            f"chunks={', '.join(r.gold_chunk_ids) or '(none)'}"
        )
        if r.note:
            lines.append(f"Note: {r.note}")
        if r.keyword_total:
            lines.append(f"Keywords (auto): {r.keyword_hits}/{r.keyword_total} on hit chunk")
        lines.append("  retrieval_ok: [ ] Y  [ ] N")
        lines.append("  answer_ok:      [ ] Y  [ ] N")
        lines.append("")

        shown = r.top_results[:spot_top_k]
        if not shown:
            lines.append("  (no retrieval results)")
            continue

        for item in shown:
            chunk_id = item["chunk_id"]
            tag = " [GOLD]" if item.get("gold_match") else ""
            clause = item.get("clause_number") or ""
            clause_part = f" clause={clause}" if clause else ""
            lines.append(
                f"--- Rank {item['rank']}: {chunk_id} | p{item.get('page_number')}"
                f"{clause_part} | dist={item['distance']:.4f}{tag} ---"
            )
            body = chunk_texts.get(chunk_id) or item.get("preview") or ""
            if len(body) > preview_chars:
                body = body[:preview_chars] + "\n... (truncated)"
            lines.append(body)
            lines.append("")

        lines.append("-" * 72)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_spot_check(
    *,
    doc_id: str,
    questions_path: Path,
    question_ids: list[str] | None,
    spot_top_k: int,
    eval_top_k: int,
    index_dir: Path,
    chunks_dir: Path,
    output_dir: Path,
    preview_chars: int,
    embedding_preset: str | None,
) -> Path:
    questions = load_questions(questions_path)
    if question_ids:
        id_set = set(question_ids)
        questions = [q for q in questions if str(q.get("question_id")) in id_set]
        if not questions:
            raise ValueError(f"No questions matched ids: {question_ids}")

    eval_rows = [q for q in questions if str(q.get("gold_doc_id", doc_id)) == doc_id]
    if not eval_rows:
        raise ValueError(f"No questions for doc_id={doc_id}")

    index_doc_dir = index_dir / doc_id
    manifest = load_manifest(index_doc_dir)
    preset = embedding_preset or manifest.get("embedding_preset", DEFAULT_EMBEDDING_PRESET)
    embed_config = resolve_embedding_config(preset, str(manifest.get("embedding_model", "")))
    model_name = str(embed_config["model"])

    chunks_path = chunks_dir / doc_id / "chunks.jsonl"
    chunks = load_chunks(chunks_path)
    chunk_texts = load_chunk_text_map(chunks_path)

    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=manifest["chroma_path"],
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(manifest["collection_name"])
    indexed_ids = set(collection.get(include=[])["ids"])

    top_k = max(spot_top_k, eval_top_k)
    query_texts = [str(row["question"]) for row in eval_rows]
    print(f"Embedding {len(query_texts)} queries for spot-check (top_k={top_k})...")
    query_vectors = embed_texts_local(query_texts, model_name, for_query=True)

    results = []
    for row, query_vector in zip(eval_rows, query_vectors):
        results.append(
            evaluate_question(collection, row, query_vector, chunks, indexed_ids, top_k)
        )

    out_path = output_dir / f"{doc_id}_spot_check.txt"
    write_spot_check_report(
        path=out_path,
        doc_id=doc_id,
        results=results,
        chunk_texts=chunk_texts,
        spot_top_k=spot_top_k,
        preview_chars=preview_chars,
    )
    return out_path


def main() -> None:
    configure_stdout()

    parser = argparse.ArgumentParser(description="RAG validation: auto score + manual spot check.")
    parser.add_argument("--doc-id", type=str, required=True)
    parser.add_argument(
        "--mode",
        choices=("auto", "spot", "both"),
        default="both",
        help="auto=Recall@k report, spot=top-k chunk workbook for manual review",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="Questions JSONL (default: data/eval/<doc_id>_questions.jsonl)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="top-k for auto eval")
    parser.add_argument("--spot-top-k", type=int, default=3, help="chunks shown per question in spot report")
    parser.add_argument("--preview-chars", type=int, default=2500, help="max chars per chunk in spot report")
    parser.add_argument("--question-id", action="append", dest="question_ids", help="Limit spot check to ids")
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/logs"))
    parser.add_argument("--embedding-preset", type=str, default=None)
    args = parser.parse_args()

    questions_path = args.questions or Path("data/eval") / f"{args.doc_id}_questions.jsonl"
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    if args.mode in ("auto", "both"):
        print("=== Automatic retrieval scoring ===")
        summary, _, json_path, txt_path = run_retrieval_eval(
            doc_id=args.doc_id,
            questions_path=questions_path,
            top_k=args.top_k,
            index_dir=args.index_dir,
            chunks_dir=args.chunks_dir,
            output_dir=args.output_dir,
            embedding_preset=args.embedding_preset,
        )
        print(
            f"Recall@{args.top_k}: {summary['recall_at_k']:.1%} "
            f"({summary['hits_at_k']}/{summary['num_questions']})"
        )
        print(f"Gold pages: p{summary['gold_page_min']} – p{summary['gold_page_max']}")
        print(f"Report: {txt_path}")
        print(f"JSON:   {json_path}")

    if args.mode in ("spot", "both"):
        print("\n=== Manual spot-check workbook ===")
        spot_path = run_spot_check(
            doc_id=args.doc_id,
            questions_path=questions_path,
            question_ids=args.question_ids,
            spot_top_k=args.spot_top_k,
            eval_top_k=args.top_k,
            index_dir=args.index_dir,
            chunks_dir=args.chunks_dir,
            output_dir=args.output_dir,
            preview_chars=args.preview_chars,
            embedding_preset=args.embedding_preset,
        )
        print(f"Open and review: {spot_path}")


if __name__ == "__main__":
    main()
