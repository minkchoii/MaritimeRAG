"""
Interactive / batch query helper for a single-document RAG index.

Examples:
  python scripts/rag_query.py --doc-id kr_1_2025 --query "902절 탈급 절차는?"
  python scripts/rag_query.py --doc-id kr_1_2025 -i
  python scripts/rag_query.py --doc-id kr_1_2025 -i --full-text --top-k 3
  python scripts/rag_query.py --doc-id kr_1_2025 --questions data/eval/kr_1_2025_questions.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from embedding_policy import DEFAULT_EMBEDDING_PRESET, embed_texts_local, resolve_embedding_config
from rag_eval_lib import load_chunk_text_map
from retrieval_search import enrich_query_for_embedding, query_with_hybrid_ranking


def load_manifest(index_doc_dir: Path) -> dict:
    path = index_doc_dir / "index_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Index not built yet: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_body(
    chunk_id: str,
    doc: str | None,
    chunk_texts: dict[str, str] | None,
    *,
    full_text: bool,
    preview_chars: int,
) -> str:
    text = (chunk_texts or {}).get(chunk_id) or doc or ""
    if full_text:
        if len(text) > preview_chars:
            return text[:preview_chars] + "\n... (truncated)"
        return text
    return text.replace("\n", " ")[:200]


def query_collection(
    collection,
    query: str,
    model_name: str,
    top_k: int,
    *,
    chunk_texts: dict[str, str] | None = None,
    full_text: bool = False,
    preview_chars: int = 2500,
) -> None:
    embed_query = enrich_query_for_embedding(query, model_name)
    vector = embed_texts_local([embed_query], model_name, for_query=True)[0]
    results = query_with_hybrid_ranking(collection, query, vector, top_k=top_k)

    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]

    print(f"\nQ: {query}")
    for rank, (chunk_id, distance, meta, doc) in enumerate(
        zip(ids, distances, metadatas, documents), start=1
    ):
        meta = meta or {}
        clause = meta.get("clause_number") or meta.get("article_number") or ""
        clause_part = f" clause={clause}" if clause else ""
        print(
            f"  {rank}. {chunk_id} | p{meta.get('page_number')}{clause_part} | "
            f"dist={distance:.4f}"
        )
        body = chunk_body(
            chunk_id, doc, chunk_texts, full_text=full_text, preview_chars=preview_chars
        )
        if full_text:
            print("     ---")
            for line in body.splitlines():
                print(f"     {line}")
            print("     ---")
        else:
            print(f"     {body}")


def run_interactive(
    collection,
    model_name: str,
    top_k: int,
    *,
    chunk_texts: dict[str, str] | None,
    full_text: bool,
    preview_chars: int,
) -> None:
    print("대화형 검색 모드 (종료: quit / exit / q / 빈 Enter)")
    print(f"doc index ready | top_k={top_k} | full_text={full_text}")
    while True:
        try:
            query = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break
        if not query or query.lower() in {"quit", "exit", "q"}:
            print("종료합니다.")
            break
        query_collection(
            collection,
            query,
            model_name,
            top_k,
            chunk_texts=chunk_texts,
            full_text=full_text,
            preview_chars=preview_chars,
        )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Query a document RAG index (single question, interactive, or JSONL batch)."
    )
    parser.add_argument("--doc-id", type=str, required=True)
    parser.add_argument("--query", type=str, default=None, help="One-shot question text")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Type questions in a loop (no JSONL needed)",
    )
    parser.add_argument("--questions", type=Path, default=None, help="JSONL with question field")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--full-text",
        action="store_true",
        help="Print full chunk text (spot-check style) instead of 200-char preview",
    )
    parser.add_argument("--preview-chars", type=int, default=2500)
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--embedding-preset", type=str, default=None)
    args = parser.parse_args()

    if sum(bool(x) for x in (args.query, args.interactive, args.questions)) > 1:
        parser.error("Use only one of: --query, -i/--interactive, --questions")

    index_doc_dir = args.index_dir / args.doc_id
    manifest = load_manifest(index_doc_dir)
    preset = args.embedding_preset or manifest.get("embedding_preset", DEFAULT_EMBEDDING_PRESET)
    embed_config = resolve_embedding_config(preset, str(manifest.get("embedding_model", "")))
    model_name = str(embed_config["model"])

    chunk_texts = None
    if args.full_text:
        chunks_path = args.chunks_dir / args.doc_id / "chunks.jsonl"
        if chunks_path.exists():
            chunk_texts = load_chunk_text_map(chunks_path)
        else:
            print(f"[WARN] chunks not found, using index text only: {chunks_path}")

    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=manifest["chroma_path"],
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(manifest["collection_name"])

    common = {
        "chunk_texts": chunk_texts,
        "full_text": args.full_text,
        "preview_chars": args.preview_chars,
    }

    if args.interactive:
        run_interactive(
            collection,
            model_name,
            args.top_k,
            **common,
        )
    elif args.questions:
        with args.questions.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                query_collection(
                    collection,
                    str(row["question"]),
                    model_name,
                    args.top_k,
                    **common,
                )
    elif args.query:
        query_collection(collection, args.query, model_name, args.top_k, **common)
    else:
        parser.error("Provide --query, -i/--interactive, or --questions")


if __name__ == "__main__":
    main()
