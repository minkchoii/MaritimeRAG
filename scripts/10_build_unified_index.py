"""
Build a single Chroma collection across multiple processed documents.

  python scripts/10_build_unified_index.py --doc-list data/manifests/pilot_100_docs.csv
  python scripts/10_build_unified_index.py --doc-id kr_1_2025 --collection-id pilot_100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from embedding_policy import (
    DEFAULT_EMBEDDING_PRESET,
    ALLOWED_EMBEDDING_PRESETS,
    REQUIRED_LANGUAGES,
    embed_texts_local,
    resolve_embedding_config,
)
from index_build_lib import (
    DEFAULT_INDEX_TYPES,
    MIN_INDEX_TEXT_CHARS,
    build_chroma_index,
    chunk_metadata,
    embedding_text_and_mode,
    filter_chunks_for_index,
    folder_from_path,
    load_chunk_ids_from_suspicious_csv,
    load_chunks,
)


def load_doc_list(path: Path) -> list[dict]:
    import pandas as pd

    df = pd.read_csv(path)
    if "doc_id" not in df.columns:
        raise ValueError(f"doc-list must have doc_id column: {path}")
    return df.to_dict(orient="records")


def load_manifest_map(manifest_path: Path) -> dict[str, dict]:
    import pandas as pd

    if not manifest_path.exists():
        return {}
    df = pd.read_csv(manifest_path)
    return {str(row["doc_id"]): row.to_dict() for _, row in df.iterrows()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified multi-document Chroma index.")
    parser.add_argument("--doc-list", type=Path, default=None, help="CSV with doc_id column")
    parser.add_argument("--doc-id", action="append", dest="doc_ids", help="Repeatable doc_id")
    parser.add_argument("--collection-id", type=str, default="pilot_100")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/pdf_manifest.csv"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--embedding-preset", type=str, default=DEFAULT_EMBEDDING_PRESET,
                        choices=sorted(ALLOWED_EMBEDDING_PRESETS))
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--include-types", type=str, default="text,table,picture")
    parser.add_argument("--skip-suspicious", action="store_true", default=True)
    parser.add_argument("--no-skip-suspicious", action="store_false", dest="skip_suspicious")
    parser.add_argument("--min-chars", type=int, default=MIN_INDEX_TEXT_CHARS)
    args = parser.parse_args()

    if args.doc_list:
        rows = load_doc_list(args.doc_list)
        doc_ids = [str(r["doc_id"]) for r in rows]
    elif args.doc_ids:
        doc_ids = args.doc_ids
        rows = [{"doc_id": d} for d in doc_ids]
    else:
        parser.error("Provide --doc-list or --doc-id")

    manifest_map = load_manifest_map(args.manifest)
    include_types = frozenset(t.strip().lower() for t in args.include_types.split(",") if t.strip())
    embed_config = resolve_embedding_config(args.embedding_preset, args.model)
    model_name = str(embed_config["model"])

    all_index_chunks: list[dict] = []
    all_texts: list[str] = []
    all_metas: list[dict] = []
    per_doc_stats: list[dict] = []
    missing_chunks: list[str] = []

    for row in rows:
        doc_id = str(row["doc_id"])
        meta_row = manifest_map.get(doc_id, row)
        source = str(meta_row.get("source", row.get("source", "UNKNOWN")))
        file_name = str(meta_row.get("file_name", row.get("file_name", "")))
        file_path = str(meta_row.get("file_path", row.get("file_path", "")))
        folder = str(row.get("folder", "")) or folder_from_path(file_path)

        chunks_path = args.chunks_dir / doc_id / "chunks.jsonl"
        if not chunks_path.exists():
            missing_chunks.append(doc_id)
            continue

        skip_ids: set[str] = set()
        if args.skip_suspicious:
            suspicious_csv = Path("data/processed/logs") / f"{doc_id}_suspicious_chunks.csv"
            skip_ids = load_chunk_ids_from_suspicious_csv(suspicious_csv)

        all_chunks = load_chunks(chunks_path)
        table_chunks_path = args.chunks_dir / doc_id / "table_chunks.jsonl"
        table_chunks: list[dict] = []
        has_structured_tables = table_chunks_path.exists()
        if has_structured_tables:
            table_chunks = load_chunks(table_chunks_path)
            # Skip legacy plain-text table elements when structured table chunks exist.
            all_chunks = [
                c
                for c in all_chunks
                if not (
                    str(c.get("element_type", "")).lower() == "table"
                    and not c.get("chunk_type")
                )
            ]
            all_chunks.extend(table_chunks)
        index_chunks = filter_chunks_for_index(all_chunks, include_types, skip_ids, args.min_chars)
        if not index_chunks:
            missing_chunks.append(doc_id)
            continue

        for chunk in index_chunks:
            text, mode = embedding_text_and_mode(
                chunk, source=source, file_name=file_name, folder=folder
            )
            all_index_chunks.append(chunk)
            all_texts.append(text)
            all_metas.append(
                chunk_metadata(
                    chunk,
                    source=source,
                    file_name=file_name,
                    folder=folder,
                    embedding_mode=mode,
                )
            )

        table_n = len(table_chunks) if has_structured_tables else 0
        per_doc_stats.append(
            {
                "doc_id": doc_id,
                "source": source,
                "folder": folder,
                "indexed": len(index_chunks),
                "total": len(all_chunks),
                "table_chunks": table_n,
            }
        )
        suffix = f", +{table_n} table chunks" if table_n else ""
        print(f"  {doc_id}: {len(index_chunks)}/{len(all_chunks)} chunks ({source}){suffix}")

    if not all_index_chunks:
        raise ValueError(
            "No chunks indexed. Process documents first (run_rag_batch --doc-list ... --steps chunks)."
            f" Missing chunks for: {missing_chunks[:10]}"
        )

    if missing_chunks:
        print(f"[WARN] Skipped {len(missing_chunks)} doc(s) without chunks: {missing_chunks[:8]}...")

    print(f"\nEmbedding {len(all_texts)} chunks with {model_name}...")
    embeddings = embed_texts_local(all_texts, model_name, for_query=False)

    collection_name = f"maritime_{args.collection_id}_chunks"
    index_root = args.index_dir / f"unified_{args.collection_id}"
    chroma_dir = index_root / "chroma"

    build_chroma_index(
        all_index_chunks,
        all_texts,
        all_metas,
        embeddings,
        chroma_dir,
        collection_name,
    )

    mode_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for m in all_metas:
        mode_counts[m["embedding_mode"]] = mode_counts.get(m["embedding_mode"], 0) + 1
        source_counts[m["source"]] = source_counts.get(m["source"], 0) + 1

    manifest_out = {
        "collection_id": args.collection_id,
        "collection_name": collection_name,
        "embedding_preset": args.embedding_preset,
        "embedding_model": model_name,
        "embedding_provider": embed_config["provider"],
        "languages": list(embed_config.get("languages", REQUIRED_LANGUAGES)),
        "chroma_path": chroma_dir.resolve().as_posix(),
        "doc_list": args.doc_list.resolve().as_posix() if args.doc_list else None,
        "doc_ids": [s["doc_id"] for s in per_doc_stats],
        "indexed_chunks": len(all_index_chunks),
        "indexed_by_source": source_counts,
        "indexed_by_embedding_mode": mode_counts,
        "per_doc": per_doc_stats,
        "missing_chunks_doc_ids": missing_chunks,
    }
    index_root.mkdir(parents=True, exist_ok=True)
    manifest_path = index_root / "index_manifest.json"
    manifest_path.write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nUnified index: {chroma_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Total indexed: {len(all_index_chunks)} chunks from {len(per_doc_stats)} documents")
    print("By source:", source_counts)
    print("By embedding_mode:", mode_counts)


if __name__ == "__main__":
    main()
