from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import re

from embedding_policy import (
    DEFAULT_EMBEDDING_PRESET,
    ALLOWED_EMBEDDING_PRESETS,
    REQUIRED_LANGUAGES,
    embed_texts_local,
    resolve_embedding_config,
)
from index_build_lib import (
    MIN_INDEX_TEXT_CHARS,
    build_chroma_index,
    chunk_metadata,
    embedding_text_and_mode,
    filter_chunks_for_index,
    folder_from_path,
    load_chunk_ids_from_suspicious_csv,
    load_chunks,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build vector index with local multilingual (KO+EN) embeddings only."
    )
    parser.add_argument("--doc-id", type=str, required=True)
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/index"))
    parser.add_argument(
        "--embedding-preset",
        type=str,
        default=DEFAULT_EMBEDDING_PRESET,
        choices=sorted(ALLOWED_EMBEDDING_PRESETS),
        help="Local multilingual (KO+EN) preset; Chinese/cloud models blocked",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override HuggingFace model id (local multilingual only; validated)",
    )
    parser.add_argument(
        "--include-types",
        type=str,
        default="text,table,picture",
        help="Comma-separated chunk types to index",
    )
    parser.add_argument(
        "--skip-suspicious",
        action="store_true",
        default=True,
        help="Exclude chunks listed in suspicious_chunks.csv (default: on)",
    )
    parser.add_argument(
        "--no-skip-suspicious",
        action="store_false",
        dest="skip_suspicious",
        help="Index all chunks including suspicious ones",
    )
    parser.add_argument("--min-chars", type=int, default=MIN_INDEX_TEXT_CHARS)
    args = parser.parse_args()

    embed_config = resolve_embedding_config(args.embedding_preset, args.model)
    include_types = frozenset(t.strip().lower() for t in args.include_types.split(",") if t.strip())

    chunks_path = args.chunks_dir / args.doc_id / "chunks.jsonl"
    index_doc_dir = args.index_dir / args.doc_id
    chroma_dir = index_doc_dir / "chroma"
    manifest_path = index_doc_dir / "index_manifest.json"

    skip_ids: set[str] = set()
    if args.skip_suspicious:
        suspicious_csv = Path("data/processed/logs") / f"{args.doc_id}_suspicious_chunks.csv"
        skip_ids = load_chunk_ids_from_suspicious_csv(suspicious_csv)

    all_chunks = load_chunks(chunks_path)
    table_chunks_path = args.chunks_dir / args.doc_id / "table_chunks.jsonl"
    table_chunks: list[dict] = []
    has_structured_tables = table_chunks_path.exists()
    if has_structured_tables:
        table_chunks = load_chunks(table_chunks_path)
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
        raise ValueError("No chunks selected for indexing. Check filters and chunks.jsonl.")

    import pandas as pd

    manifest_csv = Path("data/manifests/pdf_manifest.csv")
    source, file_name, folder = "UNKNOWN", "", ""
    if manifest_csv.exists():
        df = pd.read_csv(manifest_csv)
        row = df[df["doc_id"] == args.doc_id]
        if not row.empty:
            meta_row = row.iloc[0].to_dict()
            source = str(meta_row.get("source", "UNKNOWN"))
            file_name = str(meta_row.get("file_name", ""))
            folder = folder_from_path(str(meta_row.get("file_path", "")))

    texts: list[str] = []
    metadatas: list[dict] = []
    for chunk in index_chunks:
        text, mode = embedding_text_and_mode(
            chunk, source=source, file_name=file_name, folder=folder
        )
        texts.append(text)
        metadatas.append(
            chunk_metadata(
                chunk,
                source=source,
                file_name=file_name,
                folder=folder,
                embedding_mode=mode,
            )
        )
    model_name = str(embed_config["model"])

    print(f"Embedding preset: {args.embedding_preset}")
    note = str(embed_config.get("note", "")).encode("ascii", "replace").decode("ascii")
    print(f"Model: {model_name} ({note})")
    print(f"Deployment: local | Languages: {', '.join(REQUIRED_LANGUAGES)}")
    print(f"Indexing {len(index_chunks)} / {len(all_chunks)} chunks")

    embeddings = embed_texts_local(texts, model_name, for_query=False)

    collection_name = f"{args.doc_id}_chunks"
    build_chroma_index(index_chunks, texts, metadatas, embeddings, chroma_dir, collection_name)

    type_counts: dict[str, int] = {}
    chunk_type_counts: dict[str, int] = {}
    for chunk in index_chunks:
        t = str(chunk.get("element_type", ""))
        type_counts[t] = type_counts.get(t, 0) + 1
        ct = str(chunk.get("chunk_type", "") or "")
        if ct:
            chunk_type_counts[ct] = chunk_type_counts.get(ct, 0) + 1

    manifest = {
        "doc_id": args.doc_id,
        "embedding_preset": args.embedding_preset,
        "embedding_model": model_name,
        "embedding_provider": embed_config["provider"],
        "embedding_deployment": "local",
        "languages": list(embed_config.get("languages", REQUIRED_LANGUAGES)),
        "embedding_policy": "local_multilingual_ko_en_no_chinese",
        "collection_name": collection_name,
        "chroma_path": chroma_dir.resolve().as_posix(),
        "chunks_source": chunks_path.resolve().as_posix(),
        "total_chunks_in_file": len(all_chunks),
        "indexed_chunks": len(index_chunks),
        "skipped_suspicious": len(skip_ids),
        "include_types": sorted(include_types),
        "indexed_by_type": type_counts,
        "indexed_by_chunk_type": chunk_type_counts,
        "table_chunks_indexed": len(table_chunks) if has_structured_tables else 0,
    }
    index_doc_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Chroma index: {chroma_dir}")
    print(f"Manifest: {manifest_path}")
    print("Indexed by type:")
    for element_type, count in sorted(type_counts.items()):
        print(f"  - {element_type}: {count}")


if __name__ == "__main__":
    main()
