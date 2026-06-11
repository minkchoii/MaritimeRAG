from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import re

from clause_parse import article_number_from_text, is_article_clause_number

REFERENCE_CLAUSE_RE = re.compile(r"(\d{3,})\.\s*의\s*규정")
from embedding_policy import (
    DEFAULT_EMBEDDING_PRESET,
    ALLOWED_EMBEDDING_PRESETS,
    REQUIRED_LANGUAGES,
    embed_texts_local,
    resolve_embedding_config,
)


def text_for_embedding(chunk: dict) -> str:
    """Enrich chunks for retrieval: article headers, cross-refs (준용), clause tags."""
    text = str(chunk.get("text", "")).strip()
    if not text:
        return text

    article = str(chunk.get("article_number") or chunk.get("clause_number") or "")
    if not article or not is_article_clause_number(article):
        inferred = article_number_from_text(text)
        if inferred:
            article = inferred

    parts: list[str] = []

    if article and is_article_clause_number(article):
        parts.append(f"조문 {article}절 {article}.")

    if len(text) < 200 and article and is_article_clause_number(article):
        first_line = text.split("\n", 1)[0].strip()
        if not first_line.startswith(f"{article}."):
            text = f"{article}. {text}"

    ref = REFERENCE_CLAUSE_RE.search(text)
    if ref:
        ref_no = ref.group(1)
        if ref_no != article:
            parts.append(f"참조 {ref_no}절 {ref_no}.")

    if parts:
        prefix = " ".join(parts)
        if not text.startswith(prefix):
            return f"{prefix} {text}"
    return text


DEFAULT_INDEX_TYPES = frozenset({"text", "table", "picture"})
MIN_INDEX_TEXT_CHARS = 10
PICTURE_PLACEHOLDER_MARKERS = ("[picture element", "refer to crop image")


def load_chunk_ids_from_suspicious_csv(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    import csv

    ids: set[str] = set()
    with csv_path.open(encoding="utf-8", newline="") as csv_f:
        reader = csv.DictReader(csv_f)
        for row in reader:
            chunk_id = row.get("chunk_id", "").strip()
            if chunk_id:
                ids.add(chunk_id)
    return ids


def load_chunks(chunks_path: Path) -> list[dict]:
    chunks: list[dict] = []
    with chunks_path.open(encoding="utf-8") as chunks_f:
        for line in chunks_f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def is_placeholder_picture(text: str) -> bool:
    lower = text.lower()
    return any(m in lower for m in PICTURE_PLACEHOLDER_MARKERS)


def should_index_chunk(
    chunk: dict,
    include_types: frozenset[str],
    skip_ids: set[str],
    min_chars: int,
) -> bool:
    chunk_id = str(chunk.get("chunk_id", ""))
    if chunk_id in skip_ids:
        text = str(chunk.get("text", "")).strip()
        article = str(chunk.get("article_number") or chunk.get("clause_number") or "")
        if not (is_article_clause_number(article) and len(text) >= 80):
            return False

    element_type = str(chunk.get("element_type", "")).lower()
    if element_type not in include_types:
        return False

    text = str(chunk.get("text", "")).strip()
    text_len = len(text)

    if element_type == "picture":
        if chunk.get("linked_caption_id") and text_len >= min_chars:
            return True
        if not is_placeholder_picture(text) and text_len >= min_chars:
            return True
        return False

    return text_len >= min_chars


def filter_chunks_for_index(
    chunks: list[dict],
    include_types: frozenset[str],
    skip_ids: set[str],
    min_chars: int,
) -> list[dict]:
    return [
        chunk
        for chunk in chunks
        if should_index_chunk(chunk, include_types, skip_ids, min_chars)
    ]


def build_chroma_index(
    chunks: list[dict],
    out_dir: Path,
    collection_name: str,
    embeddings: list[list[float]],
) -> None:
    import chromadb
    from chromadb.config import Settings

    out_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(out_dir), settings=Settings(anonymized_telemetry=False))

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    ids = [str(c["chunk_id"]) for c in chunks]
    documents = [str(c.get("text", "")) for c in chunks]
    metadatas = []
    for chunk in chunks:
        meta = {
            "doc_id": str(chunk.get("doc_id", "")),
            "page_number": int(chunk.get("page_number", 0)),
            "element_type": str(chunk.get("element_type", "")),
            "element_id": str(chunk.get("element_id", "")),
            "clause_number": str(chunk.get("clause_number", "")),
            "article_number": str(chunk.get("article_number", "")),
            "crop_path": str(chunk.get("crop_path", "")),
            "source_page_image": str(chunk.get("source_page_image", "")),
        }
        metadatas.append(meta)

    batch_size = 64
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
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
    index_chunks = filter_chunks_for_index(all_chunks, include_types, skip_ids, args.min_chars)

    if not index_chunks:
        raise ValueError("No chunks selected for indexing. Check filters and chunks.jsonl.")

    texts = [text_for_embedding(c) for c in index_chunks]
    model_name = str(embed_config["model"])

    print(f"Embedding preset: {args.embedding_preset}")
    note = str(embed_config.get("note", "")).encode("ascii", "replace").decode("ascii")
    print(f"Model: {model_name} ({note})")
    print(f"Deployment: local | Languages: {', '.join(REQUIRED_LANGUAGES)}")
    print(f"Indexing {len(index_chunks)} / {len(all_chunks)} chunks")

    embeddings = embed_texts_local(texts, model_name, for_query=False)

    collection_name = f"{args.doc_id}_chunks"
    build_chroma_index(index_chunks, chroma_dir, collection_name, embeddings)

    type_counts: dict[str, int] = {}
    for chunk in index_chunks:
        t = str(chunk.get("element_type", ""))
        type_counts[t] = type_counts.get(t, 0) + 1

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
