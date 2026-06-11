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
    REQUIRED_LANGUAGES,
    embed_texts_local,
    resolve_embedding_config,
)


def load_manifest(index_doc_dir: Path) -> dict:
    manifest_path = index_doc_dir / "index_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Index manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query Chroma index (local KO+EN embeddings only)."
    )
    parser.add_argument("--doc-id", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/index"))
    parser.add_argument(
        "--embedding-preset",
        type=str,
        default=None,
        help="Must match index build preset (default: read from manifest)",
    )
    args = parser.parse_args()

    index_doc_dir = args.index_dir / args.doc_id
    manifest = load_manifest(index_doc_dir)
    preset = args.embedding_preset or manifest.get("embedding_preset", DEFAULT_EMBEDDING_PRESET)
    embed_config = resolve_embedding_config(preset, str(manifest.get("embedding_model", "")))

    import chromadb
    from chromadb.config import Settings

    chroma_path = manifest["chroma_path"]
    collection_name = manifest["collection_name"]
    client = chromadb.PersistentClient(path=chroma_path, settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection(collection_name)

    model_name = str(embed_config["model"])
    query_vector = embed_texts_local([args.query], model_name, for_query=True)[0]
    results = collection.query(query_embeddings=[query_vector], n_results=args.top_k)

    print(f"Query: {args.query}")
    print(f"Model: {model_name} (local, languages={','.join(REQUIRED_LANGUAGES)})")
    print(f"Top-{args.top_k} results:\n")

    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]

    for rank, (chunk_id, distance, meta, doc) in enumerate(
        zip(ids, distances, metadatas, documents), start=1
    ):
        preview = doc.replace("\n", " ")[:160]
        clause = meta.get("clause_number", "")
        clause_part = f" clause={clause}" if clause else ""
        print(
            f"{rank}. {chunk_id} | p{meta.get('page_number')} | "
            f"{meta.get('element_type')}{clause_part} | dist={distance:.4f}"
        )
        print(f"   {preview}\n")


if __name__ == "__main__":
    main()
