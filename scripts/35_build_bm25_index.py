"""
Build BM25 sparse index for a unified Chroma collection.

  python scripts/35_build_bm25_index.py --unified full_corpus
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from bm25_index import CHROMA_GET_BATCH_SIZE, build_bm25_from_collection
from rag_answer_lib import load_unified_collection
from rag_resource_cache import unified_index_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description="Build persisted BM25 index for unified corpus.")
    parser.add_argument("--unified", type=str, default="full_corpus")
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--batch-size", type=int, default=CHROMA_GET_BATCH_SIZE)
    args = parser.parse_args()

    collection, _, _ = load_unified_collection(args.unified, args.index_dir)
    fp = unified_index_fingerprint(args.unified, args.index_dir)
    if args.rebuild:
        from bm25_index import bm25_index_dir
        import shutil

        out = bm25_index_dir(args.index_dir, args.unified)
        if out.exists():
            shutil.rmtree(out)

    try:
        total = collection.count()
    except Exception:
        total = None
    if total is not None:
        print(f"Chroma collection: {total} chunks (batch_size={args.batch_size})")
    else:
        print(f"Fetching chunks in batches of {args.batch_size}…")

    inst = build_bm25_from_collection(
        collection,
        unified_id=args.unified,
        index_dir=args.index_dir,
        fingerprint=fp,
        batch_size=args.batch_size,
    )
    print(f"BM25 index built: {len(inst.chunk_ids)} chunks → {args.index_dir / f'unified_{args.unified}' / 'bm25'}")


if __name__ == "__main__":
    main()
