"""Regenerate table_schema chunks from tables.jsonl for a manifest (no full 07b)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from table_schema_lib import build_table_schema_chunk


def _load_doc_ids(manifest: Path) -> list[str]:
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [row["doc_id"].strip() for row in reader if row.get("doc_id")]


def regen_doc(tables_path: Path, chunks_path: Path, *, source: str = "KR") -> int:
    if not tables_path.exists():
        return 0
    tables = [json.loads(line) for line in tables_path.open(encoding="utf-8") if line.strip()]
    schema_chunks = [
        build_table_schema_chunk(t, source=source, file_name=str(t.get("source_file") or ""))
        for t in tables
        if not t.get("is_pseudo_table")
    ]
    by_id = {c["chunk_id"]: c for c in schema_chunks}

    existing: list[dict] = []
    if chunks_path.exists():
        for line in chunks_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("chunk_type") == "table_schema":
                continue
            existing.append(rec)

    merged = existing + list(by_id.values())
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(by_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/kr_table_top22.csv"))
    parser.add_argument("--tables-dir", type=Path, default=Path("data/processed/tables"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    args = parser.parse_args()

    root = _SCRIPT_DIR.parent
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    tables_dir = args.tables_dir if args.tables_dir.is_absolute() else root / args.tables_dir
    chunks_dir = args.chunks_dir if args.chunks_dir.is_absolute() else root / args.chunks_dir

    doc_ids = _load_doc_ids(manifest)
    total = 0
    for doc_id in doc_ids:
        tables_path = tables_dir / doc_id / "tables.jsonl"
        chunks_path = chunks_dir / doc_id / "table_chunks.jsonl"
        n = regen_doc(tables_path, chunks_path)
        total += n
        print(f"{doc_id}: {n} table_schema chunks")
    print(f"total table_schema regenerated: {total}")


if __name__ == "__main__":
    main()
