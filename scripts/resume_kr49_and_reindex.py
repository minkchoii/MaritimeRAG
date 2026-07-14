"""
Resume KR remaining batch (docs without chunks.jsonl), then rebuild unified full_corpus.

  python scripts/resume_kr49_batch.py          # preprocess only
  python scripts/rebuild_full_corpus_458.py    # unified index only (after chunks exist)
  python scripts/resume_kr49_and_reindex.py    # both
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KR_LIST = ROOT / "data/manifests/kr_remaining_48.csv"
RAG_CORPUS = ROOT / "data/manifests/rag_corpus_457.csv"
EXCLUDED = ROOT / "data/manifests/kr_excluded.csv"
CHUNKS_DIR = ROOT / "data/processed/chunks"
LOG = ROOT / "data/processed/logs/kr49_batch.log"
PY = sys.executable


def docs_missing_chunks() -> pd.DataFrame:
    df = pd.read_csv(KR_LIST)
    missing = []
    for doc_id in df["doc_id"].astype(str):
        if not (CHUNKS_DIR / doc_id / "chunks.jsonl").exists():
            missing.append(doc_id)
    return df[df["doc_id"].astype(str).isin(missing)]


def run_preprocess(remaining: pd.DataFrame) -> None:
    if remaining.empty:
        print("All KR 49 docs already have chunks.jsonl — skipping preprocess.")
        return
    tmp = ROOT / "data/manifests/kr_remaining_48_resume.csv"
    remaining.to_csv(tmp, index=False, encoding="utf-8")
    print(f"Preprocessing {len(remaining)} doc(s) … log: {LOG}")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY,
        "scripts/run_rag_batch.py",
        "--doc-list",
        str(tmp),
        "--steps",
        "pdf,layout,merge,crop,chunks,quality",
    ]
    with LOG.open("a", encoding="utf-8") as log_f:
        log_f.write("\n\n=== resume_kr49_and_reindex ===\n")
        subprocess.run(cmd, cwd=str(ROOT), check=False, stdout=log_f, stderr=subprocess.STDOUT)


def run_unified_rebuild() -> None:
    if not RAG_CORPUS.exists():
        raise FileNotFoundError(f"Missing {RAG_CORPUS}. Run manifest export first.")
    print(f"Rebuilding unified full_corpus from {RAG_CORPUS.name} …")
    subprocess.run(
        [
            PY,
            "scripts/10_build_unified_index.py",
            "--collection-id",
            "full_corpus",
            "--doc-list",
            str(RAG_CORPUS),
        ],
        cwd=str(ROOT),
        check=True,
    )


def main() -> None:
    remaining = docs_missing_chunks()
    run_preprocess(remaining)
    still = docs_missing_chunks()
    if not still.empty:
        print(f"WARN: {len(still)} doc(s) still missing chunks — unified rebuild may skip them.")
        for doc_id in still["doc_id"].astype(str).tolist()[:10]:
            print(f"  - {doc_id}")
    run_unified_rebuild()
    print("Done.")


if __name__ == "__main__":
    main()
