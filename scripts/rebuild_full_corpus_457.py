"""Rebuild unified full_corpus from rag_corpus_457.csv (408 + 48 KR, excludes integrated compilation)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAG_CORPUS = ROOT / "data/manifests/rag_corpus_457.csv"

if __name__ == "__main__":
    if not RAG_CORPUS.exists():
        raise SystemExit(f"Missing {RAG_CORPUS}")
    subprocess.run(
        [
            sys.executable,
            "scripts/10_build_unified_index.py",
            "--collection-id",
            "full_corpus",
            "--doc-list",
            str(RAG_CORPUS),
        ],
        cwd=str(ROOT),
        check=True,
    )
