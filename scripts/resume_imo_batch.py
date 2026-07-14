"""
Resume IMO (MSC/MEPC) batch for docs without a Chroma index.

  python scripts/resume_imo_batch.py

Log: data/processed/logs/imo_batch.log
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def safe_print(line: str) -> None:
    try:
        sys.stdout.write(line)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.buffer.write(line.encode(enc, errors="replace"))

ROOT = Path(__file__).resolve().parents[1]
IMO_CSV = ROOT / "data/manifests/imo_docs.csv"
REMAINING_CSV = ROOT / "data/manifests/imo_docs_remaining.csv"
LOG = ROOT / "data/processed/logs/imo_batch.log"
INDEX_DIR = ROOT / "data/processed/index"
UNIFIED_DIRS = {"unified_pilot_100", "unified_full_corpus"}


def indexed_ids() -> set[str]:
    ids: set[str] = set()
    if not INDEX_DIR.exists():
        return ids
    for d in INDEX_DIR.iterdir():
        if d.is_dir() and d.name not in UNIFIED_DIRS and (
            (d / "chroma").exists() or (d / "index_skipped.json").exists()
        ):
            ids.add(d.name)
    return ids


def main() -> None:
    imo = pd.read_csv(IMO_CSV)
    done = indexed_ids()
    remaining = imo[~imo["doc_id"].isin(done)]
    REMAINING_CSV.parent.mkdir(parents=True, exist_ok=True)
    remaining.to_csv(REMAINING_CSV, index=False, encoding="utf-8-sig")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    print(f"IMO indexed: {len(done & set(imo['doc_id']))} / {len(imo)}")
    print(f"Remaining: {len(remaining)} -> {REMAINING_CSV.relative_to(ROOT)}")
    print(f"Log file: {LOG.relative_to(ROOT)}")
    print("=" * 60, flush=True)

    if remaining.empty:
        print("All IMO documents already indexed.")
        return

    cmd = [
        sys.executable,
        "scripts/run_rag_batch.py",
        "--doc-list",
        str(REMAINING_CSV.relative_to(ROOT)),
        "--steps",
        "pdf,layout,merge,crop,chunks,quality,index",
    ]

    with LOG.open("a", encoding="utf-8") as log_f:
        log_f.write("\n\n=== resume_imo_batch started ===\n")
        log_f.flush()
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            safe_print(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        code = proc.wait()
        if code != 0:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
