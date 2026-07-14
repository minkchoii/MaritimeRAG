"""
Resume pilot-100 batch for docs without a Chroma index.

Writes remaining list then runs run_rag_batch with stdout+stderr tee'd to
data/processed/logs/pilot_batch.log (safe to tail from Cursor terminal).

Run from Cursor integrated terminal (recommended):
  python scripts/resume_pilot_batch.py

Or: Terminal > Run Task > Pilot 100: Resume batch (integrated terminal)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PILOT_CSV = ROOT / "data/manifests/pilot_100_docs.csv"
REMAINING_CSV = ROOT / "data/manifests/pilot_100_docs_remaining.csv"
LOG = ROOT / "data/processed/logs/pilot_batch.log"
INDEX_DIR = ROOT / "data/processed/index"


def indexed_ids() -> set[str]:
    ids: set[str] = set()
    if not INDEX_DIR.exists():
        return ids
    for d in INDEX_DIR.iterdir():
        if d.is_dir() and d.name != "unified_pilot_100" and (d / "chroma").exists():
            ids.add(d.name)
    return ids


def main() -> None:
    pilot = pd.read_csv(PILOT_CSV)
    done = indexed_ids()
    remaining = pilot[~pilot["doc_id"].isin(done)]
    REMAINING_CSV.parent.mkdir(parents=True, exist_ok=True)
    remaining.to_csv(REMAINING_CSV, index=False, encoding="utf-8-sig")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    print(f"Indexed: {len(done & set(pilot['doc_id']))} / {len(pilot)}")
    print(f"Remaining: {len(remaining)} -> {REMAINING_CSV.relative_to(ROOT)}")
    print(f"Log file: {LOG.relative_to(ROOT)}")
    print("=" * 60, flush=True)

    if remaining.empty:
        print("All pilot documents already indexed.")
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
        log_f.write(f"\n\n=== resume_pilot_batch started ===\n")
        log_f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        code = proc.wait()
        if code != 0:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
