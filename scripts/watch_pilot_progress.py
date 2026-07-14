"""
Live progress monitor for the pilot-100 RAG batch.

Usage (in your Cursor terminal):
  python scripts/watch_pilot_progress.py

Tail the log from another tab:
  Get-Content data/processed/logs/pilot_progress_live.txt -Wait -Tail 25
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PILOT_CSV = Path("data/manifests/pilot_100_docs.csv")
INDEX_DIR = Path("data/processed/index")
PROCESSED = Path("data/processed")
DEFAULT_LOG = Path("data/processed/logs/pilot_progress_live.txt")


def indexed_doc_ids() -> set[str]:
    ids: set[str] = set()
    if not INDEX_DIR.exists():
        return ids
    for d in INDEX_DIR.iterdir():
        if d.is_dir() and d.name != "unified_pilot_100" and (d / "chroma").exists():
            ids.add(d.name)
    return ids


def latest_mtime_under(path: Path, pattern: str = "*") -> float:
    if not path.exists():
        return 0.0
    best = path.stat().st_mtime
    for f in path.rglob(pattern):
        if f.is_file():
            best = max(best, f.stat().st_mtime)
    return best


def infer_stage(doc_id: str) -> tuple[str, str]:
    """Return (stage_label, detail) for the most advanced in-progress step."""
    base = PROCESSED
    pages = base / "pages" / doc_id
    layout = base / "layout_json" / doc_id
    merged = base / "layout_json_merged" / doc_id
    crops = base / "crops_merged" / doc_id
    chunks = base / "chunks" / doc_id / "chunks.jsonl"
    index = INDEX_DIR / doc_id / "chroma"

    n_pages = len(list(pages.glob("page_*.png"))) if pages.exists() else 0
    n_layout = len(list(layout.glob("page_*.json"))) if layout.exists() else 0
    n_merged = len(list(merged.glob("page_*.json"))) if merged.exists() else 0
    n_crops = len(list(crops.rglob("*.png"))) if crops.exists() else 0

    if index.exists():
        return "index", "done"
    if chunks.exists():
        return "index", "building chroma..."
    if n_crops:
        return "crop/chunks", f"crops={n_crops}"
    if n_merged:
        return "merge/crop", f"merged={n_merged}/{n_pages or '?'}"
    if n_layout:
        return "layout", f"{n_layout}/{n_pages or '?'}"
    if n_pages:
        return "pdf", f"pages={n_pages}"
    return "starting", ""


def find_active_doc(pilot_df: pd.DataFrame, done: set[str]) -> tuple[str | None, str, str]:
    remaining = pilot_df[~pilot_df["doc_id"].isin(done)]
    if remaining.empty:
        return None, "", ""

    scored: list[tuple[float, str, str, str]] = []
    for _, row in remaining.iterrows():
        doc_id = str(row["doc_id"])
        dirs = [
            PROCESSED / "pages" / doc_id,
            PROCESSED / "layout_json" / doc_id,
            PROCESSED / "layout_json_merged" / doc_id,
            PROCESSED / "crops_merged" / doc_id,
            PROCESSED / "chunks" / doc_id,
        ]
        mt = max((latest_mtime_under(d) for d in dirs), default=0.0)
        if mt > 0:
            stage, detail = infer_stage(doc_id)
            scored.append((mt, doc_id, stage, detail))

    if not scored:
        nxt = str(remaining.iloc[0]["doc_id"])
        return nxt, "queued", ""

    scored.sort(reverse=True)
    _, doc_id, stage, detail = scored[0]
    return doc_id, stage, detail


def batch_process_running() -> bool:
    if sys.platform == "win32":
        ps = (
            "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
            "Where-Object { $_.CommandLine -match 'run_rag_batch|scripts\\\\0[0-9]_|scripts\\\\1[0-2]_|scripts\\\\03_|scripts\\\\06_|scripts\\\\07_' } | "
            "Measure-Object | Select-Object -ExpandProperty Count"
        )
        cmd = ["powershell", "-NoProfile", "-Command", ps]
    else:
        cmd = ["pgrep", "-f", "run_rag_batch|scripts/0[0-9]_|scripts/1[0-2]_"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        out = (r.stdout or "").strip()
        return out not in ("", "0")
    except Exception:
        return False


def format_status() -> str:
    pilot = pd.read_csv(PILOT_CSV)
    total = len(pilot)
    done = indexed_doc_ids() & set(pilot["doc_id"])
    n_done = len(done)
    pct = 100.0 * n_done / total if total else 0.0

    done_df = pilot[pilot["doc_id"].isin(done)]
    done_pages = int(done_df["pages"].sum()) if "pages" in done_df.columns and not done_df.empty else 0
    if "pages" not in pilot.columns:
        pilot = pilot.copy()
        pilot["pages"] = 0

    total_pages = int(pilot["pages"].sum()) if pilot["pages"].notna().any() else 0
    if total_pages == 0:
        try:
            import fitz

            pages_col = []
            for fp in pilot["file_path"]:
                with fitz.open(fp) as doc:
                    pages_col.append(len(doc))
            pilot["pages"] = pages_col
            total_pages = sum(pages_col)
            done_pages = int(pilot[pilot["doc_id"].isin(done)]["pages"].sum())
        except Exception:
            pass

    active, stage, detail = find_active_doc(pilot, done)
    running = batch_process_running()
    ts = datetime.now().strftime("%H:%M:%S")

    lines = [
        f"[{ts}] Pilot 100 batch | {n_done}/{total} docs ({pct:.1f}%) | pages ~{done_pages}/{total_pages}",
        f"  batch process: {'RUNNING' if running else 'NOT RUNNING (may have finished or stopped)'}",
    ]
    if active:
        src = pilot[pilot["doc_id"] == active]["source"].iloc[0]
        short = active if len(active) <= 58 else active[:55] + "..."
        lines.append(f"  active: [{src}] {short}")
        lines.append(f"  stage:  {stage}" + (f" ({detail})" if detail else ""))
    else:
        lines.append("  active: (none - waiting or complete)")

    # last completed (by index folder mtime, pilot docs only)
    idx_times: list[tuple[float, str]] = []
    for doc_id in sorted(done):
        chroma = INDEX_DIR / doc_id / "chroma"
        if chroma.exists():
            idx_times.append((chroma.stat().st_mtime, doc_id))
    if idx_times:
        idx_times.sort(reverse=True)
        last = idx_times[0][1]
        short = last if len(last) <= 60 else last[:57] + "..."
        lines.append(f"  last done: {short}")

    bar_w = 40
    filled = int(bar_w * n_done / total) if total else 0
    bar = "#" * filled + "-" * (bar_w - filled)
    lines.append(f"  [{bar}]")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch pilot-100 batch progress.")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh seconds")
    parser.add_argument("--log-file", type=Path, default=None, help="Also append status to this file")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    args = parser.parse_args()

    log_path = args.log_file or DEFAULT_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        text = format_status()
        if args.log_file is not None or DEFAULT_LOG:
            log_path.write_text(text + "\n", encoding="utf-8")

        if sys.stdout.isatty():
            os.system("cls" if sys.platform == "win32" else "clear")
        print(text)
        print(f"\n(refresh every {args.interval:.0f}s | Ctrl+C to stop watching; batch keeps running)")
        sys.stdout.flush()

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
