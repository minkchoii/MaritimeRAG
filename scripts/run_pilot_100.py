"""
Orchestrate the 100-PDF pilot: sample list -> per-doc pipeline -> unified index.

Typical workflow:
  python scripts/run_pilot_100.py --phase sample
  python scripts/run_pilot_100.py --phase preprocess --max-docs 5   # trial
  python scripts/run_pilot_100.py --phase preprocess                  # all 100 (long)
  python scripts/run_pilot_100.py --phase index
  python scripts/rag_query.py --unified pilot_100 --source KR -i --full-text --top-k 3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run(cmd: list[str], cwd: Path) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="100-PDF pilot orchestrator.")
    parser.add_argument(
        "--phase",
        choices=("sample", "preprocess", "index", "all"),
        required=True,
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--doc-list", type=Path, default=Path("data/manifests/pilot_100_docs.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/pdf_manifest.csv"))
    parser.add_argument("--collection-id", type=str, default="pilot_100")
    parser.add_argument("--max-docs", type=int, default=None, help="Limit docs for preprocess trial")
    parser.add_argument(
        "--steps",
        type=str,
        default="pdf,layout,merge,crop,chunks,quality,index",
        help="Passed to run_rag_batch (exclude index if only preprocess phase)",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    root = Path.cwd()
    py = args.python

    if args.phase in ("sample", "all"):
        run([py, "scripts/00_build_manifest.py"], root)
        run([py, "scripts/14_select_pilot_100.py"], root)

    if args.phase in ("preprocess", "all"):
        if not args.doc_list.exists():
            raise FileNotFoundError(f"Run sample phase first: {args.doc_list}")
        doc_list = args.doc_list
        if args.max_docs:
            df = pd.read_csv(doc_list).head(args.max_docs)
            limited = root / "data/manifests/pilot_100_docs_limited.csv"
            df.to_csv(limited, index=False, encoding="utf-8-sig")
            doc_list = limited
            print(f"Limited to {len(df)} docs -> {limited}")

        steps = args.steps
        if args.phase == "all" and "index" in steps:
            steps = ",".join(s for s in steps.split(",") if s.strip() != "index")

        run(
            [
                py,
                "scripts/run_rag_batch.py",
                "--doc-list",
                str(doc_list),
                "--steps",
                steps,
                "--dpi",
                str(args.dpi),
            ],
            root,
        )

    if args.phase in ("index", "all"):
        run(
            [
                py,
                "scripts/10_build_unified_index.py",
                "--doc-list",
                str(args.doc_list if not args.max_docs else root / "data/manifests/pilot_100_docs_limited.csv"),
                "--collection-id",
                args.collection_id,
            ],
            root,
        )

    print("\nPilot phase completed:", args.phase)
    if args.phase in ("index", "all"):
        print(
            f"Query: python scripts/rag_query.py --unified {args.collection_id} "
            f"--source KR -i --full-text --top-k 3"
        )


if __name__ == "__main__":
    main()
