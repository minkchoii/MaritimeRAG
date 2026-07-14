"""Export MSC/MEPC rows from pdf_manifest.csv for batch preprocessing."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Export IMO (MSC/MEPC) doc list from manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/pdf_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/imo_docs.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    imo = df[df["source"].isin(["MSC", "MEPC"])].copy()
    if imo.empty:
        raise SystemExit("No MSC/MEPC rows in manifest. Run 00_build_manifest.py first.")

    imo["folder"] = imo["file_path"].astype(str).str.replace("\\", "/").str.extract(
        r"raw_pdfs/([^/]+)/", expand=False
    )
    imo["pilot_status"] = "pending"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imo.to_csv(args.output, index=False, encoding="utf-8-sig")

    msc = int((imo["source"] == "MSC").sum())
    mepc = int((imo["source"] == "MEPC").sum())
    print(f"Wrote {len(imo)} docs -> {args.output} (MSC={msc}, MEPC={mepc})")


if __name__ == "__main__":
    main()
