"""
Stratified sample of ~100 PDFs from raw_pdfs subfolders for the pilot batch.

Output: data/manifests/pilot_100_docs.csv (doc_id, source, folder, file_path, ...)

Sampling (default total=100):
  - ABS rules: all available (~14)
  - LR Rules: all available (~1)
  - KR Rules: up to 25
  - dnv-class_2026-04: remainder to reach total
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FOLDER_SOURCE = {
    "abs rules": "ABS",
    "kr rules": "KR",
    "dnv-class_2026-04": "DNV",
    "lr rules": "LR",
}


def folder_key(file_path: str) -> str:
    p = Path(file_path.replace("\\", "/"))
    parts = [x.lower() for x in p.parts]
    try:
        idx = parts.index("raw_pdfs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    except ValueError:
        pass
    return "unknown"


def stratified_sample(df: pd.DataFrame, total: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["folder"] = df["file_path"].map(folder_key)

    # Fixed caps first; DNV fills the remainder (do not take all DNV in the first pass).
    caps = {
        "abs rules": None,  # all
        "lr rules": None,
        "kr rules": 25,
    }

    selected: list[pd.DataFrame] = []
    used_ids: set[str] = set()

    for folder, cap in caps.items():
        sub = df[df["folder"] == folder].sort_values("file_name")
        if sub.empty:
            continue
        pick = sub if cap is None else sub.head(cap)
        selected.append(pick)
        used_ids.update(pick["doc_id"].tolist())

    merged = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    remaining = total - len(merged)

    if remaining > 0:
        dnv = df[(df["folder"] == "dnv-class_2026-04") & (~df["doc_id"].isin(used_ids))]
        dnv = dnv.sort_values("file_name")
        if len(dnv) > remaining:
            dnv = dnv.sample(n=remaining, random_state=seed)
        if not dnv.empty:
            merged = pd.concat([merged, dnv], ignore_index=True)

    if len(merged) > total:
        merged = merged.sort_values(["folder", "file_name"]).head(total)

    return merged.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select stratified pilot PDF list (~100).")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/pdf_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/pilot_100_docs.csv"))
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-kr-1-2025", action="store_true", default=True)
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Run 00_build_manifest.py first: {args.manifest}")

    df = pd.read_csv(args.manifest)
    pilot = stratified_sample(df, args.total, args.seed)

    if args.include_kr_1_2025 and "kr_1_2025" in df["doc_id"].values:
        if "kr_1_2025" not in pilot["doc_id"].values:
            row = df[df["doc_id"] == "kr_1_2025"].iloc[0:1]
            pilot = pd.concat([row.assign(folder=folder_key(str(row.iloc[0]["file_path"]))), pilot], ignore_index=True)
            if len(pilot) > args.total:
                pilot = pilot.drop_duplicates(subset=["doc_id"], keep="first").head(args.total)

    pilot["folder"] = pilot["file_path"].map(folder_key)
    pilot["pilot_status"] = "pending"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pilot.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(pilot)} docs -> {args.output}")
    print(pilot.groupby(["source", "folder"]).size().to_string())
    print(f"doc_ids sample: {pilot['doc_id'].head(5).tolist()} ...")


if __name__ == "__main__":
    main()
