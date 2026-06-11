from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


FIGURE_TYPES = frozenset({"figure", "picture"})
TABLE_TYPES = frozenset({"table"})
TEXT_TYPES = frozenset({"text"})

SUSPICIOUS_WIDTH_MAX = 40
SUSPICIOUS_HEIGHT_MAX = 20
SUSPICIOUS_AREA_RATIO_MAX = 0.0005
TABLE_AREA_RATIO_MAX = 0.01
TABLE_WIDTH_RATIO_MAX = 0.25
FIGURE_AREA_RATIO_MAX = 0.01
FIGURE_WIDTH_RATIO_MAX = 0.20
PAGE_TOTAL_ELEMENTS_MAX = 40
PAGE_TEXT_ELEMENTS_MAX = 20
PAGE_TABLE_ELEMENTS_MAX = 5
PAGE_FIGURE_ELEMENTS_MAX = 5

MAX_SAMPLES_PER_TYPE = 20


@dataclass
class PageLayoutInfo:
    page_number: int
    page_width: int
    page_height: int
    total_elements: int
    text_count: int
    table_count: int
    figure_count: int
    type_counts: Counter[str] = field(default_factory=Counter)


@dataclass
class EnrichedCrop:
    doc_id: str
    page_number: int
    element_id: str
    element_type: str
    confidence: float
    bbox: list[int]
    crop_path: str
    source_image_path: str
    width: int
    height: int
    area_ratio: float
    width_ratio: float
    suspicious_reasons: list[str] = field(default_factory=list)


def load_layout_pages(layout_doc_dir: Path) -> dict[int, PageLayoutInfo]:
    if not layout_doc_dir.exists():
        raise FileNotFoundError(f"Layout directory not found: {layout_doc_dir}")

    json_files = sorted(layout_doc_dir.glob("page_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No layout JSON files found in: {layout_doc_dir}")

    pages: dict[int, PageLayoutInfo] = {}
    for json_file in json_files:
        page_data = json.loads(json_file.read_text(encoding="utf-8"))
        page_number = int(page_data["page_number"])
        page_width = int(page_data.get("width", 0))
        page_height = int(page_data.get("height", 0))
        type_counts: Counter[str] = Counter()
        for element in page_data.get("elements", []):
            element_type = str(element.get("type", "unknown")).lower()
            type_counts[element_type] += 1

        pages[page_number] = PageLayoutInfo(
            page_number=page_number,
            page_width=page_width,
            page_height=page_height,
            total_elements=sum(type_counts.values()),
            text_count=sum(type_counts[t] for t in type_counts if t in TEXT_TYPES),
            table_count=sum(type_counts[t] for t in type_counts if t in TABLE_TYPES),
            figure_count=sum(type_counts[t] for t in type_counts if t in FIGURE_TYPES),
            type_counts=type_counts,
        )
    return pages


def bbox_metrics(
    bbox: list[float | int],
    page_width: int,
    page_height: int,
) -> tuple[int, int, float, float]:
    x1, y1, x2, y2 = bbox
    width = max(0, int(round(x2)) - int(round(x1)))
    height = max(0, int(round(y2)) - int(round(y1)))
    page_area = max(page_width * page_height, 1)
    area_ratio = (width * height) / page_area
    width_ratio = width / max(page_width, 1)
    return width, height, area_ratio, width_ratio


def load_layout_elements(doc_id: str, layout_doc_dir: Path, pages: dict[int, PageLayoutInfo]) -> list[EnrichedCrop]:
    crops: list[EnrichedCrop] = []
    for json_file in sorted(layout_doc_dir.glob("page_*.json")):
        page_data = json.loads(json_file.read_text(encoding="utf-8"))
        page_number = int(page_data["page_number"])
        page_info = pages[page_number]
        for element in page_data.get("elements", []):
            bbox = element.get("bbox", [0, 0, 0, 0])
            width, height, area_ratio, width_ratio = bbox_metrics(
                bbox,
                page_info.page_width,
                page_info.page_height,
            )
            crops.append(
                EnrichedCrop(
                    doc_id=doc_id,
                    page_number=page_number,
                    element_id=str(element.get("element_id", "")),
                    element_type=str(element.get("type", "unknown")).lower(),
                    confidence=float(element.get("confidence", 0.0)),
                    bbox=[int(v) for v in bbox],
                    crop_path="",
                    source_image_path=str(page_data.get("image_path", "")),
                    width=width,
                    height=height,
                    area_ratio=area_ratio,
                    width_ratio=width_ratio,
                )
            )
    return crops


def load_crops_manifest(manifest_path: Path, pages: dict[int, PageLayoutInfo]) -> list[EnrichedCrop]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Crops manifest not found: {manifest_path}")

    crops: list[EnrichedCrop] = []
    with manifest_path.open(encoding="utf-8") as manifest_f:
        for line_no, line in enumerate(manifest_f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            page_number = int(record["page_number"])
            page_info = pages.get(page_number)
            if page_info is None:
                raise KeyError(
                    f"Page {page_number} in manifest (line {line_no}) has no matching layout JSON."
                )

            bbox = record.get("bbox", [0, 0, 0, 0])
            width, height, area_ratio, width_ratio = bbox_metrics(
                bbox,
                page_info.page_width,
                page_info.page_height,
            )
            crops.append(
                EnrichedCrop(
                    doc_id=str(record.get("doc_id", "")),
                    page_number=page_number,
                    element_id=str(record.get("element_id", "")),
                    element_type=str(record.get("element_type", "unknown")).lower(),
                    confidence=float(record.get("confidence", 0.0)),
                    bbox=[int(v) for v in bbox],
                    crop_path=str(record.get("crop_path", "")),
                    source_image_path=str(record.get("source_image_path", "")),
                    width=width,
                    height=height,
                    area_ratio=area_ratio,
                    width_ratio=width_ratio,
                )
            )
    return crops


def crop_level_reasons(crop: EnrichedCrop) -> list[str]:
    reasons: list[str] = []
    if crop.width < SUSPICIOUS_WIDTH_MAX:
        reasons.append(f"width_lt_{SUSPICIOUS_WIDTH_MAX}")
    if crop.height < SUSPICIOUS_HEIGHT_MAX:
        reasons.append(f"height_lt_{SUSPICIOUS_HEIGHT_MAX}")
    if crop.area_ratio < SUSPICIOUS_AREA_RATIO_MAX:
        reasons.append(f"area_ratio_lt_{SUSPICIOUS_AREA_RATIO_MAX}")
    if crop.element_type in TABLE_TYPES:
        if crop.area_ratio < TABLE_AREA_RATIO_MAX:
            reasons.append(f"table_area_ratio_lt_{TABLE_AREA_RATIO_MAX}")
        if crop.width_ratio < TABLE_WIDTH_RATIO_MAX:
            reasons.append(f"table_width_ratio_lt_{TABLE_WIDTH_RATIO_MAX}")
    if crop.element_type in FIGURE_TYPES:
        if crop.area_ratio < FIGURE_AREA_RATIO_MAX:
            reasons.append(f"figure_area_ratio_lt_{FIGURE_AREA_RATIO_MAX}")
        if crop.width_ratio < FIGURE_WIDTH_RATIO_MAX:
            reasons.append(f"figure_width_ratio_lt_{FIGURE_WIDTH_RATIO_MAX}")
    return reasons


def page_level_reasons(page_info: PageLayoutInfo) -> list[str]:
    reasons: list[str] = []
    if page_info.total_elements > PAGE_TOTAL_ELEMENTS_MAX:
        reasons.append(f"page_total_elements_gt_{PAGE_TOTAL_ELEMENTS_MAX}")
    if page_info.text_count > PAGE_TEXT_ELEMENTS_MAX:
        reasons.append(f"page_text_elements_gt_{PAGE_TEXT_ELEMENTS_MAX}")
    if page_info.table_count > PAGE_TABLE_ELEMENTS_MAX:
        reasons.append(f"page_table_elements_gt_{PAGE_TABLE_ELEMENTS_MAX}")
    if page_info.figure_count > PAGE_FIGURE_ELEMENTS_MAX:
        reasons.append(f"page_figure_elements_gt_{PAGE_FIGURE_ELEMENTS_MAX}")
    return reasons


def mark_suspicious_crops(crops: list[EnrichedCrop], pages: dict[int, PageLayoutInfo]) -> list[EnrichedCrop]:
    page_reason_cache: dict[int, list[str]] = {}
    suspicious: list[EnrichedCrop] = []

    for crop in crops:
        reasons = crop_level_reasons(crop)
        if crop.page_number not in page_reason_cache:
            page_info = pages[crop.page_number]
            page_reason_cache[crop.page_number] = page_level_reasons(page_info)
        reasons.extend(page_reason_cache[crop.page_number])
        if reasons:
            crop.suspicious_reasons = sorted(set(reasons))
            suspicious.append(crop)
    return suspicious


def summarize_numeric(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def format_stats_block(title: str, stats: dict[str, float | int]) -> str:
    if stats["count"] == 0:
        return f"{title}: (no data)\n"
    return (
        f"{title}:\n"
        f"  count={stats['count']}, min={stats['min']:.6f}, max={stats['max']:.6f}, "
        f"mean={stats['mean']:.6f}, median={stats['median']:.6f}\n"
    )


def build_report(
    doc_id: str,
    crops: list[EnrichedCrop],
    pages: dict[int, PageLayoutInfo],
    suspicious: list[EnrichedCrop],
) -> str:
    lines: list[str] = []
    lines.append(f"Crop Quality Report: {doc_id}")
    lines.append("=" * 60)
    lines.append("")

    type_counter = Counter(crop.element_type for crop in crops)
    lines.append(f"Total crops: {len(crops)}")
    lines.append("Crops by element_type:")
    for element_type, count in sorted(type_counter.items()):
        lines.append(f"  - {element_type}: {count}")
    lines.append("")

    crops_per_page = Counter(crop.page_number for crop in crops)
    page_counts = list(crops_per_page.values())
    if page_counts:
        lines.append(
            f"Crops per page: mean={statistics.mean(page_counts):.2f}, "
            f"max={max(page_counts)} (page {max(crops_per_page, key=crops_per_page.get)})"
        )
    else:
        lines.append("Crops per page: (no crops)")
    lines.append("")

    lines.append("Layout page element counts (from layout JSON):")
    layout_totals = [p.total_elements for p in pages.values()]
    if layout_totals:
        lines.append(
            f"  elements per page: mean={statistics.mean(layout_totals):.2f}, max={max(layout_totals)}"
        )
    lines.append("")

    lines.append("Metrics by element_type:")
    by_type: dict[str, list[EnrichedCrop]] = defaultdict(list)
    for crop in crops:
        by_type[crop.element_type].append(crop)

    for element_type in sorted(by_type):
        group = by_type[element_type]
        widths = [float(c.width) for c in group]
        heights = [float(c.height) for c in group]
        area_ratios = [c.area_ratio for c in group]
        lines.append(f"  [{element_type}]")
        lines.append(
            "    "
            + format_stats_block("width", summarize_numeric(widths)).strip().replace("\n", "\n    ")
        )
        lines.append(
            "    "
            + format_stats_block("height", summarize_numeric(heights)).strip().replace("\n", "\n    ")
        )
        lines.append(
            "    "
            + format_stats_block("area_ratio", summarize_numeric(area_ratios)).strip().replace("\n", "\n    ")
        )
    lines.append("")

    lines.append(f"Suspicious crops: {len(suspicious)}")
    reason_counter = Counter()
    for crop in suspicious:
        for reason in crop.suspicious_reasons:
            reason_counter[reason] += 1
    lines.append("Suspicious reason counts:")
    for reason, count in sorted(reason_counter.items()):
        lines.append(f"  - {reason}: {count}")
    lines.append("")

    lines.append("Quality thresholds applied:")
    lines.append(f"  - width < {SUSPICIOUS_WIDTH_MAX}")
    lines.append(f"  - height < {SUSPICIOUS_HEIGHT_MAX}")
    lines.append(f"  - area_ratio < {SUSPICIOUS_AREA_RATIO_MAX}")
    lines.append(f"  - table area_ratio < {TABLE_AREA_RATIO_MAX}")
    lines.append(f"  - table width_ratio < {TABLE_WIDTH_RATIO_MAX}")
    lines.append(f"  - figure/picture area_ratio < {FIGURE_AREA_RATIO_MAX}")
    lines.append(f"  - figure/picture width_ratio < {FIGURE_WIDTH_RATIO_MAX}")
    lines.append(f"  - page total elements > {PAGE_TOTAL_ELEMENTS_MAX}")
    lines.append(f"  - page text elements > {PAGE_TEXT_ELEMENTS_MAX}")
    lines.append(f"  - page table elements > {PAGE_TABLE_ELEMENTS_MAX}")
    lines.append(f"  - page figure/picture elements > {PAGE_FIGURE_ELEMENTS_MAX}")

    return "\n".join(lines) + "\n"


def save_suspicious_csv(suspicious: list[EnrichedCrop], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "doc_id",
        "page_number",
        "element_id",
        "element_type",
        "width",
        "height",
        "area_ratio",
        "width_ratio",
        "confidence",
        "crop_path",
        "suspicious_reasons",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()
        for crop in suspicious:
            writer.writerow(
                {
                    "doc_id": crop.doc_id,
                    "page_number": crop.page_number,
                    "element_id": crop.element_id,
                    "element_type": crop.element_type,
                    "width": crop.width,
                    "height": crop.height,
                    "area_ratio": f"{crop.area_ratio:.8f}",
                    "width_ratio": f"{crop.width_ratio:.8f}",
                    "confidence": crop.confidence,
                    "crop_path": crop.crop_path,
                    "suspicious_reasons": ";".join(crop.suspicious_reasons),
                }
            )


def resolve_crop_path(crop: EnrichedCrop, crops_doc_dir: Path) -> Path | None:
    recorded = Path(crop.crop_path)
    if recorded.exists():
        return recorded
    local = crops_doc_dir / crop.element_type / recorded.name
    if local.exists():
        return local
    return None


def copy_quality_samples(
    suspicious: list[EnrichedCrop],
    samples_root: Path,
    crops_doc_dir: Path,
) -> int:
    samples_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    by_type: dict[str, list[EnrichedCrop]] = defaultdict(list)
    for crop in suspicious:
        by_type[crop.element_type].append(crop)

    for element_type, group in sorted(by_type.items()):
        out_dir = samples_root / element_type
        out_dir.mkdir(parents=True, exist_ok=True)
        seen_names: set[str] = set()
        for crop in group:
            if len(seen_names) >= MAX_SAMPLES_PER_TYPE:
                break
            src = resolve_crop_path(crop, crops_doc_dir)
            if src is None:
                continue
            dest_name = src.name
            if dest_name in seen_names:
                continue
            dest = out_dir / dest_name
            shutil.copy2(src, dest)
            seen_names.add(dest_name)
            copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze crop quality and export suspicious cases.")
    parser.add_argument("--doc-id", type=str, required=True, help="Document ID")
    parser.add_argument(
        "--crops-dir",
        type=Path,
        default=Path("data/processed/crops"),
        help="Root directory for crop outputs",
    )
    parser.add_argument(
        "--layout-dir",
        type=Path,
        default=Path("data/processed/layout_json"),
        help="Root directory for layout JSON outputs",
    )
    parser.add_argument(
        "--layout-only",
        action="store_true",
        help="Analyze layout JSON only (skip crops manifest; no crop CSV/samples)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("data/processed/logs"),
        help="Directory for quality report and suspicious CSV",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("outputs/quality_samples"),
        help="Directory for suspicious crop sample images",
    )
    args = parser.parse_args()

    doc_id = args.doc_id
    layout_doc_dir = args.layout_dir / doc_id
    if args.layout_only:
        suffix = "_layout_merged" if "merged" in args.layout_dir.as_posix() else "_layout_before"
    else:
        suffix = ""
    report_path = args.logs_dir / f"{doc_id}_crop_quality_report{suffix}.txt"
    suspicious_csv_path = args.logs_dir / f"{doc_id}_suspicious_crops{suffix}.csv"
    samples_dir = args.samples_dir / doc_id

    pages = load_layout_pages(layout_doc_dir)
    if args.layout_only:
        crops = load_layout_elements(doc_id, layout_doc_dir, pages)
    else:
        manifest_path = args.crops_dir / doc_id / "crops_manifest.jsonl"
        crops = load_crops_manifest(manifest_path, pages)
    suspicious = mark_suspicious_crops(crops, pages)

    report_title = doc_id + (" (layout JSON)" if args.layout_only else "")
    report_text = build_report(report_title, crops, pages, suspicious)
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    save_suspicious_csv(suspicious, suspicious_csv_path)

    copied_samples = 0
    if not args.layout_only:
        crops_doc_dir = args.crops_dir / doc_id
        copied_samples = copy_quality_samples(suspicious, samples_dir, crops_doc_dir)

    print(report_text, end="")
    print(f"Report saved: {report_path}")
    print(f"Suspicious entries CSV saved: {suspicious_csv_path} ({len(suspicious)} rows)")
    if args.layout_only:
        print("(layout-only mode: sample images skipped)")
    else:
        print(f"Sample images copied: {copied_samples} -> {samples_dir}")


if __name__ == "__main__":
    main()
