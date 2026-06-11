from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont


TYPE_COLORS: dict[str, tuple[int, int, int]] = {
    "text": (34, 197, 94),
    "title": (249, 115, 22),
    "table": (59, 130, 246),
    "figure": (239, 68, 68),
    "picture": (239, 68, 68),
    "caption": (168, 85, 247),
    "list": (16, 185, 129),
    "list-item": (16, 185, 129),
    "section-header": (234, 179, 8),
    "page-header": (156, 163, 175),
    "page-footer": (107, 114, 128),
}
DEFAULT_COLOR = (236, 72, 153)
MERGED_TEXT_COLOR = (22, 163, 74)
MERGED_TEXT_WIDTH = 4
DEFAULT_WIDTH = 2


def parse_page_numbers(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    pages: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        pages.add(int(part))
    return pages


def element_label(element: dict) -> str:
    element_type = str(element.get("type", "unknown")).lower()
    merged_from = element.get("merged_from")
    if merged_from:
        return f"{element_type} (merged x{len(merged_from)})"
    return element_type


def draw_page(
    image_path: Path,
    elements: list[dict],
    out_path: Path,
    title: str,
) -> None:
    with Image.open(image_path).convert("RGB") as img:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            title_font = ImageFont.truetype("arial.ttf", 22)
        except OSError:
            font = ImageFont.load_default()
            title_font = font

        draw.rectangle((0, 0, img.width, 36), fill=(15, 23, 42))
        draw.text((12, 8), title, fill=(248, 250, 252), font=title_font)

        for element in elements:
            bbox = element.get("bbox", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            element_type = str(element.get("type", "unknown")).lower()
            merged_from = element.get("merged_from")
            if merged_from:
                color = MERGED_TEXT_COLOR
                width = MERGED_TEXT_WIDTH
            else:
                color = TYPE_COLORS.get(element_type, DEFAULT_COLOR)
                width = DEFAULT_WIDTH

            draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
            label = element_label(element)
            text_y = max(38, y1 - 18)
            draw.text((x1, text_y), label, fill=color, font=font)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw layout JSON bboxes on page images.")
    parser.add_argument("--doc-id", type=str, required=True)
    parser.add_argument(
        "--layout-dir",
        type=Path,
        default=None,
        help="Layout JSON directory (default: data/processed/layout_json/<doc_id>)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Page image directory (default: data/processed/pages/<doc_id>)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/visualized_layout"),
        help="Visualization output root",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Comma-separated page numbers (default: all pages in layout dir)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Subfolder tag under out-dir (e.g. merged, before)",
    )
    args = parser.parse_args()

    layout_doc_dir = args.layout_dir or Path("data/processed/layout_json") / args.doc_id
    image_dir = args.image_dir or Path("data/processed/pages") / args.doc_id
    out_doc_dir = args.out_dir / args.doc_id
    if args.tag:
        out_doc_dir = out_doc_dir / args.tag

    page_filter = parse_page_numbers(args.pages)
    json_files = sorted(layout_doc_dir.glob("page_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No layout JSON files in: {layout_doc_dir}")

    saved = 0
    for json_file in json_files:
        page_data = json.loads(json_file.read_text(encoding="utf-8"))
        page_number = int(page_data["page_number"])
        if page_filter is not None and page_number not in page_filter:
            continue

        page_name = page_data.get("page_name", f"page_{page_number:04d}.png")
        image_path = image_dir / page_name
        if not image_path.exists():
            alt = Path(page_data.get("image_path", ""))
            if alt.exists():
                image_path = alt
            else:
                print(f"[WARN] Missing image: {image_path}")
                continue

        title = f"{args.doc_id} p{page_number:04d} | {layout_doc_dir.name} | {len(page_data.get('elements', []))} elements"
        out_path = out_doc_dir / f"page_{page_number:04d}.png"
        draw_page(image_path, page_data.get("elements", []), out_path, title)
        saved += 1

    print(f"Saved {saved} visualization(s) to: {out_doc_dir}")


if __name__ == "__main__":
    main()
