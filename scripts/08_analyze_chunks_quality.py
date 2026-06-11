from __future__ import annotations

import argparse
import csv
import json

from clause_parse import article_number_from_text, is_article_clause_number
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


MIN_TEXT_CHARS = 10
SHORT_HEADER_TYPES = frozenset({"section-header", "title"})
SHORT_TYPE_LEN_THRESHOLD = 80
FIGURE_TYPES = frozenset({"picture", "figure"})
TABLE_TYPES = frozenset({"table"})
PICTURE_PLACEHOLDER_MARKERS = (
    "[picture element",
    "[figure element",
    "refer to crop image",
)

HEADER_FOOTER_KEYWORDS = (
    "page-header",
    "page-footer",
    "1편 1장",
    "1편 2장",
    "1편 선급등록",
    "선급등록 및 검사 /",
    "KR 선급 및 강선규칙",
)

MAX_SAMPLES_PER_TYPE = 20
TEXT_PREVIEW_LEN = 200


@dataclass
class AnalyzedChunk:
    record: dict
    text: str
    text_len: int
    element_type: str
    suspicious_reasons: list[str] = field(default_factory=list)

    @property
    def chunk_id(self) -> str:
        return str(self.record.get("chunk_id", ""))


def load_chunks(chunks_path: Path) -> list[dict]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    chunks: list[dict] = []
    with chunks_path.open(encoding="utf-8") as chunks_f:
        for line_no, line in enumerate(chunks_f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
    return chunks


def text_length(chunk: dict) -> int:
    declared = chunk.get("text_char_count")
    if isinstance(declared, int):
        return declared
    return len(str(chunk.get("text", "")).strip())


def get_text(chunk: dict) -> str:
    return str(chunk.get("text", "")).strip()


def is_picture_placeholder(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in PICTURE_PLACEHOLDER_MARKERS)


def has_caption_signal(chunk: dict, text: str) -> bool:
    if chunk.get("linked_caption_id"):
        return True
    if chunk.get("caption_text"):
        return True
    if "그림" in text or "Figure" in text or "figure" in text.lower():
        return True
    return False


def keyword_hits(text: str, keywords: tuple[str, ...]) -> list[str]:
    hits: list[str] = []
    lower = text.lower()
    for kw in keywords:
        if kw.lower() in lower:
            hits.append(kw)
    return hits


def analyze_chunk(chunk: dict) -> AnalyzedChunk:
    element_type = str(chunk.get("element_type", "unknown")).lower()
    text = get_text(chunk)
    text_len = len(text) if text else text_length(chunk)
    reasons: list[str] = []

    if not text:
        reasons.append("empty_text")
    elif text_len < MIN_TEXT_CHARS:
        reasons.append(f"text_lt_{MIN_TEXT_CHARS}")

    header_hits = keyword_hits(text, HEADER_FOOTER_KEYWORDS)
    if header_hits:
        article = article_number_from_text(text) or str(
            chunk.get("article_number") or chunk.get("clause_number") or ""
        )
        substantive = is_article_clause_number(article) and text_len >= 80
        if not substantive:
            reasons.append("header_footer_keyword")

    if element_type in FIGURE_TYPES:
        if is_picture_placeholder(text) and not has_caption_signal(chunk, text):
            reasons.append("picture_no_caption")
        elif is_picture_placeholder(text):
            reasons.append("picture_placeholder_only")

    return AnalyzedChunk(
        record=chunk,
        text=text,
        text_len=text_len,
        element_type=element_type,
        suspicious_reasons=reasons,
    )


def length_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def format_stats_block(label: str, stats: dict[str, float | int]) -> str:
    if stats["count"] == 0:
        return f"{label}: (no data)\n"
    return (
        f"{label}: count={stats['count']}, min={stats['min']}, max={stats['max']}, "
        f"mean={stats['mean']:.1f}, median={stats['median']:.1f}\n"
    )


def build_report(
    doc_id: str,
    analyzed: list[AnalyzedChunk],
    type_counter: Counter[str],
    suspicious: list[AnalyzedChunk],
) -> str:
    lines = [
        f"Chunk Quality Report: {doc_id}",
        "=" * 60,
        "",
        f"Total chunks: {len(analyzed)}",
        "",
        "Chunks by element_type:",
    ]
    for element_type, count in sorted(type_counter.items()):
        lines.append(f"  - {element_type}: {count}")

    lines.extend(["", "Text length by element_type:"])
    by_type: dict[str, list[int]] = defaultdict(list)
    for item in analyzed:
        by_type[item.element_type].append(item.text_len)
    for element_type in sorted(by_type):
        lines.append(f"  [{element_type}]")
        lines.append("    " + format_stats_block("text_chars", length_stats(by_type[element_type])).rstrip())

    short_type_stats: dict[str, dict[str, int]] = {}
    for element_type in SHORT_HEADER_TYPES:
        subset = [a for a in analyzed if a.element_type == element_type]
        if not subset:
            continue
        short_count = sum(1 for a in subset if a.text_len < SHORT_TYPE_LEN_THRESHOLD)
        short_type_stats[element_type] = {
            "total": len(subset),
            "under_threshold": short_count,
            "threshold": SHORT_TYPE_LEN_THRESHOLD,
        }

    lines.extend(["", f"Short chunks ({', '.join(sorted(SHORT_HEADER_TYPES))}):"])
    if short_type_stats:
        for element_type, stats in sorted(short_type_stats.items()):
            lines.append(
                f"  - {element_type}: {stats['under_threshold']}/{stats['total']} "
                f"under {stats['threshold']} chars"
            )
    else:
        lines.append("  (none)")

    header_footer_hits = [a for a in analyzed if "header_footer_keyword" in a.suspicious_reasons]
    lines.extend(
        [
            "",
            f"Chunks with header/footer keywords: {len(header_footer_hits)}",
        ]
    )
    if header_footer_hits:
        for item in header_footer_hits[:15]:
            lines.append(f"  - {item.chunk_id} (p{item.record.get('page_number')})")
        if len(header_footer_hits) > 15:
            lines.append(f"  ... and {len(header_footer_hits) - 15} more")

    table_chunks = [a for a in analyzed if a.element_type in TABLE_TYPES]
    lines.extend(["", "Table chunks:"])
    if table_chunks:
        formats = Counter(str(a.record.get("content_format", "unknown")) for a in table_chunks)
        table_lengths = [a.text_len for a in table_chunks]
        lines.append(f"  count: {len(table_chunks)}")
        lines.append("  content_format:")
        for fmt, count in sorted(formats.items()):
            lines.append(f"    - {fmt}: {count}")
        lines.append("  " + format_stats_block("text_chars", length_stats(table_lengths)).strip())
    else:
        lines.append("  (none)")

    figure_chunks = [a for a in analyzed if a.element_type in FIGURE_TYPES]
    with_linked = sum(1 for a in figure_chunks if a.record.get("linked_caption_id"))
    with_caption_text_field = sum(1 for a in figure_chunks if a.record.get("caption_text"))
    with_caption_in_body = sum(
        1 for a in figure_chunks if has_caption_signal(a.record, a.text) and not is_picture_placeholder(a.text)
    )
    lines.extend(
        [
            "",
            "Picture/figure chunks:",
            f"  total: {len(figure_chunks)}",
            f"  with linked_caption_id: {with_linked}",
            f"  with caption_text field: {with_caption_text_field}",
            f"  with caption-like text (non-placeholder): {with_caption_in_body}",
        ]
    )

    lines.extend(
        [
            "",
            f"Suspicious chunks: {len(suspicious)}",
            "Suspicious reason counts:",
        ]
    )
    reason_counter: Counter[str] = Counter()
    for item in suspicious:
        for reason in item.suspicious_reasons:
            reason_counter[reason] += 1
    for reason, count in sorted(reason_counter.items()):
        lines.append(f"  - {reason}: {count}")

    lines.extend(
        [
            "",
            "Thresholds:",
            f"  - suspicious if text empty or < {MIN_TEXT_CHARS} chars",
            f"  - short {', '.join(sorted(SHORT_HEADER_TYPES))} tracked if < {SHORT_TYPE_LEN_THRESHOLD} chars",
            f"  - header/footer keywords: {', '.join(HEADER_FOOTER_KEYWORDS[:4])} ...",
        ]
    )
    return "\n".join(lines) + "\n"


def save_suspicious_csv(suspicious: list[AnalyzedChunk], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "chunk_id",
        "element_type",
        "page_number",
        "text_char_count",
        "content_format",
        "linked_caption_id",
        "suspicious_reasons",
        "text_preview",
        "crop_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()
        for item in suspicious:
            preview = item.text.replace("\n", " ")[:TEXT_PREVIEW_LEN]
            writer.writerow(
                {
                    "chunk_id": item.chunk_id,
                    "element_type": item.element_type,
                    "page_number": item.record.get("page_number"),
                    "text_char_count": item.text_len,
                    "content_format": item.record.get("content_format", ""),
                    "linked_caption_id": item.record.get("linked_caption_id", ""),
                    "suspicious_reasons": ";".join(item.suspicious_reasons),
                    "text_preview": preview,
                    "crop_path": item.record.get("crop_path", ""),
                }
            )


def save_samples(analyzed: list[AnalyzedChunk], samples_path: Path) -> None:
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    by_type: dict[str, list[AnalyzedChunk]] = defaultdict(list)
    for item in analyzed:
        by_type[item.element_type].append(item)

    lines = [f"Chunk Samples (max {MAX_SAMPLES_PER_TYPE} per type)", "=" * 60, ""]
    for element_type in sorted(by_type):
        lines.append(f"## {element_type} ({len(by_type[element_type])} total)")
        for item in by_type[element_type][:MAX_SAMPLES_PER_TYPE]:
            preview = item.text[:TEXT_PREVIEW_LEN].replace("\n", "\\n")
            if len(item.text) > TEXT_PREVIEW_LEN:
                preview += "..."
            lines.append(
                f"- {item.chunk_id} | p{item.record.get('page_number')} | "
                f"len={item.text_len} | {preview}"
            )
        lines.append("")

    samples_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze chunk quality before RAG embedding.")
    parser.add_argument("--doc-id", type=str, required=True)
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("data/processed/chunks"),
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("data/processed/logs"),
    )
    args = parser.parse_args()

    chunks_path = args.chunks_dir / args.doc_id / "chunks.jsonl"
    report_path = args.logs_dir / f"{args.doc_id}_chunk_quality_report.txt"
    suspicious_csv_path = args.logs_dir / f"{args.doc_id}_suspicious_chunks.csv"
    samples_path = args.logs_dir / f"{args.doc_id}_chunk_samples.txt"

    raw_chunks = load_chunks(chunks_path)
    analyzed = [analyze_chunk(chunk) for chunk in raw_chunks]
    type_counter = Counter(a.element_type for a in analyzed)
    suspicious = [a for a in analyzed if a.suspicious_reasons]

    args.logs_dir.mkdir(parents=True, exist_ok=True)
    report_text = build_report(args.doc_id, analyzed, type_counter, suspicious)
    report_path.write_text(report_text, encoding="utf-8")
    save_suspicious_csv(suspicious, suspicious_csv_path)
    save_samples(analyzed, samples_path)

    print(report_text, end="")
    print(f"Report saved: {report_path}")
    print(f"Suspicious CSV saved: {suspicious_csv_path} ({len(suspicious)} rows)")
    print(f"Samples saved: {samples_path}")


if __name__ == "__main__":
    main()
