"""Rule/Guidance lookup: richer context, doc manifest, hallucination checks."""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

DOC_CODE_CROSSREF_RE = re.compile(
    r"document\s+code.*title|document\s+code\s+document\s+code",
    re.I,
)
METADATA_LINE_RE = re.compile(r"^\[(dnv|abs|kr|lr|msc|mepc|figure|table)\]", re.I)
RULE_DOC_CODE_RE = re.compile(
    r"\b((?:DNV|LR|ABS|KR)[- ](?:CG|RP|RU|CP|NV)[- ]?[A-Za-z0-9.\-]+(?:\s+Pt\.?\d+)?(?:\s+Ch\.?\d+)?)",
    re.I,
)
PLACEHOLDER_RE = re.compile(r"\(context의\s*고유\s*주제\)|\(unique topic", re.I)


def is_crossref_table_chunk(chunk: Any) -> bool:
    """Bibliography-style tables listing Document code / Title only."""
    text = f"{getattr(chunk, 'caption', '')} {getattr(chunk, 'text', '')}"
    if str(getattr(chunk, "chunk_type", "")).lower() == "table" and DOC_CODE_CROSSREF_RE.search(text):
        return True
    return False


def strip_metadata_prefix(text: str) -> str:
    lines = (text or "").splitlines()
    while lines and METADATA_LINE_RE.match(lines[0].strip()):
        lines = lines[1:]
    return "\n".join(lines).strip()


def chunk_body_len(chunk: Any) -> int:
    return len(strip_metadata_prefix(getattr(chunk, "text", "")))


def allowed_file_names(chunks: list[Any]) -> set[str]:
    return {str(c.file_name) for c in chunks if getattr(c, "file_name", "")}


def doc_code_in_corpus(code: str, file_names: set[str]) -> bool:
    norm = re.sub(r"\s+", "", code.upper())
    for fn in file_names:
        fn_norm = re.sub(r"\s+", "", fn.upper())
        if norm in fn_norm or fn_norm in norm:
            return True
        stem = re.sub(r"\.PDF$", "", fn_norm, flags=re.I)
        if norm.replace("-", "") in stem.replace("-", ""):
            return True
    return False


def detect_hallucinated_doc_codes(answer: str, file_names: set[str]) -> list[str]:
    if not answer or not file_names:
        return []
    bad: list[str] = []
    for m in RULE_DOC_CODE_RE.finditer(answer):
        code = m.group(1).strip()
        if not doc_code_in_corpus(code, file_names):
            bad.append(code)
    return list(dict.fromkeys(bad))


def detect_answer_placeholders(answer: str) -> list[str]:
    if PLACEHOLDER_RE.search(answer or ""):
        return ["(context의 고유 주제) placeholder"]
    return []


def citation_doc_manifest(chunks: list[Any]) -> str:
    by_file: dict[str, list[int]] = {}
    for i, c in enumerate(chunks, start=1):
        fn = str(getattr(c, "file_name", "") or getattr(c, "doc_id", ""))
        by_file.setdefault(fn, []).append(i)

    allow_lines = [f"- {fn}" for fn in sorted(by_file)]
    map_lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        fn = str(getattr(c, "file_name", "") or "")
        page = getattr(c, "page_number", "?")
        body = strip_metadata_prefix(getattr(c, "text", ""))[:100].replace("\n", " ")
        map_lines.append(f"[{i}] → **{fn}** p{page} | {body}")

    return (
        "**인용 허용 문서 (아래 file_name만 §1·§2·§4에 쓸 것 — 목록 밖 DNV-RP/ RU-SHIP 등 번호 창작 금지):**\n"
        + "\n".join(allow_lines)
        + "\n\n**Citation 번호 ↔ 문서 (답변의 [N]은 반드시 해당 file_name 내용만 서술):**\n"
        + "\n".join(map_lines)
    )


def _load_page_texts(chunks_dir: Path, doc_id: str, page: int) -> list[str]:
    from rag_eval_lib import load_chunks

    path = chunks_dir / doc_id / "chunks.jsonl"
    if not path.exists():
        return []
    out: list[str] = []
    for ch in load_chunks(path):
        try:
            pn = int(ch.get("page_number", -1))
        except (TypeError, ValueError):
            continue
        if pn != page:
            continue
        body = strip_metadata_prefix(str(ch.get("text") or ""))
        if len(body) >= 60 and body not in out:
            out.append(body)
    out.sort(key=len, reverse=True)
    return out


def enrich_chunk_text(chunk: Any, pool: list[Any], chunks_dir: Path | None) -> str:
    page = getattr(chunk, "page_number", None)
    doc_id = str(getattr(chunk, "doc_id", ""))
    parts: list[str] = []

    if page is not None:
        for c in pool:
            if str(c.doc_id) == doc_id and c.page_number == page:
                body = strip_metadata_prefix(getattr(c, "text", ""))
                if body and body not in parts:
                    parts.append(body)
        if chunks_dir is not None:
            for body in _load_page_texts(chunks_dir, doc_id, int(page)):
                if body not in parts:
                    parts.append(body)

    merged = "\n\n".join(parts)
    current = getattr(chunk, "text", "") or ""
    if len(merged) > len(strip_metadata_prefix(current)):
        meta_lines = []
        for line in current.splitlines():
            if METADATA_LINE_RE.match(line.strip()):
                meta_lines.append(line)
                break
        prefix = "\n".join(meta_lines)
        return f"{prefix}\n{merged}".strip() if prefix else merged
    return current


def enrich_rule_lookup_chunks(
    retrieved: list[Any],
    pool: list[Any],
    *,
    chunks_dir: Path | None,
    row: dict,
) -> list[Any]:
    """Drop cross-ref tables; merge same-page text for substantive LLM context."""
    if str(row.get("category", "")) != "rule_lookup":
        return retrieved

    out: list[Any] = []
    used_ids: set[str] = set()

    for c in retrieved:
        if is_crossref_table_chunk(c) or getattr(c, "is_catalog_table", False):
            continue
        text = enrich_chunk_text(c, pool, chunks_dir)
        nc = replace(c, text=text)
        out.append(nc)
        used_ids.add(str(c.chunk_id))

    target = max(len(retrieved), 6)
    if len(out) < min(3, target):
        for c in sorted(pool, key=lambda x: float(getattr(x, "distance", 1.0))):
            if str(c.chunk_id) in used_ids or is_crossref_table_chunk(c) or getattr(c, "is_catalog_table", False):
                continue
            text = enrich_chunk_text(c, pool, chunks_dir)
            if chunk_body_len(type("_", (), {"text": text})()) < 70:
                continue
            out.append(replace(c, text=text))
            used_ids.add(str(c.chunk_id))
            if len(out) >= target:
                break

    return out[: max(len(retrieved), 1)]


def rule_lookup_answer_warnings(answer: str, chunks: list[Any]) -> list[str]:
    warnings: list[str] = []
    files = allowed_file_names(chunks)
    for code in detect_hallucinated_doc_codes(answer, files):
        warnings.append(f"답변에 검색되지 않은 문서번호 '{code}' — context file_name에 없음")
    for ph in detect_answer_placeholders(answer):
        warnings.append(f"답변에 placeholder '{ph}' 포함 — 재생성 필요")
    return warnings
