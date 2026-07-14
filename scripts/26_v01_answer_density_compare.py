"""Generate V01 before/after answer density comparison artifact."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rag_answer_lib import DEFAULT_OLLAMA_BASE, DEFAULT_OLLAMA_MODEL
from rag_eval_lib import load_questions
from rag_inprocess import DEFAULT_INDEX_DIR, DEFAULT_UNIFIED, run_full_inprocess
from ollama_warmup import bootstrap_all_resources

BEFORE_PATH = _ROOT / "data/eval/v01_answer_before.md"
AFTER_PATH = _ROOT / "data/eval/v01_answer_after.md"
OUT_PATH = _ROOT / "data/eval/v01_answer_density_compare.md"
QID = "V01"

SHALLOW_ENDINGS = re.compile(
    r"(논의되었습니다|정리되었습니다|승인되었습니다|포함됩니다|"
    r"검토가 있었|설명이 필요|필요합니다|필요함)\.?\s*$"
)
IMPACT_MARKERS = re.compile(
    r"(따라서|그 결과|이는 .{3,60}(연결|의미|영향|요구|체계|프레임))"
)
KEYWORD_LIST_MARKERS = re.compile(r"↔|/[^/].*,.*,|·.*·.*·")
KEYWORD_LIST_HEURISTIC = re.compile(
    r"^(?:\*\*[^*]+\*\*:?\s*)?"
    r"(?:[A-Za-z0-9 /·,↔\-]+(?:themes|verification|compliance|Guidelines|Framework)){0,3}"
    r"[A-Za-z /·,↔\-]{20,}$"
)


def _is_keyword_list_bullet(text: str) -> bool:
    t = text.replace("**", "").strip()
    if "↔" in t:
        return True
    if KEYWORD_LIST_MARKERS.search(t):
        return True
    # comma-heavy short clauses without Korean sentence endings
    if t.count(",") >= 2 and not re.search(r"(이다|한다|된다|있다|해야)\.", t):
        return True
    if "논의 및" in t and len(t) < 120 and "따라서" not in t and "이는" not in t:
        return True
    return False


def _count_keyword_list(bullets: list[str]) -> int:
    return sum(1 for b in bullets if _is_keyword_list_bullet(b))
SECTION3_TAGS = re.compile(r"\[(미확정 규제|해석 근거|선급별 상이 요구)\]|\*\*(미확정 규제|해석 근거|선급별 상이 요구)\*\*")
MUST_COVER = ["MEPC 84", "GHG", "emissions", "IMO Net-Zero Framework", "MARPOL Annex VI"]


def _load_v01() -> dict:
    for row in load_questions(_ROOT / "data/eval/pilot_validation_questions.jsonl"):
        if row.get("question_id") == QID:
            return row
    raise KeyError(QID)


def _generate_after(row: dict) -> str:
    boot = bootstrap_all_resources(
        DEFAULT_OLLAMA_MODEL,
        DEFAULT_OLLAMA_BASE,
        DEFAULT_UNIFIED,
        DEFAULT_INDEX_DIR,
        force_llm_warm=False,
    )
    out = run_full_inprocess(
        row,
        collection=boot["collection"],
        embed_model=boot["embed_model"],
        manifest=boot["manifest"],
        llm_model=DEFAULT_OLLAMA_MODEL,
        ollama_base=DEFAULT_OLLAMA_BASE,
        latency_mode="accurate",
        start_type="warm",
        use_rerank=True,
        top_k=10,
        fetch_k=120,
        max_doc=3,
        max_docs=10,
        auto_llm_warm=False,
    )
    return out["answer"]


def _section_bullets(text: str, section_num: int) -> list[str]:
    pat = (
        rf"(?:##|###)\s*\*?\*?{section_num}\)[^\n]*\n(.*?)"
        rf"(?=(?:##|###)\s*\*?\*?[1-4]\)|\n##\s*근거|\Z)"
    )
    m = re.search(pat, text, re.S | re.I)
    if not m:
        return []
    bullets: list[str] = []
    for ln in m.group(1).splitlines():
        s = ln.strip()
        if s.startswith("- ") or s.startswith("* "):
            bullets.append(s[2:].strip())
        elif s.startswith("    * ") or s.startswith("\t* "):
            bullets.append(s.lstrip("* ").strip())
    return bullets


def _count_shallow(bullets: list[str]) -> int:
    return sum(
        1
        for b in bullets
        if SHALLOW_ENDINGS.search(b.replace("**", ""))
        or "논의되었" in b
        or "설명이 필요" in b
    )


def _count_impact(bullets: list[str]) -> int:
    return sum(1 for b in bullets if IMPACT_MARKERS.search(b))


def _missing_must_cover(text: str) -> list[str]:
    lower = text.lower()
    missing = []
    for term in MUST_COVER:
        if term.lower() not in lower and term not in text:
            missing.append(term)
    return missing


def _improved_examples(before: str, after: str) -> list[str]:
    after_b1 = _section_bullets(after, 1)[:5]
    before_b1 = _section_bullets(before, 1)[:5]
    examples = []
    for i, ab in enumerate(after_b1):
        if IMPACT_MARKERS.search(ab) and (i >= len(before_b1) or not IMPACT_MARKERS.search(before_b1[i])):
            examples.append(ab[:400])
    return examples[:4]


def _insufficient_items(after: str) -> list[str]:
    items = []
    for b in _section_bullets(after, 3):
        if "검색 결과 내 확인 불가" in b or "확인 불가" in b:
            items.append(b[:350])
    items.extend(f"must_cover 미포함: {t}" for t in _missing_must_cover(after))
    return items[:8]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-generate", action="store_true")
    args = ap.parse_args()

    if not BEFORE_PATH.exists():
        raise FileNotFoundError(f"Missing before answer: {BEFORE_PATH}")

    before = BEFORE_PATH.read_text(encoding="utf-8")
    row = _load_v01()
    if args.skip_generate and AFTER_PATH.exists():
        after = AFTER_PATH.read_text(encoding="utf-8")
    else:
        print("Generating improved V01 answer (Accurate / multi-doc)...")
        after = _generate_after(row)
        AFTER_PATH.write_text(after, encoding="utf-8")

    b1, a1 = _section_bullets(before, 1), _section_bullets(after, 1)
    b2, a2 = _section_bullets(before, 2), _section_bullets(after, 2)
    b3, a3 = _section_bullets(before, 3), _section_bullets(after, 3)

    metrics = {
        "before_s1_bullets": len(b1),
        "after_s1_bullets": len(a1),
        "before_shallow_s1": _count_shallow(b1),
        "after_shallow_s1": _count_shallow(a1),
        "before_impact_s1": _count_impact(b1),
        "after_impact_s1": _count_impact(a1),
        "before_keyword_list_s1": _count_keyword_list(b1),
        "after_keyword_list_s1": _count_keyword_list(a1),
        "before_keyword_list_s2": _count_keyword_list(b2),
        "after_keyword_list_s2": _count_keyword_list(a2),
        "before_impact_s2": _count_impact(b2),
        "after_impact_s2": _count_impact(a2),
        "after_s3_tagged": sum(1 for b in a3 if SECTION3_TAGS.search(b)),
    }

    lines = [
        "# V01 답변 밀도 개선 비교",
        "",
        f"질문: {row['question']}",
        "",
        "## 정량 비교",
        "",
        "| 지표 | 개선 전 | 개선 후 |",
        "|------|---------|---------|",
        f"| §1 bullet 수 | {metrics['before_s1_bullets']} | {metrics['after_s1_bullets']} |",
        f"| §1 표면적 종결 문장 | {metrics['before_shallow_s1']} | {metrics['after_shallow_s1']} |",
        f"| §1 키워드 나열형 bullet | {metrics['before_keyword_list_s1']} | {metrics['after_keyword_list_s1']} |",
        f"| §2 키워드 나열형 bullet | {metrics['before_keyword_list_s2']} | {metrics['after_keyword_list_s2']} |",
        f"| §1 영향 서술(따라서/이는/그 결과) | {metrics['before_impact_s1']} | {metrics['after_impact_s1']} |",
        f"| §2 영향 서술 | {metrics['before_impact_s2']} | {metrics['after_impact_s2']} |",
        f"| §3 분류 태그 사용 | — | {metrics['after_s3_tagged']} |",
        "",
        "## 기존 답변",
        "",
        before.strip(),
        "",
        "## 개선 답변",
        "",
        after.strip(),
        "",
        "## 개선된 문장 예시",
        "",
    ]
    for ex in _improved_examples(before, after):
        lines.append(f"- {ex}")
    if not _improved_examples(before, after):
        lines.append("- (자동 추출 없음 — §1 상위 bullet 참조)")

    lines.extend(["", "## 키워드 나열형 bullet (개선 후 잔존)", ""])
    kw_after = [b for b in a1 + a2 if _is_keyword_list_bullet(b)]
    if kw_after:
        for b in kw_after[:6]:
            lines.append(f"- {b[:300]}")
    else:
        lines.append("- §1·§2에서 키워드 나열형(↔·쉼표 나열·미완성 문장) 자동 탐지 없음")

    lines.extend(["", "## 여전히 근거 부족한 항목", ""])
    insuf = _insufficient_items(after)
    if insuf:
        for it in insuf:
            lines.append(f"- {it}")
    else:
        lines.append("- §3 또는 must_cover 기준상 자동 탐지된 부족 항목 없음")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
