"""Parse table QA questions into structured lookup slots."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from table_normalize_lib import (
    ENTITY_ALIASES,
    MATERIAL_GRADE_RE,
    expand_entity_aliases,
    extract_paren_aliases,
    extract_units,
    normalize_material_grade,
    normalize_token,
)

QUERY_TYPES = (
    "table_lookup",
    "row_lookup",
    "cell_lookup",
    "note_lookup",
    "condition_lookup",
)

ROW_DOMAIN_TERMS = (
    "평형수탱크",
    "화물창",
    "화물탱크",
    "연료유탱크",
    "빌지저장탱크",
    "이중저탱크",
    "디프탱크",
    "피크탱크",
    "기관실",
    "주위벽",
    "탱크",
    "구역",
)

INSPECTION_TERMS = (
    "제1차 정기검사",
    "제2차 정기검사",
    "제3차 정기검사",
    "제4차 및 이후 정기검사",
    "정기검사",
    "reporting",
    "검사차수",
    "검사 보고",
)

MECHANICAL_TERMS = ("항복", "인장", "연신", "충격", "흡수에너지", "기계적 성질", "n/mm")
CHEMISTRY_TERMS = ("화학", "함량", "허용", "한도", "성분", "탈산", "이하", "이상")
NOTE_TERMS = ("비고", "주석", "footnote", "각주")
CONDITION_TERMS = ("선령", "두께", "조건", "구간", "년", "mm", "이상", "이하", "초과", "미만")
SUMMARY_TERMS = ("구조", "어떤", "설명", "주요", "개요", "매트릭스")


@dataclass
class ParsedTableQuery:
    raw_question: str
    query_type: str = "cell_lookup"
    row_entities: list[str] = field(default_factory=list)
    column_entities: list[str] = field(default_factory=list)
    table_topic_candidates: list[str] = field(default_factory=list)
    unit_candidates: list[str] = field(default_factory=list)
    condition_candidates: list[str] = field(default_factory=list)
    keyword_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(t for t in items if t))


def _infer_column_entities(question: str) -> list[str]:
    cols: list[str] = []
    for term in INSPECTION_TERMS:
        if term.lower() in question.lower() or term in question:
            cols.append(term)
    for _base, aliases in extract_paren_aliases(question):
        cols.extend(aliases[:3])
    for canon, forms in ENTITY_ALIASES.items():
        if len(canon) <= 3 and canon.isalpha():
            for form in forms:
                if form in question or form.lower() in question.lower():
                    cols.append(canon)
                    break
    # Age-range column patterns
    age_patterns = [
        (r"15\s*년\s*(을\s*)?(초과|넘)", "15년< 선령"),
        (r"10\s*년\s*초과.*15\s*년|10\s*[~\-–]\s*15\s*년", "10년< 선령≤15년"),
        (r"5\s*년\s*초과.*10\s*년|5\s*[~\-–]\s*10\s*년", "5년< 선령≤10년"),
    ]
    for pat, label in age_patterns:
        if re.search(pat, question):
            cols.append(label)
    return _dedupe(cols)


def _infer_row_entities(question: str) -> list[str]:
    rows: list[str] = []
    for m in MATERIAL_GRADE_RE.finditer(question):
        rows.append(normalize_material_grade(m.group(0)))
        rows.append(normalize_token(m.group(0)))
    for term in ROW_DOMAIN_TERMS:
        if term in question:
            rows.append(term)
    return _dedupe(rows)


def _infer_table_topics(question: str, cols: list[str], rows: list[str]) -> list[str]:
    topics: list[str] = []
    q = question.lower()
    if any(t in question for t in CHEMISTRY_TERMS) or any(
        c in {"C", "S", "P", "MN", "SI"} for c in cols
    ):
        topics.extend(["화학성분", "chemical_composition"])
    if any(t in question for t in MECHANICAL_TERMS):
        topics.extend(["기계적성질", "mechanical_properties"])
    if any(t in question for t in INSPECTION_TERMS) or "reporting" in q:
        topics.extend(["정기검사", "inspection", "reporting"])
    if "선령" in question:
        topics.extend(["선령", "age_range"])
    if "열처리" in question or "로트" in question:
        topics.extend(["열처리", "lot_treatment"])
    if "용접" in question or "시험재" in question:
        topics.extend(["용접", "시험재료", "welding"])
    if "치수" in question or "두께" in question:
        topics.extend(["치수", "dimension"])
    if any(t in question for t in NOTE_TERMS):
        topics.append("비고")
    if not topics and any(t in question for t in SUMMARY_TERMS):
        topics.append("table_overview")
    return _dedupe(topics)


def _infer_query_type(question: str, cols: list[str], rows: list[str], topics: list[str]) -> str:
    if any(t in question for t in NOTE_TERMS):
        return "note_lookup"
    if any(t in question for t in SUMMARY_TERMS) and not cols and not rows:
        return "table_lookup"
    if rows and not cols:
        return "row_lookup"
    if any(t in question for t in CONDITION_TERMS) and "선령" in question:
        return "condition_lookup"
    if cols and rows:
        return "cell_lookup"
    if cols:
        return "column_lookup" if "column_lookup" in QUERY_TYPES else "cell_lookup"
    if rows:
        return "row_lookup"
    if topics:
        return "table_lookup"
    return "cell_lookup"


def parse_table_query(question: str) -> ParsedTableQuery:
    q = normalize_token(question)
    row_entities = _infer_row_entities(q)
    column_entities = _infer_column_entities(q)
    table_topic_candidates = _infer_table_topics(q, column_entities, row_entities)
    unit_candidates = extract_units(q)
    condition_candidates = [t for t in CONDITION_TERMS if t in q]
    keyword_terms = _dedupe(
        row_entities
        + column_entities
        + table_topic_candidates
        + [t for t in q.split() if len(t) >= 2][:12]
    )
    query_type = _infer_query_type(q, column_entities, row_entities, table_topic_candidates)
    return ParsedTableQuery(
        raw_question=q,
        query_type=query_type,
        row_entities=row_entities,
        column_entities=column_entities,
        table_topic_candidates=table_topic_candidates,
        unit_candidates=unit_candidates,
        condition_candidates=condition_candidates,
        keyword_terms=keyword_terms,
    )


def build_embed_query(parsed: ParsedTableQuery) -> str:
    parts = (
        parsed.table_topic_candidates[:6]
        + parsed.row_entities[:6]
        + parsed.column_entities[:6]
        + parsed.unit_candidates[:3]
        + [parsed.raw_question]
    )
    return " ".join(dict.fromkeys(p for p in parts if p)).strip()
